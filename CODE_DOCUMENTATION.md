# smtpBERT Code Documentation

## Project Overview

smtpBERT is an AI-powered SMTP phishing detection gateway that uses DistilBERT and XGBoost to classify URLs in emails as malicious or benign.

---

## Directory Structure

```
smtpBERT/
├── src/
│   ├── app.py                    # Streamlit dashboard
│   ├── local_pipeline.py          # CLI menu system
│   ├── gateway/                   # SMTP gateway components
│   │   ├── gateway.py            # SMTP server
│   │   ├── handler.py            # Email processing handler
│   │   ├── detector.py           # URL classification
│   │   ├── database.py           # SQLite storage
│   │   ├── config.py             # Gateway configuration
│   │   ├── alerts.py             # Mailgun alert system
│   │   ├── monitoring.py         # Metrics tracking
│   │   └── policy.py             # Blocking policies
│   └── models/                    # ML training and evaluation
│       ├── baseline_models.py     # LR, RF, XGBoost training
│       ├── distilbert_trainer.py  # DistilBERT fine-tuning
│       ├── cross_validation.py   # 10-fold CV
│       ├── compute_*.py          # Metrics computation scripts
│       └── config.py             # Model configuration
├── data/                          # Training/test datasets
├── logs/                          # Metrics and figures
└── phishing_model/                # Trained DistilBERT model
```

---

## src/local_pipeline.py

The main CLI entry point providing an interactive menu for all workflows.

### Functions

#### Training Functions

```python
def train_baseline() -> None
```
Trains the three baseline ML models (LogisticRegression, RandomForest, XGBoost) using `baseline_models.main()`.

```python
def train_distilbert(epochs, batch_size, learning_rate) -> None
```
Fine-tunes DistilBERT on URL data. Overrides config values before calling `distilbert_trainer.train_model()`.

```python
def test_distilbert() -> None
```
Runs sample predictions with the trained DistilBERT model.

```python
def evaluate_distilbert(retrain, epochs, batch_size, learning_rate) -> None
```
Evaluates DistilBERT on the test set, optionally retraining first.

```python
def run_baseline_ten_fold_cv() -> None
```
Runs 10-fold cross-validation for baseline models only.

```python
def run_ten_fold_comparison() -> None
```
Runs full 10-fold CV comparing all models (DistilBERT + LR + RF + XGBoost). Saves results to `logs/model_comparison/ten_fold_cv_results.json`.

#### Dashboard & Gateway

```python
def run_dashboard() -> None
```
Launches Streamlit web UI via `streamlit run src/app.py`.

```python
def run_gateway(smtp_port) -> None
```
Starts the SMTP phishing detection server. Sets `SMTP_PORT` env var and calls `gateway.run_gateway()`.

```python
def send_mock_email(smtp_host, smtp_port, sender, recipient, body, subject) -> None
```
Sends a test email via SMTP using `smtplib.SMTP`.

```python
def mock_gateway_test(smtp_host, smtp_port, sender, recipient, body, auto_start_gateway) -> None
```
Full mock email test workflow. Optionally starts gateway as subprocess, sends email, then terminates gateway.

#### Metrics Collection

```python
def gather_dashboard_metrics(retrain_distilbert, run_baseline_cv, epochs, batch_size, learning_rate) -> None
```
Collects all metrics for the dashboard by running evaluation scripts and merging results into `logs/model_comparison/model_comparison.json`.

```python
def update_dashboard_with_cv_results(cv_results) -> None
```
Merges 10-fold CV summary into the dashboard's model comparison JSON file.

#### Menu System

```python
def _prompt_int(prompt, default) -> Optional[int]
def _prompt_float(prompt, default) -> Optional[float]
def _prompt_text(prompt, default) -> str
```
Helper functions for interactive CLI prompts with defaults.

```python
def run_menu() -> None
```
Main interactive menu loop displaying all available options.

```python
def main() -> None
def build_parser() -> argparse.ArgumentParser
```
CLI argument parser supporting both menu mode and direct commands like `python -m src.local_pipeline train-baseline`.

---

## src/gateway/gateway.py

SMTP server that intercepts emails and scans URLs.

### Key Components

```python
def run_gateway() -> None
```
Main entry point:
1. Initializes SQLite database via `init_db()`
2. Creates `PhishingSMTPHandler`
3. Creates a `Controller` binding to `SMTP_HOST:SMTP_PORT`
4. Starts the server and runs forever (or until Ctrl+C)

```python
handler = PhishingSMTPHandler()
controller = Controller(handler, hostname=SMTP_HOST, port=SMTP_PORT)
controller.start()
```

On Windows, defaults to `127.0.0.1:25` to avoid binding issues.

---

## src/gateway/handler.py

Processes incoming SMTP emails and extracts/ classifies URLs.

### Email Processing Flow

```
SMTP Connection → MAIL FROM → RCPT TO → DATA → Email Body
                                                      ↓
                                            Extract URLs
                                                      ↓
                                            For each URL:
                                                      ↓
                                            detector.predict(url)
                                                      ↓
                                            If malicious → 550 Reject
                                            If clean → 250 Accept
```

### Key Functions

```python
class PhishingSMTPHandler
```
继承 `aiosmtpd.SMTPHandler`，处理每个 SMTP 命令。

```python
async def handle_message(self, message) -> str
```
Called after DATA command receives full email body.

**Process:**
1. Extracts sender (`mail_from`) and recipient (`rcpt_tos`)
2. Parses email body with `email.parser`
3. Extracts all URLs using `extract_urls()`
4. For each URL, calls `detector.predict(url)`
5. If any URL is malicious → returns 550 rejection message
6. If all clean → returns 250 acceptance

```python
def extract_urls(body) -> list
```
使用正则表达式从邮件正文中提取 URL。

### URL Blocking Decision

```python
result = detector.predict(url)
# result = {
#   "label": "Phishing",
#   "blocked": True,
#   "probability": 0.97,
#   "decision_reason": "BLOCKED: Malicious URL detected (Phishing, confidence=1.00)"
# }
```

---

## src/gateway/detector.py

Loads the trained ML model and classifies URLs.

### Key Components

```python
@st.cache_resource
def get_detector() -> PhishingDetector
```
Caches the detector instance to avoid reloading model on every prediction.

```python
class PhishingDetector
```

```python
def __init__(self)
```
Loads the following on initialization:
- DistilBERT model from `phishing_model/`
- Tokenizer
- Gateway config (threshold, blocked labels)
- URL cache (Redis-like dict to avoid re-classifying same URL)

```python
def predict(self, url) -> dict
```

**Process:**
1. Check URL cache → return cached result if exists
2. Preprocess URL (add http:// if missing)
3. Tokenize with DistilBERT tokenizer
4. Run inference via `self.model(input_ids, attention_mask)`
5. Extract probabilities for 4 classes: [Benign, Phishing, Malware, Defacement]
6. Apply thresholding logic based on `decision_threshold` and class weights
7. Cache result
8. Return full prediction dict

```python
def _apply_threshold(self, probs, label) -> bool
```
Determines if a URL should be blocked based on:
- Class probability > `decision_threshold`
- Class not in `allowed_classes`
- Class weight * probability > minimum score

---

## src/models/baseline_models.py

Trains and evaluates LogisticRegression, RandomForest, and XGBoost.

### Feature Extraction

```python
def extract_url_features(url) -> list
```
提取 28 个特征:

| Feature | Description |
|---------|-------------|
| `url_length` | Total character count |
| `dot_count` | Number of dots |
| `dash_count` | Number of dashes |
| `slash_count` | Number of slashes |
| `digit_count` | Number of digits |
| `digit_ratio` | Digit count / URL length |
| `special_char_count` | [@?=&%#_~] count |
| `has_pct` | Contains % encoding |
| `domain_length` | Domain name length |
| `subdomain_count` | Number of subdomain parts |
| `suspicious_tld` | Is in SUSPICIOUS_TLDS set |
| `has_ip` | Contains IPv4 address |
| `domain_entropy` | Shannon entropy of domain |
| `has_brand` | Contains brand name |
| `tld_length` | TLD character count |
| `is_shortener` | Is URL shortener |
| `url_entropy` | Shannon entropy of full URL |
| `path_length` | Path segment length |
| `query_length` | Query string length |
| `query_param_count` | Number of & parameters |
| `path_slash_count` | Slashes in path |
| `vowel_ratio` | Vowels / URL length |
| `max_word_length` | Longest alphabetic word |
| `has_https` | Starts with https |
| `has_at` | Contains @ symbol |
| `has_double_slash` | Contains // after http |
| `multiple_http` | More than one http |
| `is_base64` | Contains base64 pattern |
| `is_url_encoded` | Contains %XX encoding |

### Model Training

```python
def prepare_features(train_urls, val_urls)
```
1. Fits TF-IDF vectorizer on training URLs (char n-grams 2-4, max 5000 features)
2. Transforms both train and val URLs
3. Extracts handcrafted features for both
4. Concatenates TF-IDF + handcrafted features
5. Returns scaler, TF-IDF vectorizer, and feature matrices

```python
def run_logistic_regression(X_train, X_val, y_train, y_val) -> tuple
```
Trains LogisticRegression with balanced class weights. Returns (model, scaler, metrics_dict).

```python
def run_random_forest(X_train, X_val, y_train, y_val) -> tuple
```
Trains RandomForest with 100 estimators and balanced class weights.

```python
def run_xgboost(X_train, X_val, y_train, y_val) -> tuple
```
Trains XGBClassifier with scale_pos_weight for class imbalance.

### Main Entry

```python
def main() -> None
```
Train all three models on training data, evaluate on test set, save models and results.

---

## src/models/distilbert_trainer.py

Fine-tunes DistilBERT (distilbert-base-uncased) for URL classification.

### Configuration (from config.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| NUM_EPOCHS | 1 | Training epochs |
| BATCH_SIZE | 8 | Batch size |
| LEARNING_RATE | 2e-5 | AdamW learning rate |
| MAX_SEQ_LENGTH | 128 | Tokenizer max length |
| WEIGHT_DECAY | 0.01 | L2 regularization |
| WARMUP_RATIO | 0.06 | Warmup proportion |
| FP16 | False | Mixed precision training |

### Key Functions

```python
def load_data() -> tuple
```
Loads train/test CSVs and creates HuggingFace `Dataset` objects.

```python
def tokenize_function(examples) -> dict
```
Tokenizes URLs using DistilBERT tokenizer with padding and truncation.

```python
def compute_metrics(eval_pred) -> dict
```
Computes accuracy, precision, recall, F1 (macro) for evaluation.

```python
def train_model() -> None
```
Main training loop:
1. Load data
2. Tokenize
3. Initialize Trainer with TrainingArguments
4. Train with per-class weights for imbalance
5. Save model + tokenizer to `phishing_model/`

```python
def evaluate_saved_model() -> None
```
Loads saved model and runs comprehensive evaluation:
- Metrics on test set (accuracy, precision, recall, F1, ROC-AUC)
- Confusion matrix
- Per-class recall (quality gates)
- Saves to `training_metrics.json`

```python
def test_model() -> None
```
Quick sample predictions to verify model works.

---

## src/models/cross_validation.py

Implements stratified 10-fold cross-validation for all models.

### Key Functions

```python
def run_kfold_cross_validation(model_type, n_splits, random_state) -> dict
```
Main CV function:

```
For each fold (1-10):
  1. Split data 80/20 (stratified)
  2. For each model (LR, RF, XGB, optionally DistilBERT):
     - Train on 20k samples
     - Evaluate on 5k samples
     - Store metrics and confusion matrix
  3. Update progress file

Aggregate results:
  - Mean and Std for each metric
  - Per-fold values stored
  - Saves to ten_fold_cv_results.json
```

```python
def _aggregate_fold_results(fold_results, model_type) -> dict
```
Computes mean ± std across folds for accuracy, precision, recall, F1, ROC-AUC.

### Constants

| Constant | Value | Description |
|----------|-------|-------------|
| SAMPLES_PER_FOLD | 25000 | Total samples per fold |
| TRAIN_RATIO | 0.8 | Training proportion |
| VAL_RATIO | 0.2 | Validation proportion |

---

## src/models/compute_inference_metrics.py

Benchmarks inference latency, memory usage, and generates ROC curves.

### Key Functions

```python
def benchmark_latency(models, test_urls, n_runs) -> dict
```
Measures average ms per URL over multiple passes for each model.

```python
def benchmark_memory() -> dict
```
Measures GPU VRAM for DistilBERT, model file size for baselines.

```python
def compute_roc_curves(models, test_data, n_samples) -> dict
```
Generates one-vs-rest ROC curves per class using sklearn.

### Output Files

| File | Description |
|------|-------------|
| `inference_metrics.json` | Latency (ms/URL) and memory (MB) |
| `roc_curves.json` | FPR, TPR, AUC per class per model |
| `model_sizes.json` | File sizes and parameter counts |

---

## src/models/compute_xgboost_shap.py

Generates SHAP explainability plots for XGBoost.

### Key Functions

```python
def extract_url_features(url) -> list
```
Same 28 features as baseline_models.py for consistency.

```python
def load_fixed_eval_indices() -> list
```
Loads pre-selected 5000 test indices for consistent evaluation.

```python
def main() -> None
```

```
1. Load XGBoost model from models/xgboost.pkl
2. Load TF-IDF vectorizer
3. Extract features from test URLs (TF-IDF + handcrafted)
4. Create SHAP TreeExplainer
5. Compute SHAP values for all samples
6. Generate per-class summary plots
7. Generate combined summary plot
8. Save to logs/figures/
```

### Output Files

| File | Description |
|------|-------------|
| `xgboost_shap_summary.png` | All classes combined |
| `xgboost_shap_benign.png` | Benign class |
| `xgboost_shap_phishing.png` | Phishing class |
| `xgboost_shap_malware.png` | Malware class |
| `xgboost_shap_defacement.png` | Defacement class |

---

## src/models/compute_distilbert_shap.py

Generates SHAP explainability for DistilBERT using KernelSHAP.

### Key Functions

```python
def main() -> None
```

```
1. Load DistilBERT model and tokenizer
2. Extract tokenized inputs from test URLs
3. Use SHAP KernelExplainer (slower but works with any model)
4. Compute SHAP values for a sample of URLs
5. Generate per-class token importance plots
6. Save to logs/figures/
```

**Note:** This takes 15-20 minutes due to KernelSHAP's sampling approach.

---

## src/app.py (Streamlit Dashboard)

Four-tab web dashboard for URL testing and model evaluation.

### Tab 1: URL Tester

```python
def _render_tester() -> None
```
Simple URL input form that calls `detector.predict()` and displays:
- Prediction label with color coding
- Confidence score
- Per-class probability bar chart
- Decision reason

### Tab 2: Evaluation

```python
def _render_evaluation() -> None
```
Displays metrics from saved JSON files:

- **Inference Performance**: Latency and memory comparison
- **ROC Curves**: One-vs-Rest per class
- **Holdout Comparison**: Bar charts for F1, Precision, Recall, ROC-AUC
- **Confusion Matrices**: Interactive heatmaps per model
- **10-Fold CV**: Box plots with scatter points showing per-fold values

```python
def _render_cv_results(cv_results) -> None
```
The main CV visualization function:
- Uses `go.Box` with `boxpoints="all"` for box + scatter combo
- Dynamic y-axis range based on actual data values
- Summary statistics table with mean ± std

### Tab 3: Methods

```python
def _render_methods() -> None
```
Displays research methodology:
- Data preprocessing and feature engineering
- Experimental setup
- Class imbalance handling
- Research questions mapping

### Tab 4: XAI / SHAP

```python
def _render_xai() -> None
```
Displays SHAP plots from `logs/figures/`:
- Toggle between XGBoost and DistilBERT SHAP
- Per-class plots for all 4 categories
- Interpretation guide

### Helper Functions

```python
def _load_json(path) -> dict
```
Safely loads JSON, returns empty dict if file missing.

```python
def _specificity_from_confusion(confusion) -> float
```
Computes average specificity across 4 classes.

```python
def _render_heatmap(confusion, title) -> None
```
Creates interactive Plotly heatmap for confusion matrices.

```python
def _make_bar_chart(df, metric, title) -> go.Figure
```
Creates Plotly bar chart with dynamic y-axis.

---

## src/gateway/database.py

SQLite storage for blocked URLs and metrics.

### Key Functions

```python
def init_db() -> sqlite3.Connection
```
Creates/opens `phishing_data.db` and enables WAL mode for performance.

```python
def log_url(url, label)
```
Inserts a blocked URL into the `malicious_urls` table.

### Schema

```sql
CREATE TABLE IF NOT EXISTS malicious_urls (
    url TEXT PRIMARY KEY,
    label INTEGER,      -- 0=Benign, 1=Phishing, 2=Malware, 3=Defacement
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## src/gateway/alerts.py

Sends alerts via Mailgun when malicious URLs are detected.

### Key Functions

```python
def send_alert(url, label, confidence, sender, recipient) -> bool
```
Sends an email via Mailgun SMTP with URL details.

### Configuration (from config.py)

```python
MAILGUN_SMTP_HOST = "smtp.mailgun.org"
MAILGUN_SMTP_PORT = 587
MAILGUN_LOGIN = ""      # Set via MAILGUN_LOGIN env var
MAILGUN_PASSWORD = ""   # Set via MAILGUN_PASSWORD env var
MAILGUN_FROM = "smtpbert@smtpbert.com"
ALERT_TO = ""           # Set via ALERT_TO env var
```

---

## src/gateway/monitoring.py

Tracks gateway metrics and performance.

### Key Functions

```python
class GatewayMetrics
```
Tracks:
- Emails processed
- URLs extracted
- URLs blocked
- Processing time
- Per-model predictions

```python
def get_metrics_tracker() -> GatewayMetrics
```
Singleton pattern to share metrics across handlers.

---

## src/gateway/config.py

Configuration for the SMTP gateway.

| Variable | Default | Description |
|----------|---------|-------------|
| SMTP_HOST | 127.0.0.1 (Windows) / 0.0.0.0 (Linux) | Bind address |
| SMTP_PORT | 25 | SMTP port |
| DB_PATH | phishing_data.db | SQLite database |
| MAILGUN_* | Various | Mailgun SMTP settings |

On Windows, defaults to `127.0.0.1` because `0.0.0.0` causes binding errors with aiosmtpd.

---

## Usage Examples

### Train All Models
```bash
python -m src.local_pipeline train-baseline
python -m src.local_pipeline train-distilbert --epochs 3 --batch-size 8
```

### Run SMTP Gateway
```bash
python -m src.local_pipeline run-gateway --smtp-port 2525
```

### Test URL via Dashboard
```bash
python -m src.local_pipeline run-dashboard
# Then open http://localhost:8501
```

### Run 10-Fold Cross-Validation
```bash
python -m src.local_pipeline ten-fold-comparison
```

### Send Mock Email
```bash
python -m src.local_pipeline mock-email \
    --smtp-host 127.0.0.1 \
    --smtp-port 25 \
    --body "Click here: http://malicious-site.com" \
    --auto-start-gateway
```

### Generate SHAP Plots
```bash
python -m src.models.compute_xgboost_shap
python -m src.models.compute_distilbert_shap  # Takes 15-20 min
```

### Gather All Metrics
```bash
python -m src.local_pipeline gather-metrics --run-baseline-cv
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| SMTP_HOST | Override SMTP bind address |
| SMTP_PORT | Override SMTP port |
| MAILGUN_LOGIN | Mailgun API key |
| MAILGUN_PASSWORD | Mailgun password |
| ALERT_TO | Alert recipient email |
| FP16 | Enable mixed precision (1=enabled) |
| MIN_RECALL_PHISHING | Quality gate threshold (default 0.85) |
| MIN_RECALL_MALWARE | Quality gate threshold (default 0.80) |
| MIN_RECALL_DEFACEMENT | Quality gate threshold (default 0.75) |

---

## Model Performance Thresholds

| Metric | Minimum Required |
|--------|------------------|
| Phishing Recall | 85% |
| Malware Recall | 80% |
| Defacement Recall | 75% |

These are enforced as "quality gates" - models must pass these thresholds before deployment.