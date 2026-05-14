# smtpBERT - AI-Powered SMTP Phishing Detection Gateway

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/Transformer-DistilBERT-orange.svg" alt="Transformer">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/Classes-4%20(Benign%20|%20Phishing%20|%20Malware%20|%20Defacement)-purple.svg" alt="Classes">
</p>

smtpBERT is a dual-service system that combines a SMTP email gateway with an AI-powered URL classification engine to detect and block phishing, malware, and defacement attempts in incoming emails — in real-time.

You can access the prototype dashboard via this codespace github deployment: https://shiny-pancake-97jw5wq9vwx4f45q-8501.app.github.dev/
---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [4-Class Classification](#4-class-classification)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the SMTP Gateway](#running-the-smtp-gateway)
  - [Running the Dashboard](#running-the-dashboard)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Gateway Thresholds](#gateway-thresholds)
  - [Quality Gates](#quality-gates)
- [Training](#training)
- [API Reference](#api-reference)
- [Tech Stack](#tech-stack)
- [Performance Notes](#performance-notes)
- [License](#license)

---

## Overview

Traditional email gateways rely on static rules, blacklists, and pattern matching. smtpBERT replaces this with a fine-tuned DistilBERT model that understands the semantic structure of URLs and can identify malicious links even when they use obfuscation techniques like URL shorteners, homograph attacks, or brand impersonation.

The system processes email as it arrives at the SMTP level — before it hits the mail server — and can reject malicious emails with a 550 SMTP response code.

---

## Architecture

```
                            ┌─────────────────────────────┐
                            │        smtpBERT             │
                            │   SMTP Gateway (port 25)     │
                            └──────────────┬──────────────┘
                                           │
                          ┌────────────────┴────────────────┐
                          ▼                                 ▼
              ┌───────────────────────┐         ┌──────────────────────┐
              │   PhishingSMTPHandler  │         │   URL Cache (SQLite) │
              │  (extracts URLs from   │         │  (malicious URLs     │
              │   email bodies)        │         │   persist across     │
              └───────────┬───────────┘         │   restarts)           │
                          │                     └──────────────────────┘
                          ▼
              ┌───────────────────────┐
              │   PhishingDetector    │
              │  (classification with │
              │   threshold logic)    │
              └───────────┬───────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
    ┌───────────┐ ┌───────────┐ ┌───────────┐
    │ DistilBERT │ │  XGBoost  │ │ Logistic  │
    │  (Primary) │ │ Baseline  │ │ Regression│
    └───────────┘ └───────────┘ └───────────┘
            │
            ▼
    ┌───────────────────┐
    │ Streamlit Dashboard│
    │   (port 8500)      │
    │ - URL Tester       │
    │ - Benchmark Viewer │
    └───────────────────┘
```

### Data Flow

1. **Email arrives** at the SMTP gateway on port 25
2. The `PhishingSMTPHandler` parses the email body and extracts all URLs
3. Each URL is checked against the **SQLite cache** (malicious URLs from previous emails)
4. For new URLs, `PhishingDetector` runs inference via **DistilBERT**
5. The gateway applies **confidence and malicious-probability thresholds** to decide whether to block
6. If blocked, the email is rejected with a 550 status and an alert is sent via Mailgun
7. If clean, the email is accepted (250 OK)

---

## Features

- **Real-time SMTP filtering** — intercepts emails at the SMTP level before delivery
- **4-class URL classification** — DistilBERT classifies URLs as Benign, Phishing, Malware, or Defacement
- **Baseline model comparison** — Logistic Regression, Random Forest, and XGBoost baselines for comparison
- **Caching** — malicious URLs are cached in SQLite to avoid repeated inference
- **Live dashboard** — Streamlit UI for testing individual URLs and viewing results
- **Hot model reload** — detects new model files on disk and reloads without restart
- **Configurable thresholds** — per-class confidence thresholds and malicious probability gates
- **Alerting** — sends email alerts via Mailgun when malicious URLs are detected
- **Metrics tracking** — records latency, cache hits, prediction distributions, and decisions

---

## 4-Class Classification

| Label | Class     | Description                                                    |
|-------|-----------|----------------------------------------------------------------|
| 0     | Benign    | Legitimate, safe URL (e-commerce, official sites, etc.)        |
| 1     | Phishing  | Credential harvesting, account takeover attempts                |
| 2     | Malware   | Drive-by downloads, exploit kits, malicious payload delivery    |
| 3     | Defacement | Vandals, hacktivism, compromised website redirects              |

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for training; inference runs on CPU)
- Mailgun account (for alerting)

### Installation

```bash
# Clone the repository
git clone https://github.com/2241812/XGBert-SMTP-Gateway.git
cd XGBert-SMTP-Gateway

# Install dependencies
pip install -r requirements.txt

# Download or train a model (see Training section)
# Place your model files in phishing_model/ or src/models/phishing_model/

# Run the local_pipeline.py to see CLI for options to run the whole system (CLI Menu)
python -m src.local_pipeline
```

### Running the SMTP Gateway

```bash
# Default port 25 (requires root on Linux)
python -m src.gateway.gateway

# Custom port (e.g., 1025 for local testing)
SMTP_PORT=1025 python -m src.gateway.gateway
```

### Running the Dashboard

```bash
streamlit run src/app.py --server.port 8501
```

Then open http://localhost:8501 in your browser. The dashboard provides a URL tester tab where you can paste any URL and see the full classification breakdown.

---

## Project Structure

```
XGBert-SMTP-Gateway/
├── src/
│   ├── __init__.py
│   ├── app.py                          # Streamlit dashboard (URL Tester)
│   ├── local_pipeline.py              # Standalone inference pipeline
│   ├── gateway/
│   │   ├── gateway.py                  # SMTP server entry point
│   │   ├── handler.py                  # Email handler + URL extraction
│   │   ├── detector.py                 # PhishingDetector (threshold logic)
│   │   ├── config.py                   # Gateway configuration
│   │   ├── database.py                 # SQLite URL cache
│   │   ├── alerts.py                   # Mailgun alert sender
│   │   ├── monitoring.py               # Metrics tracker
│   │   └── policy.py                   # URL extraction + label mapping
│   └── models/
│       ├── config.py                   # Training hyperparameters
│       ├── data.py                     # Dataset loading + tokenization
│       ├── model_loader.py             # DistilBERT inference loader
│       ├── baseline_models.py          # Classical ML training (LR, RF, XGBoost)
│       ├── distilbert_trainer.py       # DistilBERT fine-tuning
│       ├── viz.py                      # Matplotlib visualizations
│       ├── callbacks.py                # TrainingMonitor + progress callback
│       ├── cross_validation.py         # K-fold cross-validation
│       ├── phishing_model/             # Trained DistilBERT model files
│       │   ├── pytorch_model.bin
│       │   ├── tokenizer.json
│       │   ├── vocab.txt
│       │   └── training_metrics.json
│       └── logs/                       # Training logs and experiment outputs
├── requirements.txt
├── run_smoke_tests.py
├── smtpBERT_Training.ipynb             # Google Colab training notebook
└── README.md
```

---

## Configuration

### Environment Variables

| Variable              | Default                        | Description                              |
|-----------------------|--------------------------------|------------------------------------------|
| `SMTP_HOST`           | `0.0.0.0`                      | Host to bind the SMTP server             |
| `SMTP_PORT`           | `25`                           | Port for SMTP server                     |
| `MODEL_DIR`           | `phishing_model`               | Path to model files                      |
| `DB_PATH`             | `phishing_data.db`             | SQLite cache path                        |
| `MAILGUN_SMTP_HOST`   | `smtp.mailgun.org`             | Mailgun SMTP server                      |
| `MAILGUN_SMTP_PORT`   | `587`                          | Mailgun SMTP port                        |
| `MAILGUN_LOGIN`       | (empty)                        | Mailgun authentication username          |
| `MAILGUN_PASSWORD`    | (empty)                        | Mailgun authentication password          |
| `MAILGUN_FROM`        | `smtpbert@smtpbert.com`        | Sender address for alerts                |
| `ALERT_TO`            | (empty)                        | Recipient for phishing alerts             |
| `MIN_RECALL_PHISHING`  | `0.85`                         | Minimum recall for Phishing class        |
| `MIN_RECALL_MALWARE`   | `0.80`                         | Minimum recall for Malware class         |
| `MIN_RECALL_DEFACEMENT`| `0.75`                        | Minimum recall for Defacement class      |

### Gateway Thresholds

The `PhishingDetector` uses two gating mechanisms:

```python
# From src/gateway/detector.py
self.malicious_threshold = 0.65        # block if P(malicious) >= this
self.class_thresholds = {               # block if P(class) >= this
    1: 0.60,   # Phishing
    2: 0.55,   # Malware
    3: 0.55,   # Defacement
}
```

A URL is blocked if **either** condition is met:
- `confidence >= class_threshold` for its predicted class, OR
- `malicious_probability >= malicious_threshold`

Where `malicious_probability = P(Phishing) + P(Malware) + P(Defacement)`

### Quality Gates

During training, the model must meet minimum recall thresholds before the checkpoint is saved:

| Class      | Minimum Recall | Environment Variable    |
|------------|---------------|-------------------------|
| Phishing   | 85%           | `MIN_RECALL_PHISHING`   |
| Malware    | 80%           | `MIN_RECALL_MALWARE`    |
| Defacement | 75%           | `MIN_RECALL_DEFACEMENT` |

---

## Training

### Training Pipeline

```bash
# Interactive TUI + live dashboard
python -m src.models.train_unified

# DistilBERT only (no dashboard)
python -m src.models.distilbert_trainer

# Baseline models only (LR, RF, XGBoost)
python -m src.models.baseline_models
```

### Training Data Format

Training data should be CSV files with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `url`  | str  | The full URL to classify |
| `label`| int  | Integer class label (0=Benign, 1=Phishing, 2=Malware, 3=Defacement) |

Place training data in `data/train_data.csv` and test data in `data/test_data.csv`.

### Data Augmentation

The training pipeline includes URL augmentation strategies:
- Subdomain injection (`https://sub.domain.com/path` → `https://sub.domain.com.subdomain.com/path`)
- Port manipulation
- HTTPS/HTTP variation
- Trailing slash variation
- Query parameter shuffling

---

## API Reference

### PhishingDetector

```python
from src.gateway.detector import PhishingDetector

detector = PhishingDetector()
result = detector.predict("https://example.com/login")

# Returns:
# {
#     "prediction": 0,                    # integer class label
#     "label": "Benign",                  # human-readable label
#     "probability": 0.943,                # confidence for predicted class
#     "class_probabilities": [0.943, ...], # probabilities for all 4 classes
#     "malicious_probability": 0.057,      # P(Phishing) + P(Malware) + P(Defacement)
#     "blocked": False,                    # True if gateway should reject
#     "decision_reason": "below-threshold(...)"  # why it was/wasn't blocked
# }
```

### predict_url (standalone function)

```python
from src.models.model_loader import predict_url

result = predict_url("https://malicious-site.com/phish")
# Returns the same dict as above (without 'blocked' and 'decision_reason')
```

### Model Loader

```python
from src.models.model_loader import get_model_loader, DistilBERTLoader

loader = get_model_loader()  # singleton instance
result = loader.predict("https://example.com")

# Or instantiate directly with a custom model path:
loader = DistilBERTLoader(model_dir="/path/to/model")
```

---

## Tech Stack

| Layer           | Technology                                      |
|-----------------|-------------------------------------------------|
| SMTP Gateway    | aiosmtpd 1.4.6                                  |
| Primary Model   | DistilBERT (distilbert-base-uncased) fine-tuned |
| ML Framework    | PyTorch 2.3.1, Transformers 4.41.2             |
| Data            | HuggingFace Datasets 2.20.0, Accelerate 0.31.0  |
| Baseline Models  | XGBoost 2.0+, Scikit-learn 1.3.0, Joblib 1.3.0  |
| Visualization   | Matplotlib 3.7.1                                 |
| Web UI          | Streamlit 1.35.0                                 |
| Database        | SQLite (phishing_data.db)                        |
| Alerting        | Mailgun SMTP                                     |
| Python Version  | 3.10+                                           |

---

## Performance Notes

- **Inference latency**: ~50ms per URL on CPU (DistilBERT with batch size 1)
- **Model size**: ~265MB (DistilBERT pytorch_model.bin)
- **Cache**: SQLite恶意 URL cache eliminates repeated inference for known-bad URLs
- **Hot reload**: The detector monitors `pytorch_model.bin` modification time and reloads automatically when a new model is deployed
- **GPU acceleration**: The model loader automatically uses CUDA if available (`torch.device("cuda" if torch.cuda.is_available() else "cpu")`)

---

## License

MIT License. See LICENSE file for details.
