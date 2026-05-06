import json
import math
import os
import pickle
import re
import time
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

from .config import EVAL_DIR, EXPERIMENTS_DIR, FIXED_EVAL_SAMPLES, QUALITY_GATES, RANDOM_SEED
from .evaluation import create_or_load_fixed_eval_indices
from .experiment_tracking import finalize_experiment_run, start_experiment_run
from .quality_gates import evaluate_quality_gates

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "logs", "model_comparison")
MODELS_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

PROGRESS_FILE = os.path.join(BASE_DIR, "logs", "training_progress.json")

def _update_progress(status, percent, loss="--", details=""):
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    data = {"status": status, "progress": percent, "loss": loss, "details": details}
    with open(PROGRESS_FILE, "w") as f:
        json.dump(data, f)

SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "pw", "club", "zip", "mov"}
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd"}
BRANDS = ["paypal", "apple", "google", "microsoft", "amazon", "icloud", "ebay", "bank", "wellsfargo"]

def shannon_entropy(s: str) -> float:
    if not s:
        return 0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((f / len(s)) * math.log2(f / len(s)) for f in freq.values())

def extract_url_features(url: str) -> list:
    try:
        parsed = urlparse(url if url.startswith("http") else "http://" + url)
    except Exception:
        parsed = urlparse("http://invalid")

    domain = parsed.netloc or parsed.path.split("/")[0]
    path = parsed.path
    query = parsed.query
    tld = domain.split(".")[-1].lower() if "." in domain else ""
    subdomain_parts = domain.split(".")[:-2] if domain.count(".") > 1 else []
    words = re.split(r"[^a-zA-Z]", url)
    words = [w for w in words if w]
    vowels = sum(c in "aeiouAEIOU" for c in url)

    features = [
        len(url),
        url.count("."),
        url.count("-"),
        url.count("/"),
        sum(c.isdigit() for c in url),
        sum(c.isdigit() for c in url) / max(len(url), 1),
        len(re.findall(r"[@?=&%#_~]", url)),
        int("%" in url),
        len(domain),
        len(subdomain_parts),
        int(tld in SUSPICIOUS_TLDS),
        int(bool(re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", url))),
        shannon_entropy(domain),
        int(any(b in url.lower() for b in BRANDS)),
        len(tld),
        int(domain.lower() in SHORTENERS),
        shannon_entropy(url),
        len(path),
        len(query),
        len(query.split("&")) if query else 0,
        path.count("/"),
        vowels / max(len(url), 1),
        max((len(w) for w in words), default=0),
        int(url.startswith("https")),
        int("@" in url),
        int("//" in url.replace("https://", "").replace("http://", "")),
        int(url.count("http") > 1),
        int(bool(re.search(r"[A-Za-z0-9+/]{20,}={0,2}", url))),
        int(bool(re.search(r"%[0-9a-fA-F]{2}", url))),
    ]
    return features

FEATURE_NAMES = [
    "url_length", "num_dots", "num_hyphens", "num_slashes", "num_digits",
    "digit_ratio", "num_special_chars", "has_encoded_chars",
    "domain_length", "num_subdomains", "tld_suspicious", "has_ip_address",
    "domain_entropy", "brand_spoofing", "tld_length", "is_shortened",
    "url_entropy", "path_length", "query_length", "num_params",
    "path_depth", "vowel_ratio", "longest_word_len",
    "has_https", "has_at_symbol", "has_double_slash", "has_redirect",
    "base64_in_url", "hex_in_url"
]

def load_data() -> tuple:
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_data.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"))
    return train_df, test_df

def prepare_features(train_urls, test_urls):
    print("\n[1/3] Extracting TF-IDF features...", flush=True)
    tfidf = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), max_features=5000, lowercase=True)
    X_train_tfidf = tfidf.fit_transform(train_urls)
    X_test_tfidf = tfidf.transform(test_urls)
    print(f"  TF-IDF features: {X_train_tfidf.shape[1]}", flush=True)

    print("\n[2/3] Extracting handcrafted features...", flush=True)
    X_train_hand = np.array([extract_url_features(u) for u in train_urls])
    X_test_hand = np.array([extract_url_features(u) for u in test_urls])
    print(f"  Handcrafted features: {X_train_hand.shape[1]}")

    X_train = np.hstack([X_train_tfidf.toarray(), X_train_hand])
    X_test = np.hstack([X_test_tfidf.toarray(), X_test_hand])
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1e6, neginf=-1e6)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
    print(f"  Combined features: {X_train.shape[1]}")

    return tfidf, X_train, X_test

def evaluate_model(y_true, y_pred, model_name: str) -> dict:
    per_class_recall_values = recall_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    per_class_recall = {idx: float(per_class_recall_values[idx]) for idx in range(4)}
    quality_gates = evaluate_quality_gates(per_class_recall, QUALITY_GATES)
    return {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "per_class_recall": per_class_recall,
        "quality_gates": quality_gates,
        "deployment_recommended": quality_gates["passed"],
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def run_logistic_regression(X_train, X_test, y_train, y_test) -> tuple:
    print("Training Logistic Regression...", flush=True)
    start = time.time()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    model = LogisticRegression(
        max_iter=2000, 
        random_state=RANDOM_SEED, 
        solver='lbfgs', 
        tol=0.01,
        class_weight='balanced'
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    elapsed = time.time() - start
    metrics = evaluate_model(y_test, y_pred, "LogisticRegression")
    metrics["training_time"] = elapsed
    metrics["feature_importance"] = None
    return model, scaler, metrics

def run_random_forest(X_train, X_test, y_train, y_test) -> tuple:
    print("Training Random Forest...", flush=True)
    start = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        class_weight='balanced_subsample'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    elapsed = time.time() - start
    metrics = evaluate_model(y_test, y_pred, "RandomForest")
    metrics["training_time"] = elapsed
    metrics["feature_importance"] = model.feature_importances_.tolist()
    return model, metrics

def run_xgboost(X_train, X_test, y_train, y_test) -> tuple:
    if XGBClassifier is None:
        raise ImportError("xgboost is not installed")
    print("Training XGBoost...", flush=True)
    start = time.time()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)
    model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    y_pred = model.predict(X_test)
    elapsed = time.time() - start
    metrics = evaluate_model(y_test, y_pred, "XGBoost")
    metrics["training_time"] = elapsed
    metrics["feature_importance"] = model.feature_importances_.tolist()
    return model, metrics

def save_models(tfidf, lr_model, lr_scaler, rf_model, xgb_model=None):
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "wb") as f:
        pickle.dump(lr_model, f)
    with open(os.path.join(MODELS_DIR, "logistic_regression_scaler.pkl"), "wb") as f:
        pickle.dump(lr_scaler, f)
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "wb") as f:
        pickle.dump(rf_model, f)
    if xgb_model is not None:
        with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "wb") as f:
            pickle.dump(xgb_model, f)
    print(f"\nModels saved to: {MODELS_DIR}")

def run_benchmark(model, X_test, n_samples=1000) -> float:
    subset = X_test[:n_samples]
    start = time.time()
    model.predict(subset)
    elapsed = time.time() - start
    return n_samples / elapsed

def main():
    print("=" * 60)
    print("Baseline Models Comparison for URL Phishing Detection")
    print("=" * 60)

    train_df, test_df = load_data()
    run_context = start_experiment_run(
        experiments_dir=EXPERIMENTS_DIR,
        run_type="training",
        model_name="baseline",
        config={
            "fixed_eval_samples": FIXED_EVAL_SAMPLES,
            "quality_gates": QUALITY_GATES,
            "seed": RANDOM_SEED,
        },
        dataset={
            "train_path": os.path.join(DATA_DIR, "train_data.csv"),
            "test_path": os.path.join(DATA_DIR, "test_data.csv"),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
        },
    )
    print(f"Loaded {len(train_df)} train, {len(test_df)} test samples")

    max_train = 20000
    if len(train_df) > max_train:
        samples_per_class = max_train // 4
        train_df = pd.concat([
            train_df[train_df["label"] == i].sample(
                n=min(samples_per_class, (train_df["label"] == i).sum()),
                random_state=RANDOM_SEED,
            )
            for i in range(4)
        ])
        print(f"Sampled {len(train_df)} training rows (4-class stratified)")

    eval_indices_file = os.path.join(EVAL_DIR, "fixed_eval_indices.json")
    eval_indices = create_or_load_fixed_eval_indices(
        labels=test_df["label"].tolist(),
        output_path=eval_indices_file,
        max_samples=FIXED_EVAL_SAMPLES,
        seed=RANDOM_SEED,
    )
    if eval_indices:
        test_df = test_df.iloc[eval_indices].reset_index(drop=True)
        print(f"Using fixed evaluation split: {len(test_df)} rows")

    train_urls = train_df["url"].tolist()
    train_labels = train_df["label"].values
    test_urls = test_df["url"].tolist()
    test_labels = test_df["label"].values

    tfidf, X_train, X_test = prepare_features(train_urls, test_urls)

    results = []

    _update_progress("running", 12, details="Training Logistic Regression...")
    lr_model, lr_scaler, lr_metrics = run_logistic_regression(X_train, X_test, train_labels, test_labels)
    results.append(lr_metrics)

    _update_progress("running", 18, details="Training Random Forest...")
    rf_model, rf_metrics = run_random_forest(X_train, X_test, train_labels, test_labels)
    results.append(rf_metrics)

    xgb_model = None
    if XGBClassifier is not None:
        _update_progress("running", 25, details="Training XGBoost...")
        xgb_model, xgb_metrics = run_xgboost(X_train, X_test, train_labels, test_labels)
        results.append(xgb_metrics)
    else:
        print("Skipping XGBoost (package not installed).")

    print("\n" + "=" * 60)
    print("Model Comparison Results")
    print("=" * 60)

    try:
        distilbert_metrics = json.load(open(os.path.join(BASE_DIR, "phishing_model", "training_metrics.json")))
        distilbert_speed = 0
        distilbert_preds = []

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(BASE_DIR, "phishing_model"))
            model = AutoModelForSequenceClassification.from_pretrained(os.path.join(BASE_DIR, "phishing_model"))
            model.eval()

            for i, url in enumerate(test_urls[:2000]):
                inputs = tokenizer(url, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    outputs = model(**inputs)
                    pred = outputs.logits.argmax(dim=1).item()
                    distilbert_preds.append(pred)

                if i % 500 == 0:
                    print(f"  DistilBERT predictions: {i}/{min(2000, len(test_urls))}")

            print("Benchmarking DistilBERT inference speed...")
            db_start = time.time()
            db_subset = test_urls[:200]
            for u in db_subset:
                inputs = tokenizer(u, return_tensors="pt", truncation=True, max_length=512)
                with torch.no_grad():
                    model(**inputs)
            db_elapsed = time.time() - db_start
            distilbert_speed = len(db_subset) / db_elapsed
            print(f"  DistilBERT: {distilbert_speed:.0f} samples/sec")

        except Exception as e:
            print(f"  Could not compute DistilBERT metrics: {e}")

        db_true = test_labels[: len(distilbert_preds)] if distilbert_preds else []
        per_class_recall = {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0}
        quality_gates = {"passed": False, "checks": {}}
        if len(db_true) > 0:
            per_class_values = recall_score(db_true, distilbert_preds, labels=[0, 1, 2, 3], average=None, zero_division=0)
            per_class_recall = {idx: float(per_class_values[idx]) for idx in range(4)}
            quality_gates = evaluate_quality_gates(per_class_recall, QUALITY_GATES)

        distilbert_result = {
            "model": "DistilBERT",
            "accuracy": accuracy_score(db_true, distilbert_preds) if len(db_true) > 0 else 0.0,
            "precision": precision_score(db_true, distilbert_preds, average="weighted", zero_division=0) if len(db_true) > 0 else 0.0,
            "recall": recall_score(db_true, distilbert_preds, average="weighted", zero_division=0) if len(db_true) > 0 else 0.0,
            "f1": f1_score(db_true, distilbert_preds, average="weighted", zero_division=0) if len(db_true) > 0 else 0.0,
            "training_time": distilbert_metrics.get("training_time", 600),
            "inference_speed": distilbert_speed,
            "per_class_recall": per_class_recall,
            "quality_gates": quality_gates,
            "deployment_recommended": quality_gates["passed"],
            "confusion_matrix": (
                confusion_matrix(db_true, distilbert_preds, labels=[0, 1, 2, 3]).tolist()
                if len(db_true) > 0
                else [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            ),
        }

    except Exception as e:
        distilbert_result = {
            "model": "DistilBERT",
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "training_time": 0.0,
            "inference_speed": 0,
            "per_class_recall": {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0},
            "quality_gates": {"passed": False, "checks": {}},
            "deployment_recommended": False,
            "confusion_matrix": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        }
    
    results.append(distilbert_result)

    print("\nInference Speed Benchmark (samples/sec):")
    benchmark_inputs = [
        (lr_model, "LogisticRegression", lr_scaler.transform(X_test)),
        (rf_model, "RandomForest", X_test),
    ]
    if xgb_model is not None:
        benchmark_inputs.append((xgb_model, "XGBoost", X_test))
    for model, name, benchmark_matrix in benchmark_inputs:
        speed = run_benchmark(model, benchmark_matrix)
        for r in results:
            if r["model"] == name:
                r["inference_speed"] = speed
                break
        print(f"  {name}: {speed:.0f} samples/sec")

    for r in results:
        print(f"\n{r['model']}:")
        print(f"  Accuracy:  {r['accuracy']:.4f}")
        print(f"  Precision: {r['precision']:.4f}")
        print(f"  Recall:    {r['recall']:.4f}")
        print(f"  F1 Score:  {r['f1']:.4f}")
        print(f"  Train Time: {r['training_time']:.2f}s")
        if "inference_speed" in r:
            print(f"  Inference:  {r['inference_speed']:.0f} samples/sec")

    comparison = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": {
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "features": X_train.shape[1],
            "fixed_eval_indices_file": eval_indices_file,
        },
        "quality_gates": QUALITY_GATES,
        "results": results,
    }

    output_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    with open(output_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    accepted_models = [result["model"] for result in results if result.get("deployment_recommended")]
    finalize_experiment_run(
        run_context=run_context,
        status="accepted" if accepted_models else "rejected",
        metrics=comparison,
        quality_gates={"accepted_models": accepted_models},
        artifacts=[output_path, MODELS_DIR],
        notes="Baseline benchmark run with fixed evaluation split and malicious-class recall gates.",
    )

    save_models(tfidf, lr_model, lr_scaler, rf_model, xgb_model)
    print("\nBaseline model comparison complete!")

    return comparison

if __name__ == "__main__":
    main()
