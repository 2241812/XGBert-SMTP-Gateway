"""
compute_inference_metrics.py
Precomputes inference latency, memory usage, and OvR ROC curve data for all models.
Run after training is complete and saved models exist.

Usage:
    D:\smtpBERT\venv\Scripts\python.exe -m src.models.compute_inference_metrics

Output (saved to logs/model_comparison/):
    - inference_metrics.json  : ms/url + memory per model
    - roc_curves.json         : FPR, TPR, AUC per class per model
    - model_sizes.json        : param count + size per model
"""

import gc
import json
import math
import os
import pickle
import re
import sys
import time
import warnings
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import config as model_config

BASE_DIR = model_config.BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(model_config.LOG_DIR, "model_comparison")
EVAL_DIR = model_config.EVAL_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
PHISHING_MODEL_DIR = model_config.MODEL_OUTPUT_DIR_DISTILBERT

SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "pw", "club", "zip", "mov"}
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd"}
BRANDS = ["paypal", "apple", "google", "microsoft", "amazon", "icloud", "ebay", "bank", "wellsfargo"]

CLASS_NAMES = ["benign", "phishing", "malware", "defacement"]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((f / len(s)) * math.log2(f / len(s)) for f in freq.values())


def extract_url_features(url: str) -> np.ndarray:
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
    return np.array(features, dtype=np.float32)


def load_fixed_eval_indices():
    eval_file = os.path.join(EVAL_DIR, "fixed_eval_indices.json")
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Fixed eval indices not found: {eval_file}. "
                                "Run compute_baseline_roc_auc.py first to create them.")
    with open(eval_file, "r") as f:
        payload = json.load(f)
    return payload["indices"]


def load_baseline_models():
    models = {}
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        models["tfidf"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "rb") as f:
        models["lr"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "logistic_regression_scaler.pkl"), "rb") as f:
        models["lr_scaler"] = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "rb") as f:
        models["rf"] = pickle.load(f)
    xgb_path = os.path.join(MODELS_DIR, "xgboost.pkl")
    if os.path.exists(xgb_path):
        with open(xgb_path, "rb") as f:
            models["xgb"] = pickle.load(f)
    return models


def load_distilbert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_path = PHISHING_MODEL_DIR
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DistilBERT model not found at {model_path}. Train first.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, num_labels=4
    ).to(device)
    model.eval()
    return model, tokenizer, device


def prepare_baseline_features(urls, tfidf, lr_scaler):
    tfidf_features = tfidf.transform(urls)
    hand_features = np.array([extract_url_features(u) for u in urls])
    X = np.hstack([tfidf_features.toarray(), hand_features])
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    X_scaled = lr_scaler.transform(X)
    return X, X_scaled


def predict_baseline_batch(model, X, model_type):
    if model_type == "lr":
        return model.predict_proba(X)
    elif model_type == "rf":
        return model.predict_proba(X)
    elif model_type == "xgb":
        return model.predict_proba(X)
    raise ValueError(f"Unknown model type: {model_type}")


def predict_distilbert_batch(urls, model, tokenizer, device, batch_size=32):
    probas = []
    with torch.no_grad():
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(device)
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            probas.append(probs)
    return np.vstack(probas)


def benchmark_latency(fn, n_runs=5):
    timings = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        timings.append((t1 - t0))
    return (sum(timings) / len(timings)) * 1000


def get_memory_mb(device_type):
    if device_type == "cuda":
        torch.cuda.synchronize()
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0.0


def main():
    print("=" * 60)
    print("Inference Metrics + ROC Curve Precomputation")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("\n[1/7] Loading fixed evaluation indices...")
    eval_indices = load_fixed_eval_indices()
    print(f"  Test set: {len(eval_indices)} samples")

    print("\n[2/7] Loading test data...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"), engine="python")
    test_df = test_df.iloc[eval_indices].reset_index(drop=True)
    test_urls = test_df["url"].astype(str).tolist()
    test_labels = test_df["label"].astype(int).values
    print(f"  Loaded: {len(test_urls)} URLs, labels: {np.bincount(test_labels)}")

    print("\n[3/7] Loading baseline models...")
    baseline_models = load_baseline_models()
    print(f"  TF-IDF vocab: {len(baseline_models['tfidf'].vocabulary_)} terms")

    print("\n[4/7] Loading DistilBERT...")
    distilbert, tokenizer, device = load_distilbert()
    print(f"  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    print("\n[5/7] Benchmarking latency (5 passes)...")
    baseline_latencies = {}
    n_runs = 5

    _, X_scaled = prepare_baseline_features(test_urls, baseline_models["tfidf"], baseline_models["lr_scaler"])

    baseline_model_keys = [
        ("LogisticRegression", "lr", baseline_models["lr"]),
        ("RandomForest", "rf", baseline_models["rf"]),
        ("XGBoost", "xgb", baseline_models["xgb"]),
    ]
    for name, key, model in baseline_model_keys:
        print(f"\n  {name}...")
        avg_ms = benchmark_latency(lambda m=model, k=key, x=X_scaled: predict_baseline_batch(m, x, k), n_runs)
        ms_per_url = avg_ms / len(test_urls)
        baseline_latencies[name] = ms_per_url
        print(f"    {avg_ms:.1f}ms total / {len(test_urls)} URLs = {ms_per_url:.4f} ms/URL")

    print(f"\n  DistilBERT...")
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    mem_before = get_memory_mb("cuda") if torch.cuda.is_available() else 0
    distilbert_fn = lambda: predict_distilbert_batch(test_urls, distilbert, tokenizer, device, batch_size=32)
    distilbert_total_ms = benchmark_latency(distilbert_fn, n_runs)
    distilbert_ms_per_url = distilbert_total_ms / len(test_urls)
    baseline_latencies["DistilBERT"] = distilbert_ms_per_url
    mem_after = get_memory_mb("cuda") if torch.cuda.is_available() else 0
    print(f"    {distilbert_total_ms:.1f}ms total / {len(test_urls)} URLs = {distilbert_ms_per_url:.4f} ms/URL")

    print("\n[6/7] Computing per-class ROC curves...")

    model_map = {
        "LogisticRegression": ("lr", baseline_models["lr"], X_scaled),
        "RandomForest": ("rf", baseline_models["rf"], X_scaled),
        "XGBoost": ("xgb", baseline_models["xgb"], X_scaled),
        "DistilBERT": ("distilbert", None, None),
    }

    roc_curves = {}
    y_true_bin = label_binarize(test_labels, classes=[0, 1, 2, 3])

    for model_name, (key, model, X) in model_map.items():
        print(f"\n  {model_name}...")
        if key == "distilbert":
            proba = predict_distilbert_batch(test_urls, distilbert, tokenizer, device, batch_size=32)
        else:
            proba = predict_baseline_batch(model, X, key)

        model_roc = {}
        for i, cls_name in enumerate(CLASS_NAMES):
            fpr, tpr, _ = roc_curve(y_true_bin[:, i], proba[:, i])
            auc = roc_auc_score(y_true_bin[:, i], proba[:, i])
            model_roc[cls_name] = {
                "fpr": [float(v) for v in fpr],
                "tpr": [float(v) for v in tpr],
                "auc": float(auc),
            }
            print(f"    {cls_name.capitalize()}: AUC={auc:.4f}")

        roc_curves[model_name.lower().replace(" ", "_")] = model_roc

    print("\n[7/7] Saving results...")

    model_accuracies = {}
    holdout_data = {}
    holdout_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    if os.path.exists(holdout_path):
        with open(holdout_path, "r") as f:
            holdout_data = json.load(f)
        for r in holdout_data.get("results", []):
            mn = r.get("model", "")
            model_accuracies[mn] = {
                "accuracy": r.get("accuracy", 0),
                "roc_auc": r.get("roc_auc", 0),
            }
    distilbert_holdout_path = os.path.join(model_config.MODEL_OUTPUT_DIR_DISTILBERT, "training_metrics.json")
    if os.path.exists(distilbert_holdout_path):
        with open(distilbert_holdout_path, "r") as f:
            d = json.load(f)
            model_accuracies["DistilBERT"] = {
                "accuracy": d.get("eval_accuracy", 0),
                "roc_auc": d.get("eval_roc_auc", 0),
            }

    inference_metrics = {}
    for name in ["LogisticRegression", "RandomForest", "XGBoost", "DistilBERT"]:
        key = name.lower().replace(" ", "_")
        model_file_size = 0
        if name != "DistilBERT":
            model_file = os.path.join(MODELS_DIR, f"{name.lower().replace(' ', '_')}.pkl")
            if os.path.exists(model_file):
                model_file_size = os.path.getsize(model_file) / (1024 * 1024)

        inference_metrics[name] = {
            "ms_per_url": float(baseline_latencies.get(name, 0)),
            "memory_mb": float(mem_after) if name == "DistilBERT" else float(model_file_size),
            "accuracy": model_accuracies.get(name, {}).get("accuracy", 0),
            "roc_auc": model_accuracies.get(name, {}).get("roc_auc", 0),
        }

    with open(os.path.join(RESULTS_DIR, "inference_metrics.json"), "w") as f:
        json.dump(inference_metrics, f, indent=2)

    with open(os.path.join(RESULTS_DIR, "roc_curves.json"), "w") as f:
        json.dump(roc_curves, f, indent=2)

    print(f"\n  inference_metrics.json  → {RESULTS_DIR}")
    print(f"  roc_curves.json         → {RESULTS_DIR}")

    del distilbert
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("\n[DONE] Precomputation complete.")
    print("\nLatency summary (ms/URL):")
    for name, ms in baseline_latencies.items():
        print(f"  {name:20s}: {ms:.4f} ms/URL")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()