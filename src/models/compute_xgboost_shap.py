"""
compute_xgboost_shap.py
Computes SHAP values for XGBoost model and generates summary plots.
Run after training is complete.

Usage:
    D:\smtpBERT\venv\Scripts\python.exe -m src.models.compute_xgboost_shap
"""

import json
import math
import os
import pickle
import re
import sys
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import config as model_config

BASE_DIR = model_config.BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(model_config.LOG_DIR, "model_comparison")
EVAL_DIR = model_config.EVAL_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(model_config.LOG_DIR, "figures")

SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "pw", "club", "zip", "mov"}
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd"}
BRANDS = ["paypal", "apple", "google", "microsoft", "amazon", "icloud", "ebay", "bank", "wellsfargo"]

CLASS_NAMES = ["benign", "phishing", "malware", "defacement"]

FEATURE_NAMES = [
    "url_length", "dot_count", "dash_count", "slash_count", "digit_count", "digit_ratio",
    "special_char_count", "has_pct", "domain_length", "subdomain_count", "suspicious_tld",
    "has_ip", "domain_entropy", "has_brand", "tld_length", "is_shortener",
    "url_entropy", "path_length", "query_length", "query_param_count", "path_slash_count",
    "vowel_ratio", "max_word_length", "has_https", "has_at", "has_double_slash",
    "multiple_http", "is_base64", "is_url_encoded",
]


def shannon_entropy(s: str) -> float:
    if not s:
        return 0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((f / len(s)) * math.log2(f / len(s)) for f in freq.values())


def extract_url_features(url: str):
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

    return [
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


def load_fixed_eval_indices():
    eval_file = os.path.join(EVAL_DIR, "fixed_eval_indices.json")
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Fixed eval indices not found: {eval_file}")
    with open(eval_file, "r") as f:
        return json.load(f)["indices"]


def main():
    print("=" * 60)
    print("XGBoost SHAP Computation")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n[1/4] Loading XGBoost model...")
    with open(os.path.join(MODELS_DIR, "xgboost.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    print("  XGBoost loaded.")

    print("\n[2/4] Loading TF-IDF and test data...")
    with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
        tfidf = pickle.load(f)
    print(f"  TF-IDF vocab: {len(tfidf.vocabulary_)} terms")

    eval_indices = load_fixed_eval_indices()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"), engine="python")
    test_df = test_df.iloc[eval_indices].reset_index(drop=True)

    test_urls = test_df["url"].astype(str).tolist()
    print(f"  Test URLs: {len(test_urls)}")

    print("\n[3/4] Extracting features...")
    tfidf_features = tfidf.transform(test_urls)
    hand_features = [extract_url_features(u) for u in test_urls]
    hand_feature_names = FEATURE_NAMES

    all_feature_names = [f"tfidf_{i}" for i in range(tfidf_features.shape[1])] + hand_feature_names

    X_tfidf = tfidf_features.toarray()
    X = np.hstack([X_tfidf, hand_features])
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6).astype(np.float32)
    print(f"  Feature matrix: {X.shape}")

    print("\n[4/4] Computing SHAP values (this takes ~2-3 min)...")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X)

    print(f"  SHAP values shape: {shap_values.shape}")
    print("\n  Saving per-class SHAP summary plots...")

    for i, cls_name in enumerate(CLASS_NAMES):
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values[:, :, i],
            X,
            feature_names=all_feature_names,
            class_names=CLASS_NAMES,
            show=False,
            max_display=20,
        )
        plt.title(f"XGBoost SHAP — {cls_name.capitalize()} Class\nImpact on prediction", fontsize=12)
        plt.xlabel("SHAP value (impact on model output)", fontsize=9)
        plt.tight_layout()
        out_path = os.path.join(FIGURES_DIR, f"xgboost_shap_{cls_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        [shap_values[:, :, i] for i in range(len(CLASS_NAMES))],
        X,
        feature_names=all_feature_names,
        class_names=CLASS_NAMES,
        show=False,
        max_display=20,
    )
    plt.title("XGBoost SHAP — All Classes\nFeature impact across all 4 classes", fontsize=12)
    plt.xlabel("SHAP value", fontsize=9)
    plt.tight_layout()
    all_path = os.path.join(FIGURES_DIR, "xgboost_shap_summary.png")
    plt.savefig(all_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {all_path}")

    print(f"\n[DONE] XGBoost SHAP plots saved to {FIGURES_DIR}/")
    print("       Run the dashboard to view them in the XAI / SHAP tab.")


if __name__ == "__main__":
    main()