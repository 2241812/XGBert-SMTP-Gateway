"""
Standalone script to compute ROC-AUC for baseline models using saved models.
Does NOT retrain anything — purely evaluation using saved artifacts.
"""

import json
import math
import os
import pickle
import re
from urllib.parse import urlparse

import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "logs", "model_comparison")
EVAL_DIR = os.path.join(BASE_DIR, "logs", "evaluation")
DATA_DIR = os.path.join(BASE_DIR, "data")


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


def load_fixed_eval_indices():
    eval_file = os.path.join(EVAL_DIR, "fixed_eval_indices.json")
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Fixed eval indices not found: {eval_file}")
    with open(eval_file, "r") as f:
        payload = json.load(f)
    return payload["indices"]


def load_models():
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


def compute_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    return roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="macro")


def main():
    print("=" * 60)
    print("Baseline Models ROC-AUC Computation (No Retraining)")
    print("=" * 60)

    print("\n[1/4] Loading fixed evaluation indices...")
    eval_indices = load_fixed_eval_indices()
    print(f"  Fixed eval set: {len(eval_indices)} samples")

    print("\n[2/4] Loading test data and applying fixed split...")
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"), engine="python")
    test_df["url"] = test_df["url"].astype(str)
    test_df["label"] = test_df["label"].astype(int)
    test_df = test_df.iloc[eval_indices].reset_index(drop=True)
    print(f"  Test subset: {len(test_df)} samples")

    print("\n[3/4] Loading saved models...")
    models = load_models()
    print(f"  Models loaded: TF-IDF, LR, LR_Scaler, RF" + ("XGB" if "xgb" in models else ""))

    print("\n[4/4] Extracting features and computing ROC-AUC...")
    test_urls = test_df["url"].tolist()
    test_labels = test_df["label"].values

    tfidf_features = models["tfidf"].transform(test_urls)
    hand_features = np.array([extract_url_features(u) for u in test_urls])
    X_test = np.hstack([tfidf_features.toarray(), hand_features])
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)

    lr_scaler = models["lr_scaler"]
    X_test_scaled = lr_scaler.transform(X_test)

    results = {}

    print("\n  Logistic Regression...")
    lr_proba = models["lr"].predict_proba(X_test_scaled)
    lr_auc = compute_roc_auc(test_labels, lr_proba)
    results["LogisticRegression"] = {"roc_auc": lr_auc}
    print(f"    ROC-AUC: {lr_auc:.4f}")

    print("\n  Random Forest...")
    rf_proba = models["rf"].predict_proba(X_test)
    rf_auc = compute_roc_auc(test_labels, rf_proba)
    results["RandomForest"] = {"roc_auc": rf_auc}
    print(f"    ROC-AUC: {rf_auc:.4f}")

    if "xgb" in models:
        print("\n  XGBoost...")
        xgb_proba = models["xgb"].predict_proba(X_test)
        xgb_auc = compute_roc_auc(test_labels, xgb_proba)
        results["XGBoost"] = {"roc_auc": xgb_auc}
        print(f"    ROC-AUC: {xgb_auc:.4f}")

    model_comparison_path = os.path.join(RESULTS_DIR, "model_comparison.json")
    if os.path.exists(model_comparison_path):
        print(f"\n[+] Updating {model_comparison_path}")
        with open(model_comparison_path, "r") as f:
            comparison = json.load(f)

        model_name_map = {
            "LogisticRegression": "LogisticRegression",
            "RandomForest": "RandomForest",
            "XGBoost": "XGBoost",
        }

        for item in comparison.get("results", []):
            model_name = item.get("model")
            if model_name in results:
                item["roc_auc"] = results[model_name]["roc_auc"]

        with open(model_comparison_path, "w") as f:
            json.dump(comparison, f, indent=2)
        print("  ROC-AUC values updated in model_comparison.json")
    else:
        print(f"\n[!] model_comparison.json not found at {model_comparison_path}")
        print("    Computed values (for manual update):")
        for name, data in results.items():
            print(f"    {name}: roc_auc = {data['roc_auc']:.4f}")

    print("\n" + "=" * 60)
    print("Done. Refresh the dashboard to see updated ROC-AUC values.")
    print("=" * 60)


if __name__ == "__main__":
    main()