"""
Baseline Model Predictor
========================
Loads saved baseline models (LR, RF) and provides URL classification.
Uses pre-trained TF-IDF vectorizer + handcrafted features.
"""

import pickle
import os
import re
import math
from urllib.parse import urlparse
from typing import Dict, Any

from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

SUSPICIOUS_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "pw", "club", "zip", "loan", "work", "racing", "win", "review", "country", "stream", "download", "trade"}
SHORTENERS = {"bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly", "adf.ly", "j.mp", "tr.im", "tiny.cc", "lnkd.in", "db.tt", "qr.ae", "ady.me", "gg.gg", "bit.do", "t2mio.com", "cot.ag", "北海", "cut.ly", "go.ok" }
BRANDS = {"google", "facebook", "amazon", "apple", "microsoft", "paypal", "ebay", "netflix", "instagram", "twitter", "linkedin", "reddit", "dropbox", "adobe", "salesforce", "shopify", "stripe", "slack", "zoom", "whatsapp", "telegram", "discord", "spotify", "steam", "blizzard", "ea", "ubisoft", "rockstar", "epic", "valve"}

LR_MODEL = None
LR_SCALER = None
RF_MODEL = None
TFIDF_VECTORIZER = None


def _load_models():
    """Lazy load baseline models."""
    global LR_MODEL, LR_SCALER, RF_MODEL, TFIDF_VECTORIZER

    if LR_MODEL is None:
        print("[BASELINE] Loading Logistic Regression model...")
        with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "rb") as f:
            LR_MODEL = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "logistic_regression_scaler.pkl"), "rb") as f:
            LR_SCALER = pickle.load(f)
        print("[BASELINE] LR model loaded.")

    if RF_MODEL is None:
        print("[BASELINE] Loading Random Forest model...")
        with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "rb") as f:
            RF_MODEL = pickle.load(f)
        print("[BASELINE] RF model loaded.")

    if TFIDF_VECTORIZER is None:
        print("[BASELINE] Loading TF-IDF vectorizer...")
        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "rb") as f:
            TFIDF_VECTORIZER = pickle.load(f)
        print("[BASELINE] TF-IDF vectorizer loaded.")


def _shannon_entropy(s: str) -> float:
    """Calculate Shannon entropy of a string."""
    if not s:
        return 0
    freq = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    return -sum((f / len(s)) * math.log2(f / len(s)) for f in freq.values())


def _extract_url_features(url: str) -> list:
    """Extract 28 handcrafted features from URL."""
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
        _shannon_entropy(domain),
        int(any(b in url.lower() for b in BRANDS)),
        len(tld),
        int(domain.lower() in SHORTENERS),
        _shannon_entropy(url),
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


def _get_tfidf_features(url: str):
    """Get TF-IDF features for URL."""
    _load_models()
    return TFIDF_VECTORIZER.transform([url]).toarray()[0]


def _prepare_combined_features(url: str):
    """Prepare combined TF-IDF + handcrafted features."""
    tfidf_feat = _get_tfidf_features(url)
    hand_feat = _extract_url_features(url)
    return list(tfidf_feat) + hand_feat


def predict_with_lr(url: str) -> Dict[str, Any]:
    """Predict using Logistic Regression."""
    _load_models()
    features = _prepare_combined_features(url)
    features_scaled = LR_SCALER.transform([features])
    pred = LR_MODEL.predict(features_scaled)[0]
    probs = LR_MODEL.predict_proba(features_scaled)[0]

    return {
        "prediction": int(pred),
        "label": ["Benign", "Defacement", "Malware", "Phishing"][int(pred)],
        "probability": float(max(probs)),
        "class_probabilities": [float(p) for p in probs],
        "malicious_probability": float(sum(probs[1:])),
    }


def predict_with_rf(url: str) -> Dict[str, Any]:
    """Predict using Random Forest."""
    _load_models()
    features = _prepare_combined_features(url)
    pred = RF_MODEL.predict([features])[0]
    probs = RF_MODEL.predict_proba([features])[0]

    return {
        "prediction": int(pred),
        "label": ["Benign", "Defacement", "Malware", "Phishing"][int(pred)],
        "probability": float(max(probs)),
        "class_probabilities": [float(p) for p in probs],
        "malicious_probability": float(sum(probs[1:])),
    }