"""
compute_model_sizes.py
Computes model size metrics: DistilBERT parameter count and baseline model file sizes.
Run after training is complete.

Usage:
    D:\smtpBERT\venv\Scripts\python.exe -m src.models.compute_model_sizes
"""

import json
import os
import pickle
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import config as model_config

BASE_DIR = model_config.BASE_DIR
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(model_config.LOG_DIR, "model_comparison")
MODEL_OUTPUT_DIR = model_config.MODEL_OUTPUT_DIR_DISTILBERT


def count_distilbert_params(model_dir):
    total = 0
    safetensors_file = os.path.join(model_dir, "model.safetensors")
    if os.path.exists(safetensors_file):
        from safetensors.torch import load_file
        state_dict = load_file(safetensors_file, device="cpu")
        total = sum(v.numel() for v in state_dict.values())
    else:
        from transformers import AutoModelForSequenceClassification
        try:
            model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, num_labels=4)
            total = sum(p.numel() for p in model.parameters())
        except Exception:
            pass
    return total


def get_model_file_size(path):
    if os.path.exists(path):
        return os.path.getsize(path) / (1024 * 1024)
    return 0.0


def main():
    print("=" * 60)
    print("Model Size Computation")
    print("=" * 60)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    sizes = {}

    print("\n[1/3] DistilBERT parameter count...")
    n_params = count_distilbert_params(MODEL_OUTPUT_DIR)
    model_file_size = get_model_file_size(os.path.join(MODEL_OUTPUT_DIR, "model.safetensors"))
    sizes["DistilBERT"] = {
        "parameters": n_params,
        "params_millions": round(n_params / 1_000_000, 2),
        "size_mb": round(model_file_size, 2),
        "size_str": f"{n_params / 1_000_000:.1f}M params / {model_file_size:.1f} MB",
    }
    print(f"  {n_params:,} params ({n_params / 1_000_000:.1f}M), {model_file_size:.1f} MB")

    print("\n[2/3] Baseline model file sizes...")
    baselines = {
        "XGBoost": "xgboost.pkl",
        "LogisticRegression": "logistic_regression.pkl",
        "RandomForest": "random_forest.pkl",
    }
    for name, filename in baselines.items():
        path = os.path.join(MODELS_DIR, filename)
        size_mb = get_model_file_size(path)
        model_key = name.lower().replace(" ", "_")
        if name == "XGBoost":
            model_key = "xgboost"
        sizes[name] = {
            "parameters": 0,
            "params_millions": 0,
            "size_mb": round(size_mb, 2),
            "size_str": f"{size_mb:.1f} MB",
        }
        if size_mb > 0:
            with open(path, "rb") as f:
                model = pickle.load(f)
            if hasattr(model, "n_estimators"):
                sizes[name]["n_estimators"] = model.n_estimators
            if hasattr(model, "max_depth"):
                sizes[name]["max_depth"] = model.max_depth
            if name == "LogisticRegression":
                sizes[name]["params_millions"] = model.coef_.size / 1_000_000
                sizes[name]["params_str"] = f"{model.coef_.size / 1_000_000:.2f}M coefficients"
        print(f"  {name}: {size_mb:.1f} MB")

    print("\n[3/3] Loading performance metrics from inference results...")
    inf_path = os.path.join(RESULTS_DIR, "inference_metrics.json")
    if os.path.exists(inf_path):
        with open(inf_path, "r") as f:
            inf = json.load(f)
        for name in sizes:
            if name in inf:
                sizes[name]["accuracy"] = inf[name].get("accuracy", "N/A")
                sizes[name]["roc_auc"] = inf[name].get("roc_auc", "N/A")

    out_path = os.path.join(RESULTS_DIR, "model_sizes.json")
    with open(out_path, "w") as f:
        json.dump(sizes, f, indent=2)
    print(f"\n[DONE] Saved to {out_path}")

    print("\nModel size summary:")
    for name, data in sizes.items():
        print(f"  {name:20s}: {data.get('size_str', 'N/A')}")


if __name__ == "__main__":
    main()