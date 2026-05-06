"""
compute_distilbert_shap.py
Computes SHAP values for DistilBERT model using shap.Explainer (token-level attribution).
Uses HuggingFace tokenizer internally for proper subword tokenization.

Usage:
    D:\smtpBERT\venv\Scripts\python.exe -m src.models.compute_distilbert_shap

Note: Uses shap.Explainer with the tokenizer for proper token-level attribution.
      Runs ~10-15 minutes with 100 eval samples.
"""

import json
import os
import sys
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.models import config as model_config

BASE_DIR = model_config.BASE_DIR
RESULTS_DIR = os.path.join(model_config.LOG_DIR, "model_comparison")
EVAL_DIR = model_config.EVAL_DIR
DATA_DIR = os.path.join(BASE_DIR, "data")
FIGURES_DIR = os.path.join(model_config.LOG_DIR, "figures")

CLASS_NAMES = ["benign", "phishing", "malware", "defacement"]

EVAL_SIZE = 200
SHAP_SAMPLE_SIZE = 100
MAX_SEQ_LENGTH = 128


def load_fixed_eval_indices():
    eval_file = os.path.join(EVAL_DIR, "fixed_eval_indices.json")
    if not os.path.exists(eval_file):
        raise FileNotFoundError(f"Fixed eval indices not found: {eval_file}")
    with open(eval_file, "r") as f:
        return json.load(f)["indices"]


def main():
    print("=" * 60)
    print("DistilBERT SHAP Computation (shap.Explainer — Token-Level)")
    print(f"  Evaluation: {EVAL_SIZE} URLs (SHAP computed on {SHAP_SAMPLE_SIZE})")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("\n[1/5] Loading DistilBERT...")
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    model_path = model_config.MODEL_OUTPUT_DIR_DISTILBERT
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"DistilBERT model not found at {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, local_files_only=True, num_labels=4
    ).to(device)
    model.eval()
    print("  Model loaded.")

    print("\n[2/5] Loading test URLs...")
    eval_indices = load_fixed_eval_indices()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_data.csv"), engine="python")
    test_df = test_df.iloc[eval_indices].reset_index(drop=True)
    all_urls = test_df["url"].astype(str).tolist()
    labels = test_df["label"].astype(int).values
    print(f"  Total URLs: {len(all_urls)}, labels: {np.bincount(labels)}")

    eval_urls = all_urls[:EVAL_SIZE]
    eval_sample = eval_urls[:SHAP_SAMPLE_SIZE]
    print(f"  Eval set: {len(eval_urls)}, SHAP sample: {len(eval_sample)}")

    print("\n[3/5] Building shap.Explainer with tokenizer...")
    start_time = time.time()

    def predict_fn(texts):
        model.eval()
        inputs = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        return torch.softmax(outputs.logits, dim=-1).cpu().numpy()

    print("  Creating explainer (this may take a moment)...")
    explainer = shap.Explainer(predict_fn, tokenizer, max_evals=int(SHAP_SAMPLE_SIZE * 50))

    print("\n[4/5] Computing SHAP values (runs once, all classes)...")
    print("  This takes ~10-15 min total. Computing...")
    shap_values = explainer(eval_sample)
    print(f"  Done. SHAP shape: {shap_values.shape}")
    print(f"  Total SHAP computation time: {time.time() - start_time:.0f}s")

    print("\n[5/5] Saving SHAP summary plots (Token-Level Aggregation)...")

    multi_dir = os.path.join(FIGURES_DIR, "distilbert_shap")
    os.makedirs(multi_dir, exist_ok=True)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        token_shap_sums = {}
        token_counts = {}

        for i in range(len(eval_sample)):
            tokens = shap_values.data[i]
            vals = shap_values.values[i][:, cls_idx]

            for token, val in zip(tokens, vals):
                tok_str = str(token).strip()
                if not tok_str:
                    continue
                if tok_str not in token_shap_sums:
                    token_shap_sums[tok_str] = 0.0
                    token_counts[tok_str] = 0
                token_shap_sums[tok_str] += abs(val)
                token_counts[tok_str] += 1

        mean_shap = {tok: token_shap_sums[tok] / token_counts[tok] for tok in token_shap_sums}
        top_tokens = sorted(mean_shap.items(), key=lambda x: x[1], reverse=True)[:20]

        if not top_tokens:
            print(f"  No tokens found for {cls_name}. Skipping.")
            continue

        labels = [x[0] for x in top_tokens][::-1]
        values = [x[1] for x in top_tokens][::-1]

        plt.figure(figsize=(10, 8))
        plt.barh(labels, values, color="#0891b2")
        plt.title(f"DistilBERT SHAP — {cls_name.capitalize()}\nTop 20 Tokens by Mean |SHAP|", fontsize=12)
        plt.xlabel("Mean |SHAP value|", fontsize=10)
        plt.tight_layout()
        out_path = os.path.join(multi_dir, f"{cls_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {out_path}")

    print(f"\n[DONE] DistilBERT SHAP plots saved to {multi_dir}/")
    print("       Run the dashboard to view them in the XAI / SHAP tab.")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()