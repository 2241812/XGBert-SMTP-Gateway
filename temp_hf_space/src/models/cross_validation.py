"""
K-fold cross-validation for model evaluation.
Implements stratified k-fold cross-validation as specified in the paper.
Now includes DistilBERT per-fold training and progress tracking with ETA.
"""

import json
import os
import random
import time
import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, List
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from . import baseline_models
from .config import (
    TRAIN_DATA_PATH,
    LOG_DIR,
    PROGRESS_FILE,
    RANDOM_SEED,
    NUM_EPOCHS,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_SEQ_LENGTH,
    WEIGHT_DECAY,
    WARMUP_RATIO,
    LR_SCHEDULER_TYPE,
    FP16,
)


CV_RESULTS_PATH = os.path.join(LOG_DIR, "model_comparison", "ten_fold_cv_results.json")

SAMPLES_PER_FOLD = 25000
TRAIN_RATIO = 0.8
VAL_RATIO = 0.2


def _update_progress(
    status: str,
    percent: float,
    fold: int = 0,
    total_folds: int = 10,
    current_model: str = "",
    eta_seconds: float = 0,
    details: str = "",
) -> None:
    """Update the progress tracking file for dashboard/monitoring."""
    eta_str = ""
    if eta_seconds > 0:
        mins = int(eta_seconds // 60)
        secs = int(eta_seconds % 60)
        eta_str = f"{mins}m {secs}s"
    data = {
        "status": status,
        "progress": percent,
        "fold": fold,
        "total_folds": total_folds,
        "current_model": current_model,
        "eta": eta_str,
        "details": details,
    }
    try:
        os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
        with open(PROGRESS_FILE, "w") as f:
            json.dump(data, f)
    except Exception:
        pass


def _compute_macro_specificity(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    total = float(matrix.sum())
    specificities: List[float] = []
    for class_idx in range(num_classes):
        tp = float(matrix[class_idx, class_idx])
        fp = float(matrix[:, class_idx].sum() - tp)
        fn = float(matrix[class_idx, :].sum() - tp)
        tn = total - tp - fp - fn
        denominator = tn + fp
        specificities.append((tn / denominator) if denominator > 0 else 0.0)
    return float(np.mean(specificities))


def _compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
) -> Dict[str, Any]:
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    roc_auc = roc_auc_score(y_true_bin, y_proba, multi_class="ovr", average="macro")
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "specificity": _compute_macro_specificity(y_true, y_pred, num_classes=4),
        "roc_auc": float(roc_auc),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist(),
    }


def _sample_stratified(df: pd.DataFrame, n_per_class: int, seed: int) -> pd.DataFrame:
    """Sample n_per_class from each class with stratified sampling."""
    sampled = []
    for class_label in range(4):
        class_rows = df[df["label"] == class_label]
        n = min(n_per_class, len(class_rows))
        sampled.append(class_rows.sample(n=n, random_state=seed))
    result = pd.concat(sampled).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return result


def _train_distilbert_fold(
    train_urls: List[str],
    train_labels: List[int],
    val_urls: List[str],
    val_labels: List[int],
    fold_idx: int,
    device: torch.device,
) -> Dict[str, Any]:
    """Train DistilBERT on fold data and evaluate on validation set."""
    start_time = time.time()

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        device_name = torch.cuda.get_device_name(0)
        print(f"  [DistilBERT Fold {fold_idx+1}] GPU: {device_name}", flush=True)
    else:
        print(f"  [DistilBERT Fold {fold_idx+1}] Using CPU", flush=True)

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=4
    )
    model.to(device)

    train_dataset = Dataset.from_dict({"url": train_urls, "label": train_labels})
    val_dataset = Dataset.from_dict({"url": val_urls, "label": val_labels})

    def tokenize_fn(examples):
        return tokenizer(
            examples["url"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False,
        )

    tokenized_train = train_dataset.map(tokenize_fn, batched=True, remove_columns=["url"])
    tokenized_val = val_dataset.map(tokenize_fn, batched=True, remove_columns=["url"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    warmup_steps = int(len(tokenized_train) * WARMUP_RATIO / BATCH_SIZE)

    training_args = TrainingArguments(
        output_dir=os.path.join(LOG_DIR, f"temp_distilbert_fold_{fold_idx}"),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        logging_steps=100,
        eval_strategy="no",
        save_strategy="no",
        fp16=FP16 and cuda_available,
        dataloader_num_workers=0,
        report_to="none",
        seed=RANDOM_SEED,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        data_collator=data_collator,
    )

    print(f"  [DistilBERT Fold {fold_idx+1}] Training on {len(train_urls)} samples...", flush=True)
    trainer.train()
    train_time = time.time() - start_time
    print(f"  [DistilBERT Fold {fold_idx+1}] Training done in {train_time:.1f}s. Evaluating...", flush=True)

    del trainer
    torch.cuda.empty_cache() if cuda_available else None

    model.eval()
    val_preds = []
    val_proba = []

    for i in range(0, len(val_urls), BATCH_SIZE * 4):
        batch_urls = val_urls[i:i + BATCH_SIZE * 4]
        inputs = tokenizer(
            batch_urls,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=True,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            val_preds.extend(preds.cpu().numpy().tolist())
            val_proba.extend(probs.cpu().numpy().tolist())

    val_preds = np.array(val_preds)
    val_proba = np.array(val_proba)

    metrics = _compute_metrics(
        y_true=np.array(val_labels),
        y_pred=val_preds,
        y_proba=val_proba,
        model_name="DistilBERT",
    )
    metrics["training_time"] = train_time

    del model
    torch.cuda.empty_cache() if cuda_available else None

    return metrics


def run_kfold_cross_validation(
    model_type: str = "all",
    n_splits: int = 10,
    random_state: int = RANDOM_SEED,
    max_samples: int = 250000,
) -> Dict[str, Any]:
    """
    Run stratified k-fold cross-validation with per-fold 80/20 split.
    Each fold samples 25k stratified → 20k train / 5k val.
    Trains DistilBERT per fold when model_type includes 'distilbert' or 'all'.
    """
    print(f"\n{'=' * 60}")
    print(f"Starting {n_splits}-fold cross-validation")
    print(f"Per-fold sample: {SAMPLES_PER_FOLD} (80/20 split = {int(SAMPLES_PER_FOLD*TRAIN_RATIO)} train / {int(SAMPLES_PER_FOLD*VAL_RATIO)} val)")
    print(f"Model types: {model_type}")
    print(f"{'=' * 60}\n")

    full_train_df = pd.read_csv(TRAIN_DATA_PATH, engine="python")
    full_train_df["url"] = full_train_df["url"].astype(str)
    full_train_df["label"] = full_train_df["label"].astype(int)

    cuda_available = torch.cuda.is_available()
    device = torch.device("cuda" if cuda_available else "cpu")
    print(f"Device: {'CUDA GPU' if cuda_available else 'CPU'}\n")

    fold_results = []
    fold_times = []
    total_start_time = time.time()

    samples_per_class_per_fold = SAMPLES_PER_FOLD // 4

    for fold_idx in range(n_splits):
        fold_start = time.time()
        print(f"\n{'=' * 50}")
        print(f"FOLD {fold_idx + 1}/{n_splits}")
        print(f"{'=' * 50}")

        sampled_df = _sample_stratified(full_train_df, samples_per_class_per_fold, random_state + fold_idx)
        n_train = int(len(sampled_df) * TRAIN_RATIO)
        train_df = sampled_df.iloc[:n_train].reset_index(drop=True)
        val_df = sampled_df.iloc[n_train:].reset_index(drop=True)

        print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

        train_urls = train_df["url"].tolist()
        train_labels = train_df["label"].tolist()
        val_urls = val_df["url"].tolist()
        val_labels = val_df["label"].tolist()

        y_train_np = np.array(train_labels)
        y_val_np = np.array(val_labels)

        fold_result = {"fold": fold_idx + 1}

        _, X_train_features, X_val_features = baseline_models.prepare_features(train_urls, val_urls)

        total_models = 4 if model_type in ["all", "distilbert"] else 3
        model_num = 0

        if model_type in ["all", "baseline", "lr"]:
            model_num += 1
            _update_progress(
                "training",
                fold_idx / n_splits * 100,
                fold_idx + 1,
                n_splits,
                "LogisticRegression",
                0,
            )
            lr_model, lr_scaler, lr_metrics = baseline_models.run_logistic_regression(
                X_train_features, X_val_features, y_train_np, y_val_np
            )
            lr_predictions = lr_model.predict(lr_scaler.transform(X_val_features))
            lr_proba = lr_model.predict_proba(lr_scaler.transform(X_val_features))
            fold_result["logistic_regression"] = _compute_metrics(
                y_true=y_val_np,
                y_pred=lr_predictions,
                y_proba=lr_proba,
                model_name=lr_metrics["model"],
            )
            print(f"  LR: accuracy={fold_result['logistic_regression']['accuracy']:.4f}", flush=True)

        if model_type in ["all", "baseline", "rf"]:
            model_num += 1
            remaining = (n_splits - fold_idx - 1) * 5 if fold_times else 0
            _update_progress(
                "training",
                (fold_idx + model_num / total_models) / n_splits * 100,
                fold_idx + 1,
                n_splits,
                "RandomForest",
                remaining * 20,
            )
            rf_model, rf_metrics = baseline_models.run_random_forest(
                X_train_features, X_val_features, y_train_np, y_val_np
            )
            rf_predictions = rf_model.predict(X_val_features)
            rf_proba = rf_model.predict_proba(X_val_features)
            fold_result["random_forest"] = _compute_metrics(
                y_true=y_val_np,
                y_pred=rf_predictions,
                y_proba=rf_proba,
                model_name=rf_metrics["model"],
            )
            print(f"  RF: accuracy={fold_result['random_forest']['accuracy']:.4f}", flush=True)

        if model_type in ["all", "baseline", "xgb"]:
            model_num += 1
            remaining = (n_splits - fold_idx - 1) * 30 if fold_times else 0
            _update_progress(
                "training",
                (fold_idx + model_num / total_models) / n_splits * 100,
                fold_idx + 1,
                n_splits,
                "XGBoost",
                remaining * 20,
            )
            if baseline_models.XGBClassifier is None:
                fold_result["xgboost"] = {"status": "skipped_missing_dependency"}
            else:
                xgb_model, xgb_metrics = baseline_models.run_xgboost(
                    X_train_features, X_val_features, y_train_np, y_val_np
                )
                xgb_predictions = xgb_model.predict(X_val_features)
                xgb_proba = xgb_model.predict_proba(X_val_features)
                fold_result["xgboost"] = _compute_metrics(
                    y_true=y_val_np,
                    y_pred=xgb_predictions,
                    y_proba=xgb_proba,
                    model_name=xgb_metrics["model"],
                )
                print(f"  XGB: accuracy={fold_result['xgboost']['accuracy']:.4f}", flush=True)

        if model_type in ["all", "distilbert"]:
            model_num += 1
            remaining = (n_splits - fold_idx - 1) * 30 if fold_times else 0
            _update_progress(
                "training",
                (fold_idx + model_num / total_models) / n_splits * 100,
                fold_idx + 1,
                n_splits,
                "DistilBERT",
                remaining * 30,
            )
            print(f"  Training DistilBERT on {len(train_urls)} samples...", flush=True)
            distilbert_metrics = _train_distilbert_fold(
                train_urls, train_labels, val_urls, val_labels, fold_idx, device
            )
            fold_result["distilbert"] = distilbert_metrics
            print(f"  DistilBERT: accuracy={fold_result['distilbert']['accuracy']:.4f}", flush=True)

        fold_time = time.time() - fold_start
        fold_times.append(fold_time)
        fold_results.append(fold_result)

        avg_fold_time = sum(fold_times) / len(fold_times)
        eta = avg_fold_time * (n_splits - fold_idx - 1)

        print(f"\n  Fold {fold_idx + 1} completed in {fold_time:.1f}s")
        print(f"  ETA remaining: {eta / 60:.1f} min ({eta:.0f}s)")

        _update_progress(
            "fold_complete",
            (fold_idx + 1) / n_splits * 100,
            fold_idx + 1,
            n_splits,
            "idle",
            eta,
        )

    total_time = time.time() - total_start_time
    print(f"\n{'=' * 60}")
    print(f"All {n_splits} folds completed in {total_time / 60:.1f} minutes")
    print(f"{'=' * 60}\n")

    aggregated = _aggregate_fold_results(fold_results, model_type)
    aggregated["max_samples"] = max_samples
    aggregated["total_time_seconds"] = total_time

    os.makedirs(os.path.dirname(CV_RESULTS_PATH), exist_ok=True)
    with open(CV_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2)

    _update_progress("complete", 100, n_splits, n_splits, "", 0)

    print(f"Results saved to: {CV_RESULTS_PATH}")
    return aggregated


def _aggregate_fold_results(
    fold_results: List[Dict[str, Any]],
    model_type: str,
) -> Dict[str, Any]:
    """Aggregate metrics across folds, computing mean and std."""
    metrics_to_aggregate = [
        "accuracy", "precision", "recall", "f1", "specificity", "roc_auc", "training_time"
    ]
    model_keys = ["logistic_regression", "random_forest", "xgboost", "distilbert"]
    model_labels = {
        "logistic_regression": "LogisticRegression",
        "random_forest": "RandomForest",
        "xgboost": "XGBoost",
        "distilbert": "DistilBERT",
    }

    aggregated = {
        "n_splits": len(fold_results),
        "model_type": model_type,
        "folds": fold_results,
        "summary": {},
    }

    for model_key in model_keys:
        if model_type not in ["all", "baseline"] and model_key not in model_type:
            continue

        model_results = []
        for fold in fold_results:
            if model_key in fold and isinstance(fold[model_key], dict):
                if fold[model_key].get("status") not in ("skipped_in_baseline_cv", "skipped_missing_dependency"):
                    model_results.append(fold[model_key])

        if not model_results:
            continue

        model_agg = {"model": model_labels.get(model_key, model_key)}
        for metric in metrics_to_aggregate:
            values = []
            for result in model_results:
                if metric in result and isinstance(result[metric], (int, float)):
                    values.append(float(result[metric]))

            if values:
                model_agg[f"mean_{metric}"] = float(np.mean(values))
                model_agg[f"std_{metric}"] = float(np.std(values))
                model_agg[f"{metric}_values"] = values

        aggregated["summary"][model_labels.get(model_key, model_key)] = model_agg

    return aggregated


def run_baseline_kfold(max_samples: int = 250000) -> Dict[str, Any]:
    """Run k-fold CV for baseline models only (no DistilBERT)."""
    return run_kfold_cross_validation(model_type="baseline", max_samples=max_samples)


def run_distilbert_kfold(max_samples: int = 250000) -> Dict[str, Any]:
    """Run k-fold CV for DistilBERT only."""
    return run_kfold_cross_validation(model_type="distilbert", max_samples=max_samples)


if __name__ == "__main__":
    print("Testing 10-fold cross-validation with DistilBERT...")
    results = run_kfold_cross_validation(model_type="all", n_splits=10)
    print(json.dumps(results, indent=2))