"""
DistilBERT Model Training Script
================================
A standalone training script for DistilBERT URL phishing detection.
Based on the tinybert_trainer.py but using DistilBERT for better accuracy.

Usage:
    python src/models/distilbert_trainer.py

Configuration:
    Modify src/models/config.py to adjust paths, hyperparameters, and options.
"""

import json
import logging
import os
import random
import sys
from datetime import datetime
from typing import Any, Dict

import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from .callbacks import ProgressCallback, TrainingMonitor
from .config import (
    BATCH_SIZE,
    EVAL_DIR,
    EXPERIMENTS_DIR,
    FIXED_EVAL_SAMPLES,
    FP16,
    GENERATE_PLOTS,
    GRADIENT_ACCUMULATION_STEPS,
    LEARNING_RATE,
    LOG_DIR,
    LOGGING_STEPS,
    LR_SCHEDULER_TYPE,
    MAX_TRAIN_SAMPLES,
    MODEL_NAME_DISTILBERT,
    MODEL_OUTPUT_DIR_DISTILBERT,
    NUM_EPOCHS,
    NUM_WORKERS,
    PLOT_DIR,
    SAVE_STEPS,
    QUALITY_GATES,
    RANDOM_SEED,
    TEST_DATA_PATH,
    TRAIN_DATA_PATH,
    WARMUP_RATIO,
    WEIGHT_DECAY,
)
from .data import load_training_data, tokenize_function
from .evaluation import create_or_load_fixed_eval_indices
from .experiment_tracking import finalize_experiment_run, start_experiment_run
from .quality_gates import compute_per_class_recall, evaluate_quality_gates
from .viz import generate_visualizations


def setup_logging(log_file):
    """Setup logging configuration."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """Compute metrics for evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == labels).mean())
    precision = float(precision_score(labels, predictions, average="weighted", zero_division=0))
    recall = float(recall_score(labels, predictions, average="weighted", zero_division=0))
    f1 = float(f1_score(labels, predictions, average="weighted", zero_division=0))
    class_recalls = compute_per_class_recall(labels, predictions, num_classes=4)
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "recall_class_0": class_recalls[0],
        "recall_class_1": class_recalls[1],
        "recall_class_2": class_recalls[2],
        "recall_class_3": class_recalls[3],
    }


def _compute_extended_eval_metrics(y_true: np.ndarray, y_logits: np.ndarray) -> Dict[str, Any]:
    y_pred = np.argmax(y_logits, axis=-1)
    y_prob = torch.softmax(torch.tensor(y_logits), dim=-1).cpu().numpy()
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])
    eval_confusion_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3]).tolist()
    eval_roc_auc = float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average="macro"))
    confusion_np = np.array(eval_confusion_matrix)
    total = float(confusion_np.sum())
    specificity_values = []
    for class_idx in range(confusion_np.shape[0]):
        tp = float(confusion_np[class_idx, class_idx])
        fp = float(confusion_np[:, class_idx].sum() - tp)
        fn = float(confusion_np[class_idx, :].sum() - tp)
        tn = total - tp - fp - fn
        denominator = tn + fp
        specificity_values.append((tn / denominator) if denominator > 0 else 0.0)
    eval_specificity = float(np.mean(specificity_values))
    per_class_recall = compute_per_class_recall(y_true, y_pred, num_classes=4)
    quality_gate_result = evaluate_quality_gates(per_class_recall, QUALITY_GATES)
    return {
        "eval_specificity": eval_specificity,
        "eval_roc_auc": eval_roc_auc,
        "eval_confusion_matrix": eval_confusion_matrix,
        "per_class_recall": per_class_recall,
        "quality_gates": quality_gate_result,
        "deployment_recommended": quality_gate_result["passed"],
    }


class WeightedTrainer(Trainer):
    """Custom Trainer that applies class weights to the loss function."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fn = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_model():
    """Train the DistilBERT model."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    os.makedirs(MODEL_OUTPUT_DIR_DISTILBERT, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    log_file = os.path.join(LOG_DIR, f"distilbert_training_{timestamp}.log")
    logger = setup_logging(log_file)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    effective_fp16 = FP16 or cuda_available

    print(f"\n{'=' * 60}")
    print("smtpBERT DistilBERT Model Training")
    print(f"{'=' * 60}")
    print(f"Model: {MODEL_NAME_DISTILBERT}")
    print(f"Epochs: {NUM_EPOCHS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Output: {MODEL_OUTPUT_DIR_DISTILBERT}")
    print(f"Torch: {torch.__version__}")
    print(f"Torch CUDA build: {torch.version.cuda}")
    if cuda_available:
        print(f"Device: cuda ({device_name})")
        print(f"FP16: enabled ({effective_fp16})")
    else:
        print("Device: cpu (CUDA unavailable)")
        print("FP16: disabled (CUDA unavailable)")
    print(f"{'=' * 60}\n")

    monitor = TrainingMonitor()
    monitor.on_train_start()

    logger.info("Starting DistilBERT training pipeline")
    logger.info(f"Model: {MODEL_NAME_DISTILBERT}")
    logger.info(f"Training data: {TRAIN_DATA_PATH}")
    logger.info(f"Test data: {TEST_DATA_PATH}")
    logger.info("Torch version: %s", torch.__version__)
    logger.info("Torch CUDA build: %s", torch.version.cuda)
    logger.info("CUDA available: %s", cuda_available)
    if cuda_available:
        logger.info("CUDA device: %s", device_name)
        logger.info("FP16 enabled: %s", effective_fp16)
    else:
        logger.warning("CUDA is unavailable. DistilBERT training will run on CPU.")

    train_dataset, test_dataset = load_training_data(TRAIN_DATA_PATH, TEST_DATA_PATH)

    eval_indices_file = os.path.join(EVAL_DIR, "fixed_eval_indices.json")
    eval_indices = create_or_load_fixed_eval_indices(
        labels=[int(sample["label"]) for sample in test_dataset],
        output_path=eval_indices_file,
        max_samples=FIXED_EVAL_SAMPLES,
        seed=RANDOM_SEED,
    )
    if eval_indices:
        test_dataset = test_dataset.select(eval_indices)
        logger.info("Loaded fixed evaluation split with %d samples", len(test_dataset))

    if MAX_TRAIN_SAMPLES and len(train_dataset) > MAX_TRAIN_SAMPLES:
        print(f"\n[DATA] Sampling {MAX_TRAIN_SAMPLES} training samples (stratified)...")
        labels = [d["label"] for d in train_dataset]
        sampled_indices = []
        samples_per_class = MAX_TRAIN_SAMPLES // 4
        for class_label in range(4):
            class_indices = [i for i, l in enumerate(labels) if l == class_label]
            random.seed(RANDOM_SEED)
            sampled = random.sample(class_indices, min(samples_per_class, len(class_indices)))
            sampled_indices.extend(sampled)
        random.shuffle(sampled_indices)
        train_dataset = train_dataset.select(sampled_indices)
        print(f"  Sampled dataset size: {len(train_dataset)}")

    run_context = start_experiment_run(
        experiments_dir=EXPERIMENTS_DIR,
        run_type="training",
        model_name="distilbert",
        config={
            "model_name": MODEL_NAME_DISTILBERT,
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "max_train_samples": MAX_TRAIN_SAMPLES,
            "fixed_eval_samples": FIXED_EVAL_SAMPLES,
            "seed": RANDOM_SEED,
        },
        dataset={
            "train_path": TRAIN_DATA_PATH,
            "test_path": TEST_DATA_PATH,
            "train_samples": len(train_dataset),
            "eval_samples": len(test_dataset),
        },
    )

    print("\n[MODEL] Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_DISTILBERT)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME_DISTILBERT, num_labels=4
    )

    print("[MODEL] Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["url"]
    )
    tokenized_test = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["url"]
    )
    tokenized_train = tokenized_train.rename_column("label", "labels")
    tokenized_test = tokenized_test.rename_column("label", "labels")

    data_collator = DataCollatorWithPadding(tokenizer)

    train_labels = [sample["labels"] for sample in tokenized_train]
    class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2, 3]), y=train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    print(f"[CLASS WEIGHTS] Benign: {class_weights[0]:.3f}, Phishing: {class_weights[1]:.3f}, "
          f"Malware: {class_weights[2]:.3f}, Defacement: {class_weights[3]:.3f}")

    training_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR_DISTILBERT,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_dir=os.path.join(LOG_DIR, "runs"),
        logging_steps=LOGGING_STEPS,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        load_best_model_at_end=False,
        fp16=effective_fp16,
        no_cuda=not cuda_available,
        dataloader_num_workers=NUM_WORKERS,
        report_to="none",
        seed=RANDOM_SEED,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type=LR_SCHEDULER_TYPE,
        max_grad_norm=1.0,
    )

    monitor = TrainingMonitor()
    progress_callback = ProgressCallback(monitor)
    trainer = WeightedTrainer(
        class_weights=class_weights_tensor,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[progress_callback],
    )

    print("\n[TRAIN] Starting training...")
    print("-" * 60)

    train_result = trainer.train()
    eval_results = trainer.evaluate()
    prediction_output = trainer.predict(tokenized_test)
    y_true = prediction_output.label_ids
    y_logits = prediction_output.predictions
    extended_metrics = _compute_extended_eval_metrics(y_true, y_logits)

    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"Training Loss: {train_result.training_loss:.4f}")
    print(f"Evaluation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Evaluation Loss: {eval_results['eval_loss']:.4f}")
    print(f"Evaluation Specificity: {extended_metrics['eval_specificity']:.4f}")
    print(f"Evaluation ROC-AUC: {extended_metrics['eval_roc_auc']:.4f}")
    print(f"Quality Gates Passed: {extended_metrics['quality_gates']['passed']}")
    print("=" * 60)

    print("\n[SAVE] Saving model...")
    trainer.save_model(MODEL_OUTPUT_DIR_DISTILBERT)
    tokenizer.save_pretrained(MODEL_OUTPUT_DIR_DISTILBERT)

    metrics = {
        "model_name": MODEL_NAME_DISTILBERT,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "num_epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "train_loss": train_result.training_loss,
        "eval_accuracy": eval_results["eval_accuracy"],
        "eval_loss": eval_results["eval_loss"],
        "eval_precision": eval_results.get("eval_precision", 0.0),
        "eval_recall": eval_results.get("eval_recall", 0.0),
        "eval_f1": eval_results.get("eval_f1", 0.0),
        "eval_specificity": extended_metrics["eval_specificity"],
        "eval_roc_auc": extended_metrics["eval_roc_auc"],
        "eval_confusion_matrix": extended_metrics["eval_confusion_matrix"],
        "per_class_recall": extended_metrics["per_class_recall"],
        "quality_gates": extended_metrics["quality_gates"],
        "deployment_recommended": extended_metrics["deployment_recommended"],
        "training_time": train_result.metrics.get("train_runtime", 0.0),
    }

    metrics_file = os.path.join(MODEL_OUTPUT_DIR_DISTILBERT, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_file}")

    run_status = "accepted" if quality_gate_result["passed"] else "rejected"
    finalize_experiment_run(
        run_context=run_context,
        status=run_status,
        metrics=metrics,
        quality_gates=quality_gate_result,
        artifacts=[metrics_file, MODEL_OUTPUT_DIR_DISTILBERT, log_file],
        notes="DistilBERT training run with fixed evaluation split and quality gates.",
    )

    if GENERATE_PLOTS:
        print("\n[PLOT] Generating visualizations...")
        generate_visualizations(metrics, log_file)

    monitor.on_train_end()

    print("\n[DONE] Training complete!")
    print(f"Model saved to: {MODEL_OUTPUT_DIR_DISTILBERT}")
    print(f"Logs saved to: {LOG_DIR}")

    return metrics


def test_model():
    """Test the DistilBERT model with sample URLs."""
    if not os.path.exists(MODEL_OUTPUT_DIR_DISTILBERT):
        print(f"\nWarning: Model not found at {MODEL_OUTPUT_DIR_DISTILBERT}")
        print("Run training first with: python distilbert_trainer.py")
        return

    print("\n" + "=" * 60)
    print("DISTILBERT MODEL TESTING")
    print("=" * 60)

    import torch.nn.functional as F

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR_DISTILBERT)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR_DISTILBERT)

    labels = {0: "Benign", 1: "Phishing", 2: "Malware", 3: "Defacement"}

    test_urls = [
        ("https://google.com/search?q=test", "Benign"),
        ("http://malicious-phishing.example.com/login", "Phishing"),
        ("https://safe-bank.com/portal", "Benign"),
        ("http://malware-download.evil.org/payload.exe", "Malware"),
        ("http://fake-paypal.scam.net/login", "Phishing"),
        ("http://defaced-site.vandal.com/index", "Defacement"),
        ("https://github.com/search", "Benign"),
    ]

    print("\nSample Predictions:")
    print("-" * 60)

    for url, expected in test_urls:
        inputs = tokenizer(url, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = F.softmax(logits, dim=-1)
        pred_idx = probs.argmax(dim=-1).item()
        prediction = labels.get(pred_idx, f"Class-{pred_idx}")
        correct = "✓" if prediction == expected else "✗"

        print(f"URL: {url[:50]}...")
        print(f"  Prediction: {prediction} (expected: {expected}) {correct}")
        print()

    print("=" * 60)


def evaluate_saved_model() -> Dict[str, Any]:
    """Evaluate saved DistilBERT model and update training_metrics.json without retraining."""
    if not os.path.exists(MODEL_OUTPUT_DIR_DISTILBERT):
        raise FileNotFoundError(f"Saved model not found: {MODEL_OUTPUT_DIR_DISTILBERT}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_OUTPUT_DIR_DISTILBERT)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_OUTPUT_DIR_DISTILBERT)

    _, test_dataset = load_training_data(TRAIN_DATA_PATH, TEST_DATA_PATH)
    eval_indices_file = os.path.join(EVAL_DIR, "fixed_eval_indices.json")
    eval_indices = create_or_load_fixed_eval_indices(
        labels=[int(sample["label"]) for sample in test_dataset],
        output_path=eval_indices_file,
        max_samples=FIXED_EVAL_SAMPLES,
        seed=RANDOM_SEED,
    )
    if eval_indices:
        test_dataset = test_dataset.select(eval_indices)

    tokenized_test = test_dataset.map(
        lambda x: tokenize_function(x, tokenizer), batched=True, remove_columns=["url"]
    )
    tokenized_test = tokenized_test.rename_column("label", "labels")
    data_collator = DataCollatorWithPadding(tokenizer)

    cuda_available = torch.cuda.is_available()
    eval_args = TrainingArguments(
        output_dir=MODEL_OUTPUT_DIR_DISTILBERT,
        per_device_eval_batch_size=BATCH_SIZE,
        fp16=(FP16 or cuda_available),
        no_cuda=not cuda_available,
        dataloader_num_workers=NUM_WORKERS,
        report_to="none",
        seed=RANDOM_SEED,
    )
    trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    eval_results = trainer.evaluate(eval_dataset=tokenized_test)
    prediction_output = trainer.predict(tokenized_test)
    extended_metrics = _compute_extended_eval_metrics(
        y_true=prediction_output.label_ids,
        y_logits=prediction_output.predictions,
    )

    metrics_file = os.path.join(MODEL_OUTPUT_DIR_DISTILBERT, "training_metrics.json")
    existing_metrics: Dict[str, Any] = {}
    if os.path.exists(metrics_file):
        with open(metrics_file, "r", encoding="utf-8") as file:
            existing_metrics = json.load(file)

    existing_metrics.update(
        {
            "model_name": MODEL_NAME_DISTILBERT,
            "test_samples": len(test_dataset),
            "eval_accuracy": eval_results.get("eval_accuracy", 0.0),
            "eval_loss": eval_results.get("eval_loss", 0.0),
            "eval_precision": eval_results.get("eval_precision", 0.0),
            "eval_recall": eval_results.get("eval_recall", 0.0),
            "eval_f1": eval_results.get("eval_f1", 0.0),
            "eval_specificity": extended_metrics["eval_specificity"],
            "eval_roc_auc": extended_metrics["eval_roc_auc"],
            "eval_confusion_matrix": extended_metrics["eval_confusion_matrix"],
            "per_class_recall": extended_metrics["per_class_recall"],
            "quality_gates": extended_metrics["quality_gates"],
            "deployment_recommended": extended_metrics["deployment_recommended"],
        }
    )

    with open(metrics_file, "w", encoding="utf-8") as file:
        json.dump(existing_metrics, file, indent=2)

    print(f"[Eval-Only] Metrics updated at: {metrics_file}")
    return existing_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="smtpBERT DistilBERT Training Script"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "test", "both", "evaluate"],
        default="train",
        help="Choose mode: train, test, both, or evaluate",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Training batch size")
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate"
    )
    args = parser.parse_args()

    if args.epochs is not None:
        global NUM_EPOCHS
        NUM_EPOCHS = args.epochs
    if args.batch_size is not None:
        global BATCH_SIZE
        BATCH_SIZE = args.batch_size
    if args.learning_rate is not None:
        global LEARNING_RATE
        LEARNING_RATE = args.learning_rate

    print("\n" + "=" * 60)
    print("smtpBERT DistilBERT Training Script")
    print(f"Epochs: {NUM_EPOCHS}, Batch Size: {BATCH_SIZE}, LR: {LEARNING_RATE}")
    print("=" * 60)

    if args.mode in ["train", "both"]:
        try:
            metrics = train_model()
        except Exception as e:
            print(f"\nError during training: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    if args.mode in ["test", "both"]:
        test_model()
    if args.mode == "evaluate":
        try:
            evaluate_saved_model()
        except Exception as e:
            print(f"\nError during evaluation-only pass: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "=" * 60)
    print("SCRIPT COMPLETE")
    print("=" * 60)
    print(f"\nTo view logs: Check {LOG_DIR}/")
    print(f"To view plots: Check {PLOT_DIR}/")
