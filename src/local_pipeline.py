"""
Local-only CLI workflow for smtpBERT training and SMTP gateway testing.
"""

import argparse
import json
import os
import smtplib
import subprocess
import sys
import time
from email.message import EmailMessage
from typing import Optional, Dict, Any


def train_baseline() -> None:
    from src.models import baseline_models

    baseline_models.main()


def train_distilbert(epochs: Optional[int], batch_size: Optional[int], learning_rate: Optional[float]) -> None:
    from src.models import distilbert_trainer

    if epochs is not None:
        distilbert_trainer.NUM_EPOCHS = int(epochs)
    if batch_size is not None:
        distilbert_trainer.BATCH_SIZE = int(batch_size)
    if learning_rate is not None:
        distilbert_trainer.LEARNING_RATE = float(learning_rate)

    distilbert_trainer.train_model()


def test_distilbert() -> None:
    from src.models import distilbert_trainer

    distilbert_trainer.test_model()


def evaluate_distilbert(
    retrain: bool = False,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> None:
    from src.models import distilbert_trainer

    if retrain:
        print("Retraining DistilBERT before evaluation...")
        train_distilbert(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
    distilbert_trainer.evaluate_saved_model()


def run_baseline_ten_fold_cv() -> None:
    from src.models.cross_validation import run_kfold_cross_validation

    run_kfold_cross_validation(model_type="baseline", n_splits=10)


def run_ten_fold_comparison() -> None:
    """Run full 10-fold CV comparing all models (LR, RF, XGBoost, DistilBERT)."""
    from src.models.cross_validation import run_kfold_cross_validation

    print("\n" + "=" * 60)
    print("10-FOLD CROSS-VALIDATION: ALL MODELS")
    print("Models: LogisticRegression, RandomForest, XGBoost, DistilBERT")
    print("Per fold: 25k sample (20k train / 5k val)")
    print("=" * 60 + "\n")

    results = run_kfold_cross_validation(model_type="all", n_splits=10)

    cv_summary_path = os.path.join(
        os.path.dirname(__file__), "models", "..", "logs", "model_comparison", "ten_fold_cv_results.json"
    )
    cv_summary_path = os.path.normpath(cv_summary_path)

    print("\n" + "=" * 60)
    print("10-FOLD CV SUMMARY (Mean ± Std across folds)")
    print("=" * 60)
    summary = results.get("summary", {})
    for model_name, metrics in summary.items():
        print(f"\n{model_name}:")
        for metric in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            mean_key = f"mean_{metric}"
            std_key = f"std_{metric}"
            if mean_key in metrics:
                print(f"  {metric}: {metrics[mean_key]:.4f} ± {metrics.get(std_key, 0):.4f}")

    print(f"\nResults saved to: {cv_summary_path}")

    update_dashboard_with_cv_results(results)
    print("\nDashboard updated with 10-fold CV results.")


def gather_dashboard_metrics(
    retrain_distilbert: bool = False,
    run_baseline_cv: bool = False,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
) -> None:
    """Collect metrics for dashboard from existing artifacts, with optional retraining."""
    from src.models import config as model_config

    holdout_results_path = os.path.join(model_config.LOG_DIR, "model_comparison", "model_comparison.json")
    distilbert_metrics_path = os.path.join(model_config.MODEL_OUTPUT_DIR_DISTILBERT, "training_metrics.json")

    if os.path.exists(model_config.MODEL_OUTPUT_DIR_DISTILBERT):
        mode_label = "with retrain" if retrain_distilbert else "without retrain"
        print(f"Refreshing DistilBERT metrics from saved model ({mode_label})...")
        evaluate_distilbert(
            retrain=retrain_distilbert,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
    if run_baseline_cv:
        print("Running baseline ten-fold cross-validation (retrain across folds)...")
        run_baseline_ten_fold_cv()

    holdout_data = {"results": []}
    if os.path.exists(holdout_results_path):
        with open(holdout_results_path, "r", encoding="utf-8") as file:
            holdout_data = json.load(file)

    results = holdout_data.get("results", [])
    results = [item for item in results if item.get("model") != "DistilBERT"]

    if os.path.exists(distilbert_metrics_path):
        with open(distilbert_metrics_path, "r", encoding="utf-8") as file:
            distilbert = json.load(file)
        results.append(
            {
                "model": "DistilBERT",
                "accuracy": distilbert.get("eval_accuracy", 0.0),
                "precision": distilbert.get("eval_precision", 0.0),
                "recall": distilbert.get("eval_recall", 0.0),
                "f1": distilbert.get("eval_f1", 0.0),
                "roc_auc": distilbert.get("eval_roc_auc", 0.0),
                "confusion_matrix": distilbert.get("eval_confusion_matrix", [[0, 0, 0, 0]] * 4),
                "training_time": distilbert.get("training_time", 0.0),
                "per_class_recall": distilbert.get("per_class_recall", {}),
                "quality_gates": distilbert.get("quality_gates", {}),
                "deployment_recommended": distilbert.get("deployment_recommended", False),
            }
        )
    else:
        print(f"Warning: DistilBERT metrics file not found: {distilbert_metrics_path}")

    holdout_data["results"] = results
    os.makedirs(os.path.dirname(holdout_results_path), exist_ok=True)
    with open(holdout_results_path, "w", encoding="utf-8") as file:
        json.dump(holdout_data, file, indent=2)

    print(f"Dashboard metrics refreshed at: {holdout_results_path}")


def update_dashboard_with_cv_results(cv_results: Dict[str, Any]) -> None:
    """Merge 10-fold CV summary into the dashboard model_comparison.json."""
    from src.models import config as model_config

    dashboard_path = os.path.join(model_config.LOG_DIR, "model_comparison", "model_comparison.json")

    dashboard_data = {"results": []}
    if os.path.exists(dashboard_path):
        try:
            with open(dashboard_path, "r", encoding="utf-8") as f:
                dashboard_data = json.load(f)
        except Exception:
            pass

    existing_models = {r["model"] for r in dashboard_data.get("results", [])}

    summary = cv_results.get("summary", {})
    for model_name, metrics in summary.items():
        if model_name in existing_models:
            continue

        result_entry = {
            "model": model_name,
            "accuracy": metrics.get("mean_accuracy", 0.0),
            "precision": metrics.get("mean_precision", 0.0),
            "recall": metrics.get("mean_recall", 0.0),
            "f1": metrics.get("mean_f1", 0.0),
            "roc_auc": metrics.get("mean_roc_auc", 0.0),
            "confusion_matrix": [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            "training_time": metrics.get("mean_training_time", 0.0),
            "per_class_recall": {},
            "quality_gates": {"passed": False, "checks": {}},
            "deployment_recommended": False,
            "source": "ten_fold_cv",
        }
        dashboard_data["results"].append(result_entry)

    os.makedirs(os.path.dirname(dashboard_path), exist_ok=True)
    with open(dashboard_path, "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, indent=2)


def run_gateway(smtp_port: Optional[int] = None) -> None:
    if smtp_port is not None:
        os.environ["SMTP_PORT"] = str(smtp_port)
    from src.gateway.gateway import run_gateway

    run_gateway()


def run_dashboard() -> None:
    subprocess.run([sys.executable, "-m", "streamlit", "run", "src/app.py"], check=True)


def _start_gateway_process(smtp_port: int) -> subprocess.Popen:
    env = os.environ.copy()
    env["SMTP_PORT"] = str(smtp_port)
    proc = subprocess.Popen([sys.executable, "-m", "src.gateway.gateway"], env=env)
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError("Gateway failed to start.")
    return proc


def send_mock_email(
    smtp_host: str,
    smtp_port: int,
    sender: str,
    recipient: str,
    body: str,
    subject: str = "smtpBERT mock email test",
) -> None:
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = recipient
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP(smtp_host, smtp_port, timeout=20) as smtp:
        smtp.send_message(msg)


def mock_gateway_test(
    smtp_host: str,
    smtp_port: int,
    sender: str,
    recipient: str,
    body: str,
    auto_start_gateway: bool,
) -> None:
    gateway_proc = None
    try:
        if auto_start_gateway:
            gateway_proc = _start_gateway_process(smtp_port=smtp_port)

        send_mock_email(
            smtp_host=smtp_host,
            smtp_port=smtp_port,
            sender=sender,
            recipient=recipient,
            body=body,
        )
        print("Mock email sent successfully.")
    finally:
        if gateway_proc is not None:
            gateway_proc.terminate()


def run_local_all_in_one(
    epochs: int,
    batch_size: int,
    learning_rate: float,
    smtp_host: str,
    smtp_port: int,
    sender: str,
    recipient: str,
    body: str,
    skip_baseline: bool,
) -> None:
    """Train locally, run a quick model test, and verify SMTP gateway with a mock email."""
    if not skip_baseline:
        print("\n[1/4] Training baseline models...")
        train_baseline()

    print("\n[2/4] Training DistilBERT...")
    train_distilbert(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)

    print("\n[3/4] Running DistilBERT sample predictions...")
    test_distilbert()

    print("\n[4/4] Starting gateway and sending a mock email...")
    mock_gateway_test(
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        sender=sender,
        recipient=recipient,
        body=body,
        auto_start_gateway=True,
    )
    print("\nAll-in-one local flow completed successfully.")


def _prompt_int(prompt: str, default: Optional[int]) -> Optional[int]:
    raw = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if not raw:
        return default
    return int(raw)


def _prompt_float(prompt: str, default: Optional[float]) -> Optional[float]:
    raw = input(f"{prompt}" + (f" [{default}]" if default is not None else "") + ": ").strip()
    if not raw:
        return default
    return float(raw)


def _prompt_text(prompt: str, default: str) -> str:
    raw = input(f"{prompt} [{default}]: ").strip()
    return raw or default


def run_menu() -> None:
    while True:
        print("""
=== smtpBERT Local CLI ===

[ Training ]
  1) Train Baseline Models       (LR / RF / XGBoost)
  2) Train DistilBERT             (fine-tune on URL data)
  3) Test DistilBERT              (sample predictions)

[ Evaluation ]
  4) Run 10-Fold CV              (all models: DistilBERT + baselines)
  5) Benchmark Inference         (latency / memory / ROC curves)
  6) Compute Baseline ROC-AUC    (saved baseline models, no retrain)
  7) Compute Model Sizes          (param counts / file sizes)
  8) Evaluate DistilBERT          (accuracy + metrics on test set)
  9) Gather All Metrics           (runs 10-fold + benchmark + shap + sizes)

[ Deployment ]
  10) Run SMTP Gateway            (start phishing detection server)
  11) Send Mock Email             (test gateway with sample URL)
  12) Run All-in-One              (train + test + email in one shot)

[ Utilities ]
  13) Open Dashboard              (Streamlit web UI)
  14) XGBoost SHAP                (explainability plots, ~2-3 min)
  15) DistilBERT SHAP             (explainability plots, ~15-20 min)
  0) Exit
""")
        choice = input("Select option: ").strip()

        if choice == "1":
            train_baseline()
        elif choice == "2":
            print("\n--- Select DistilBERT Training Profile ---")
            print("  1) Fast Run    (1 Epoch, Batch 16, LR 5e-5) - Best for quick tests")
            print("  2) Standard    (3 Epochs, Batch 8, LR 2e-5) - Good balance")
            print("  3) Deep Train  (5 Epochs, Batch 4, LR 1e-5) - Maximum accuracy")
            print("  4) Custom      (Type values manually)")
            
            profile = input("\nSelect profile [1-4]: ").strip()
            
            if profile == "1":
                epochs, batch_size, learning_rate = 1, 16, 5e-5
            elif profile == "2":
                epochs, batch_size, learning_rate = 3, 8, 2e-5
            elif profile == "3":
                epochs, batch_size, learning_rate = 5, 4, 1e-5
            elif profile == "4":
                epochs = _prompt_int("Epochs", 1)
                batch_size = _prompt_int("Batch size", 8)
                learning_rate = _prompt_float("Learning rate", 2e-5)
            else:
                print("Invalid choice, defaulting to Standard.")
                epochs, batch_size, learning_rate = 3, 8, 2e-5
                
            print(f"\nStarting training with {epochs} Epochs, Batch Size {batch_size}, LR {learning_rate}...")
            train_distilbert(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
        elif choice == "3":
            test_distilbert()
        elif choice == "4":
            run_ten_fold_comparison()
        elif choice == "5":
            from src.models.compute_inference_metrics import main as benchmark_main
            benchmark_main()
        elif choice == "6":
            from src.models.compute_baseline_roc_auc import main as compute_roc_auc_main
            compute_roc_auc_main()
        elif choice == "7":
            from src.models.compute_model_sizes import main as model_sizes_main
            model_sizes_main()
        elif choice == "8":
            retrain = _prompt_text("Retrain DistilBERT before evaluation? (y/n)", "n").lower().startswith("y")
            epochs = _prompt_int("Epochs", 1) if retrain else None
            batch_size = _prompt_int("Batch size", 4) if retrain else None
            learning_rate = _prompt_float("Learning rate", 2e-5) if retrain else None
            evaluate_distilbert(
                retrain=retrain,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )
        elif choice == "9":
            retrain_distilbert = _prompt_text(
                "Retrain DistilBERT before gathering metrics? (y/n)",
                "n",
            ).lower().startswith("y")
            run_baseline_cv = _prompt_text(
                "Run 10-fold cross-validation now? (y/n)",
                "n",
            ).lower().startswith("y")
            run_shap = _prompt_text(
                "Generate SHAP plots? (y/n)",
                "n",
            ).lower().startswith("y")
            if run_shap:
                run_distilbert_shap = _prompt_text(
                    "  Also DistilBERT SHAP? (slow, ~15-20 min) (y/n)",
                    "n",
                ).lower().startswith("y")
            else:
                run_distilbert_shap = False
            epochs = _prompt_int("DistilBERT epochs", 1) if retrain_distilbert else None
            batch_size = _prompt_int("DistilBERT batch size", 4) if retrain_distilbert else None
            learning_rate = _prompt_float("DistilBERT learning rate", 2e-5) if retrain_distilbert else None

            print("\n--- Gathering all metrics ---")
            print("\n[Step 1/7] Refreshing holdout metrics...")
            gather_dashboard_metrics(
                retrain_distilbert=retrain_distilbert,
                run_baseline_cv=run_baseline_cv,
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
            )

            print("\n[Step 2/7] Benchmarking inference (latency / memory / ROC curves)...")
            from src.models.compute_inference_metrics import main as benchmark_main
            benchmark_main()

            print("\n[Step 3/7] Computing baseline ROC-AUC...")
            from src.models.compute_baseline_roc_auc import main as compute_roc_auc_main
            compute_roc_auc_main()

            print("\n[Step 4/7] Computing model sizes...")
            from src.models.compute_model_sizes import main as model_sizes_main
            model_sizes_main()

            if run_shap:
                print("\n[Step 5/7] Computing XGBoost SHAP plots...")
                from src.models.compute_xgboost_shap import main as xgb_shap_main
                xgb_shap_main()

                if run_distilbert_shap:
                    print("\n[Step 6/7] Computing DistilBERT SHAP plots (slow, ~15-20 min)...")
                    from src.models.compute_distilbert_shap import main as distilbert_shap_main
                    distilbert_shap_main()
                else:
                    print("\n[Step 6/7] Skipping DistilBERT SHAP (use option 15 to run later)")

                print("\n[Step 7/7] Done! All metrics gathered.")
                print("  Run 'Open Dashboard' to view all results in the web UI.")
            else:
                print("\n[Steps 5-7] SHAP skipped (use options 14/15 to generate later)")
                print("\n[DONE] All metrics gathered.")
                print("  Run 'Open Dashboard' to view all results in the web UI.")
        elif choice == "10":
            smtp_port = _prompt_int("SMTP port", None)
            run_gateway(smtp_port=smtp_port)
        elif choice == "11":
            smtp_host = _prompt_text("SMTP host", "127.0.0.1")
            smtp_port = _prompt_int("SMTP port", 25)
            sender = _prompt_text("Sender", "sender@example.local")
            recipient = _prompt_text("Recipient", "recipient@example.local")
            body = _prompt_text("Email body", "Hello from smtpBERT test. URL: https://example.com/login")
            auto_start = _prompt_text("Auto-start gateway? (y/n)", "n").lower().startswith("y")
            mock_gateway_test(
                smtp_host=smtp_host,
                smtp_port=int(smtp_port) if smtp_port is not None else 25,
                sender=sender,
                recipient=recipient,
                body=body,
                auto_start_gateway=auto_start,
            )
        elif choice == "12":
            skip_baseline = _prompt_text("Skip baseline? (y/n)", "n").lower().startswith("y")
            epochs = _prompt_int("Epochs", 1)
            batch_size = _prompt_int("Batch size", 4)
            learning_rate = _prompt_float("Learning rate", 2e-5)
            smtp_host = _prompt_text("SMTP host", "127.0.0.1")
            smtp_port = _prompt_int("SMTP port", 2525)
            sender = _prompt_text("Sender", "sender@example.local")
            recipient = _prompt_text("Recipient", "recipient@example.local")
            body = _prompt_text("Email body", "Local SMTP test from smtpBERT. URL: https://example.com/login")
            run_local_all_in_one(
                epochs=int(epochs) if epochs is not None else 1,
                batch_size=int(batch_size) if batch_size is not None else 4,
                learning_rate=float(learning_rate) if learning_rate is not None else 2e-5,
                smtp_host=smtp_host,
                smtp_port=int(smtp_port) if smtp_port is not None else 2525,
                sender=sender,
                recipient=recipient,
                body=body,
                skip_baseline=skip_baseline,
            )
        elif choice == "13":
            run_dashboard()
        elif choice == "14":
            from src.models.compute_xgboost_shap import main as xgb_shap_main
            xgb_shap_main()
        elif choice == "15":
            from src.models.compute_distilbert_shap import main as distilbert_shap_main
            distilbert_shap_main()
        elif choice == "0":
            print("Exiting smtpBERT local CLI.")
            return
        else:
            print("Invalid option. Please select 0-13.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="smtpBERT local CLI")
    subparsers = parser.add_subparsers(dest="command", required=False)

    subparsers.add_parser("train-baseline", help="Train baseline models (LR / RF / XGBoost)")

    train_distilbert_parser = subparsers.add_parser("train-distilbert", help="Fine-tune DistilBERT on URL data")
    train_distilbert_parser.add_argument("--epochs", type=int, default=None)
    train_distilbert_parser.add_argument("--batch-size", type=int, default=None)
    train_distilbert_parser.add_argument("--learning-rate", type=float, default=None)

    subparsers.add_parser("test-distilbert", help="Run sample DistilBERT predictions (no retrain)")

    eval_distilbert_parser = subparsers.add_parser(
        "evaluate-distilbert",
        help="Evaluate DistilBERT on test set (optional retrain)",
    )
    eval_distilbert_parser.add_argument("--retrain", action="store_true")
    eval_distilbert_parser.add_argument("--epochs", type=int, default=None)
    eval_distilbert_parser.add_argument("--batch-size", type=int, default=None)
    eval_distilbert_parser.add_argument("--learning-rate", type=float, default=None)

    gather_metrics_parser = subparsers.add_parser(
        "gather-metrics",
        help="Gather all dashboard metrics (optional retrain + 10-fold CV)",
    )
    gather_metrics_parser.add_argument("--retrain-distilbert", action="store_true")
    gather_metrics_parser.add_argument("--run-baseline-cv", action="store_true")
    gather_metrics_parser.add_argument("--epochs", type=int, default=None)
    gather_metrics_parser.add_argument("--batch-size", type=int, default=None)
    gather_metrics_parser.add_argument("--learning-rate", type=float, default=None)

    run_gateway_parser = subparsers.add_parser("run-gateway", help="Start SMTP phishing detection gateway")
    run_gateway_parser.add_argument("--smtp-port", type=int, default=None)
    subparsers.add_parser("run-dashboard", help="Open Streamlit dashboard")

    subparsers.add_parser(
        "ten-fold-comparison",
        help="Run 10-fold CV for all models (DistilBERT + LR + RF + XGBoost)",
    )
    subparsers.add_parser(
        "benchmark",
        help="Benchmark inference latency, memory, and ROC curves (precomputes all metrics)",
    )
    subparsers.add_parser(
        "baseline-roc-auc",
        help="Compute ROC-AUC for saved baseline models (no retrain)",
    )
    subparsers.add_parser(
        "model-sizes",
        help="Compute model size metrics (param counts / file sizes)",
    )
    subparsers.add_parser("xgboost-shap", help="Generate XGBoost SHAP explainability plots (~2-3 min)")
    subparsers.add_parser("distilbert-shap", help="Generate DistilBERT SHAP plots (~15-20 min)")

    mock_parser = subparsers.add_parser("mock-email", help="Send a mock email to SMTP gateway")
    mock_parser.add_argument("--smtp-host", default="127.0.0.1")
    mock_parser.add_argument("--smtp-port", type=int, default=25)
    mock_parser.add_argument("--sender", default="sender@example.local")
    mock_parser.add_argument("--recipient", default="recipient@example.local")
    mock_parser.add_argument(
        "--body",
        default="Hello from smtpBERT test. URL: https://example.com/login",
    )
    mock_parser.add_argument(
        "--auto-start-gateway",
        action="store_true",
        help="Start gateway in background for this test, then stop it.",
    )

    run_all_parser = subparsers.add_parser(
        "run-local-all",
        help="Train + test + run mock SMTP email in one command",
    )
    run_all_parser.add_argument("--epochs", type=int, default=1)
    run_all_parser.add_argument("--batch-size", type=int, default=4)
    run_all_parser.add_argument("--learning-rate", type=float, default=2e-5)
    run_all_parser.add_argument("--skip-baseline", action="store_true")
    run_all_parser.add_argument("--smtp-host", default="127.0.0.1")
    run_all_parser.add_argument("--smtp-port", type=int, default=2525)
    run_all_parser.add_argument("--sender", default="sender@example.local")
    run_all_parser.add_argument("--recipient", default="recipient@example.local")
    run_all_parser.add_argument(
        "--body",
        default="Local SMTP test from smtpBERT. URL: https://example.com/login",
    )
    subparsers.add_parser("menu", help="Open interactive CLI menu")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None or args.command == "menu":
        run_menu()
    elif args.command == "train-baseline":
        train_baseline()
    elif args.command == "train-distilbert":
        train_distilbert(args.epochs, args.batch_size, args.learning_rate)
    elif args.command == "test-distilbert":
        test_distilbert()
    elif args.command == "evaluate-distilbert":
        evaluate_distilbert(
            retrain=args.retrain,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    elif args.command == "gather-metrics":
        gather_dashboard_metrics(
            retrain_distilbert=args.retrain_distilbert,
            run_baseline_cv=args.run_baseline_cv,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )
    elif args.command == "run-gateway":
        run_gateway(smtp_port=args.smtp_port)
    elif args.command == "run-dashboard":
        run_dashboard()
    elif args.command == "ten-fold-comparison":
        run_ten_fold_comparison()
    elif args.command == "benchmark":
        from src.models.compute_inference_metrics import main as benchmark_main
        benchmark_main()
    elif args.command == "baseline-roc-auc":
        from src.models.compute_baseline_roc_auc import main as compute_roc_auc_main
        compute_roc_auc_main()
    elif args.command == "model-sizes":
        from src.models.compute_model_sizes import main as model_sizes_main
        model_sizes_main()
    elif args.command == "xgboost-shap":
        from src.models.compute_xgboost_shap import main as xgb_shap_main
        xgb_shap_main()
    elif args.command == "distilbert-shap":
        from src.models.compute_distilbert_shap import main as distilbert_shap_main
        distilbert_shap_main()
    elif args.command == "mock-email":
        mock_gateway_test(
            smtp_host=args.smtp_host,
            smtp_port=args.smtp_port,
            sender=args.sender,
            recipient=args.recipient,
            body=args.body,
            auto_start_gateway=args.auto_start_gateway,
        )
    elif args.command == "run-local-all":
        run_local_all_in_one(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            smtp_host=args.smtp_host,
            smtp_port=args.smtp_port,
            sender=args.sender,
            recipient=args.recipient,
            body=args.body,
            skip_baseline=args.skip_baseline,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
