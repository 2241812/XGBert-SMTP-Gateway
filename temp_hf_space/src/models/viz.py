import os
import matplotlib.pyplot as plt
from .config import PLOT_DIR

def generate_visualizations(metrics, log_file=None):
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(2, 2, 1)
    categories = ["Training\nLoss", "Validation\nAccuracy", "Validation\nLoss"]
    values = [
        metrics.get("train_loss", 0),
        metrics.get("eval_accuracy", 0),
        metrics.get("eval_loss", 0),
    ]
    colors = ["#3498db", "#2ecc71", "#e74c3c"]
    bars = ax1.bar(categories, values, color=colors, edgecolor="black", linewidth=1.2)
    ax1.set_title("Training Results Summary", fontsize=14, fontweight="bold")
    ax1.set_ylabel("Value")
    for bar, val in zip(bars, values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    ax2 = fig.add_subplot(2, 2, 2)
    train_samples = metrics.get("train_samples", 0)
    test_samples = metrics.get("test_samples", 0)
    labels_pie = ["Training", "Test"]
    sizes = [train_samples, test_samples]
    colors_pie = ["#3498db", "#2ecc71"]
    ax2.pie(
        sizes,
        labels=labels_pie,
        autopct="%1.1f%%",
        colors=colors_pie,
        startangle=90,
        explode=(0.05, 0),
        shadow=True,
    )
    ax2.set_title("Dataset Split", fontsize=14, fontweight="bold")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.axis("off")
    params_text = f"""
    Model Configuration
    ═══════════════════════════

    Model: {metrics.get("model_name", "N/A").split("/")[-1]}

    Training:
    ├─ Epochs: {metrics.get("num_epochs", "N/A")}
    ├─ Batch Size: {metrics.get("batch_size", "N/A")}
    ├─ Learning Rate: {metrics.get("learning_rate", "N/A")}
    └─ Weight Decay: {metrics.get("weight_decay", "0.01")}

    Dataset:
    ├─ Training Samples: {train_samples:,}
    ├─ Test Samples: {test_samples:,}
    └─ Total: {train_samples + test_samples:,}
    """
    ax3.text(
        0.1,
        0.9,
        params_text,
        transform=ax3.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#ecf0f1", alpha=0.8),
    )

    ax4 = fig.add_subplot(2, 2, 4)
    ax4.axis("off")
    results_text = f"""
    Performance Metrics
    ═══════════════════════════

    ✓ Final Training Loss: {metrics.get("train_loss", 0):.4f}
    ✓ Validation Accuracy: {metrics.get("eval_accuracy", 0) * 100:.2f}%
    ✓ Validation Loss: {metrics.get("eval_loss", 0):.4f}

    Status: {"SUCCESS" if metrics.get("eval_accuracy", 0) > 0.5 else "NEEDS IMPROVEMENT"}
    """
    ax4.text(
        0.1,
        0.9,
        results_text,
        transform=ax4.transAxes,
        fontsize=11,
        verticalalignment="top",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="#d5f5e3", alpha=0.8),
    )

    plt.suptitle("smtpBERT Training Dashboard", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()

    output_path = os.path.join(PLOT_DIR, "training_summary.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Summary plot saved to: {output_path}")

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].bar(
        ["Train Loss", "Val Loss"],
        [metrics.get("train_loss", 0), metrics.get("eval_loss", 0)],
        color=["#3498db", "#e74c3c"],
        edgecolor="black",
    )
    axes[0].set_title("Loss Comparison", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Loss")
    for i, v in enumerate([metrics.get("train_loss", 0), metrics.get("eval_loss", 0)]):
        axes[0].text(i, v + 0.02, f"{v:.4f}", ha="center", fontweight="bold")

    axes[1].bar(
        ["Accuracy"],
        [metrics.get("eval_accuracy", 0) * 100],
        color=["#2ecc71"],
        edgecolor="black",
        width=0.5,
    )
    axes[1].set_title("Validation Accuracy", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].set_ylim(0, 100)
    axes[1].text(
        0,
        metrics.get("eval_accuracy", 0) * 100 + 2,
        f"{metrics.get('eval_accuracy', 0) * 100:.2f}%",
        ha="center",
        fontweight="bold",
    )

    output_path2 = os.path.join(PLOT_DIR, "training_metrics.png")
    plt.savefig(output_path2, dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Metrics plot saved to: {output_path2}")

    plt.close("all")
    print("[PLOT] Visualizations complete!")