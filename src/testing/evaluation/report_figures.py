"""Generate evaluation figures for the dissertation report.

Usage:
    python -m testing.evaluation.report_figures
    python -m testing.evaluation.report_figures --output path/to/figures
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import EVAL_RESULTS_PATH

FIGURES_DIR = Path(__file__).parent.parent.parent.parent / "figures"


# Data loading

def load_primary_results() -> dict:
    """Load primary model (RoBERTa zero-shot) evaluation results."""
    path = EVAL_RESULTS_PATH / "evaluation_results_zero_shot.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Per-Horizon Line Chart

def fig_per_horizon(output_dir: Path) -> None:
    """Line chart of RoBERTa performance across prediction windows."""
    data = load_primary_results()
    windows = [1, 3, 5, 7, 14, 31]
    window_labels = ["1", "3", "5", "7", "14", "31"]

    acc_by_w = [data["per_window_metrics"][f"W={w}"]["accuracy"] for w in windows]
    f1_by_w = [data["per_window_metrics"][f"W={w}"]["macro_f1"] for w in windows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(window_labels, acc_by_w, "o-", color="#4A90D9", linewidth=2, markersize=8, label="Accuracy")
    ax.plot(window_labels, f1_by_w, "s--", color="#5BA85B", linewidth=2, markersize=8, label="Macro F1")

    peak_idx = acc_by_w.index(max(acc_by_w))
    ax.annotate(
        f"{acc_by_w[peak_idx]:.1%}", xy=(window_labels[peak_idx], acc_by_w[peak_idx]),
        xytext=(0, 12), textcoords="offset points", ha="center",
        fontsize=9, fontweight="bold", color="#4A90D9",
    )

    ax.set_xlabel("Prediction Window (days)", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0.15, 0.5)
    ax.axhline(y=0.484, color="#333333", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(5, 0.488, "Majority baseline", fontsize=8, color="#333333")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "per_horizon.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: per_horizon.png")


# Confusion Matrix Heatmap

def fig_confusion_matrix(output_dir: Path) -> None:
    """RoBERTa confusion matrix heatmap."""
    data = load_primary_results()
    cm = data["overall_metrics"]["confusion_matrix"]
    labels = ["negative", "neutral", "positive"]
    matrix = np.array([[cm[actual][pred] for pred in labels] for actual in labels])

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")

    for i in range(3):
        for j in range(3):
            color = "white" if matrix[i, j] > matrix.max() * 0.6 else "black"
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                    fontsize=14, fontweight="bold", color=color)

    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(["Negative", "Neutral", "Positive"], fontsize=10)
    ax.set_yticklabels(["Negative", "Neutral", "Positive"], fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("Actual Label", fontsize=11)
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: confusion_matrix.png")

# Confusion Matrices for All Models (Appendix)

def fig_all_model_confusion_matrices(output_dir: Path) -> None:
    """Confusion matrix heatmaps for all non-primary models (appendix)."""
    models = {
        "FinBERT": "evaluation_results_finbert.json",
        "FinGPT": "evaluation_results_fingpt.json",
        "Llama 3.1 8B": "evaluation_results_llama.json",
        "Mistral 7B": "evaluation_results_mistral.json",
    }
    labels = ["negative", "neutral", "positive"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()

    for idx, (model_name, filename) in enumerate(models.items()):
        path = EVAL_RESULTS_PATH / filename
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        cm = data["overall_metrics"]["confusion_matrix"]
        matrix = np.array([[cm[actual][pred] for pred in labels] for actual in labels])

        ax = axes[idx]
        im = ax.imshow(matrix, cmap="Blues", aspect="auto")

        for i in range(3):
            for j in range(3):
                color = "white" if matrix[i, j] > matrix.max() * 0.6 else "black"
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center",
                        fontsize=13, fontweight="bold", color=color)

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["Neg", "Neu", "Pos"], fontsize=9)
        ax.set_yticklabels(["Neg", "Neu", "Pos"], fontsize=9)
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)
        ax.set_title(model_name, fontsize=11, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "all_model_confusion_matrices.png", dpi=200,
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: all_model_confusion_matrices.png")



# Gaussian Horizon Weighting Curves

def fig_gaussian_horizon(output_dir: Path) -> None:
    """Gaussian horizon weight curves for different prediction windows."""
    fig, ax = plt.subplots(figsize=(9, 5))

    windows = [1, 3, 5, 7, 14, 31]
    colors_w = ["#D94A4A", "#D9A84A", "#B07BD9", "#4A90D9", "#4AD9A8", "#5BA85B"]
    impact_days = np.linspace(-10, 40, 300)

    for W, color in zip(windows, colors_w):
        mu = W / 2
        sigma = max(W / 2, 3.0)
        weights = np.exp(-((impact_days - mu) ** 2) / (2 * sigma ** 2))
        weights = np.maximum(weights, 0.05)
        ax.plot(impact_days, weights, color=color, linewidth=2,
                label=f"W={W} days")

    ax.set_xlabel("Days Until Expected Impact", fontsize=11)
    ax.set_ylabel("Horizon Weight", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_xlim(-10, 40)
    ax.axhline(y=0.05, color="#999999", linestyle=":", linewidth=1, alpha=0.7)
    ax.text(35, 0.07, "Floor (0.05)", fontsize=8, color="#999999")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "gaussian_horizon.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: gaussian_horizon.png")


# Recency Decay Curves

def fig_recency_decay(output_dir: Path) -> None:
    """Recency decay curves for different lambda values."""
    fig, ax = plt.subplots(figsize=(9, 5))

    anchor_windows = [1, 5, 10, 21]
    anchor_lambdas = [0.89, 0.92, 0.95, 0.97]

    def interpolate_lambda(W):
        """Linearly interpolate decay factor for a given prediction window."""
        if W <= anchor_windows[0]:
            return anchor_lambdas[0]
        if W >= anchor_windows[-1]:
            return anchor_lambdas[-1]
        for i in range(len(anchor_windows) - 1):
            if anchor_windows[i] <= W <= anchor_windows[i + 1]:
                t = (W - anchor_windows[i]) / (anchor_windows[i + 1] - anchor_windows[i])
                return anchor_lambdas[i] + t * (anchor_lambdas[i + 1] - anchor_lambdas[i])
        return anchor_lambdas[-1]

    windows = [1, 3, 5, 7, 14, 31]
    days = np.arange(0, 31)
    colors_r = ["#D94A4A", "#D9A84A", "#B07BD9", "#4A90D9", "#4AD9A8", "#5BA85B"]

    for W, color in zip(windows, colors_r):
        lam = interpolate_lambda(W)
        weights = lam ** days
        ax.plot(days, weights, color=color, linewidth=2, label=f"W={W} days (lambda={lam:.3f})")

    ax.set_xlabel("Article Age (trading days)", fontsize=11)
    ax.set_ylabel("Recency Weight", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "recency_decay.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: recency_decay.png")


# Evidence Quality vs Accuracy

def fig_quality_accuracy(output_dir: Path) -> None:
    """Bar chart showing prediction accuracy by evidence quality rating."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ratings = ["HIGH\n(0-1 flags)", "MEDIUM\n(2 flags)", "LOW\n(3+ flags)"]
    cases = [83, 90, 19]
    accuracies = [49.4, 28.9, 26.3]
    colors = ["#5BA85B", "#D9A84A", "#D94A4A"]

    bars = ax.bar(ratings, accuracies, color=colors, edgecolor="white", width=0.5)

    for bar, acc, n in zip(bars, accuracies, cases):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{acc:.1f}%\n({n} cases)", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.axhline(y=48.4, color="#333333", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(2.3, 49.5, "Majority baseline (48.4%)", fontsize=8, color="#333333", ha="right")

    ax.set_ylabel("Prediction Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 65)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "quality_accuracy.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: quality_accuracy.png")


# Main

def main() -> None:
    """Generate all report figures and save to the figures directory."""
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--output", type=str, default=str(FIGURES_DIR),
                        help="Output directory for figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures in: {output_dir}")
    print()

    fig_per_horizon(output_dir)
    fig_confusion_matrix(output_dir)
    fig_all_model_confusion_matrices(output_dir)
    fig_gaussian_horizon(output_dir)
    fig_recency_decay(output_dir)
    fig_quality_accuracy(output_dir)

    print()
    print("All figures generated.")


if __name__ == "__main__":
    main()
