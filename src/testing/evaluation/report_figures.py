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

def load_model_results() -> dict:
    """Load overall metrics for all five models."""
    models = ["zero_shot", "finbert", "fingpt", "llama", "mistral"]
    results = {}
    for m in models:
        path = EVAL_RESULTS_PATH / f"evaluation_results_{m}.json"
        with open(path, "r", encoding="utf-8") as f:
            results[m] = json.load(f)
    return results


def load_primary_results() -> dict:
    """Load primary model (RoBERTa zero-shot) evaluation results."""
    path = EVAL_RESULTS_PATH / "evaluation_results_zero_shot.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Figure 1: Model Comparison (Grouped Bar)

def fig_model_comparison(output_dir: Path) -> None:
    """Grouped bar chart comparing all five models."""
    data = load_model_results()

    model_labels = [
        "RoBERTa\nZero-Shot", "FinBERT", "FinGPT",
        "Llama 3.1\n8B", "Mistral\n7B",
    ]
    models = ["zero_shot", "finbert", "fingpt", "llama", "mistral"]

    accuracy, macro_f1, neg_f1, neu_f1, pos_f1 = [], [], [], [], []
    for m in models:
        om = data[m]["overall_metrics"]
        accuracy.append(om["accuracy"])
        macro_f1.append(om["macro_f1"])
        neg_f1.append(om["per_class"]["negative"]["f1"])
        neu_f1.append(om["per_class"]["neutral"]["f1"])
        pos_f1.append(om["per_class"]["positive"]["f1"])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(models))
    width = 0.15

    ax.bar(x - 2 * width, accuracy, width, label="Accuracy", color="#4A90D9", edgecolor="white")
    ax.bar(x - width, macro_f1, width, label="Macro F1", color="#5BA85B", edgecolor="white")
    ax.bar(x, neg_f1, width, label="Neg F1", color="#D94A4A", edgecolor="white")
    ax.bar(x + width, neu_f1, width, label="Neu F1", color="#B0B0B0", edgecolor="white")
    ax.bar(x + 2 * width, pos_f1, width, label="Pos F1", color="#D9A84A", edgecolor="white")

    ax.axhline(y=0.484, color="#333333", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(2.5, 0.49, "Majority baseline (48.4%)", fontsize=8, color="#333333", ha="left")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "model_comparison.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: model_comparison.png/.pdf")


# Figure 2: Score Distribution Overlap (Box Plots)

def fig_score_distributions(output_dir: Path) -> None:
    """Two-panel box plots showing positive and negative score overlap across actual classes."""
    data = load_primary_results()

    groups = {
        "negative": {"pos": [], "neg": []},
        "neutral": {"pos": [], "neg": []},
        "positive": {"pos": [], "neg": []},
    }
    for c in data["case_results"]:
        actual = c["actual_label"]
        groups[actual]["pos"].append(c["normalized_scores"]["positive"])
        groups[actual]["neg"].append(c["normalized_scores"]["negative"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    colors = ["#D94A4A", "#B0B0B0", "#4A90D9"]
    class_labels = [
        f"Actually\nNegative\n(n={len(groups['negative']['pos'])})",
        f"Actually\nNeutral\n(n={len(groups['neutral']['pos'])})",
        f"Actually\nPositive\n(n={len(groups['positive']['pos'])})",
    ]
    rng = np.random.default_rng(42)

    def _draw_box(ax, data_list, title, ylabel):
        bp = ax.boxplot(
            data_list, tick_labels=class_labels, patch_artist=True, widths=0.5,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(color="#555555"),
            capprops=dict(color="#555555"),
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        for i, vals in enumerate(data_list):
            jitter = rng.normal(0, 0.04, len(vals))
            ax.scatter(
                np.full(len(vals), i + 1) + jitter, vals,
                alpha=0.3, s=15, color=colors[i], zorder=3,
            )
        means = [np.mean(v) for v in data_list]
        ax.scatter([1, 2, 3], means, marker="D", color="black", s=40, zorder=5,
                   label=f"Means: {means[0]:.3f}, {means[1]:.3f}, {means[2]:.3f}")
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="upper right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3)

    pos_data = [groups["negative"]["pos"], groups["neutral"]["pos"], groups["positive"]["pos"]]
    neg_data = [groups["negative"]["neg"], groups["neutral"]["neg"], groups["positive"]["neg"]]

    _draw_box(ax1, pos_data, "Positive Sentiment Score", "Score")
    _draw_box(ax2, neg_data, "Negative Sentiment Score", "Score")

    fig.suptitle("RoBERTa Score Distributions by Actual Ground-Truth Class",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "score_distributions.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: score_distributions.png/.pdf")


# Figure 3: Flip-Set Distribution Histogram

def fig_flipset_distribution(output_dir: Path) -> None:
    """Histogram of flip-set sizes across holdout cases."""
    fig, ax = plt.subplots(figsize=(9, 5))

    bins_labels = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", ">10"]
    counts = [16, 12, 15, 7, 13, 4, 7, 11, 3, 5, 99]

    x = np.arange(len(bins_labels))
    colors = ["#D94A4A"] + ["#4A90D9"] * 9 + ["#5BA85B"]

    bars = ax.bar(x, counts, color=colors, edgecolor="white", width=0.7)

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            str(count), ha="center", va="bottom", fontsize=9, fontweight="bold",
        )


    ax.set_xlabel("Minimum Articles to Remove (Flip-Set Size)", fontsize=11)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(bins_labels, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "flipset_distribution.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "flipset_distribution.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: flipset_distribution.png/.pdf")


# Figure 4: Per-Horizon Line Chart

def fig_per_horizon(output_dir: Path) -> None:
    """Line chart of RoBERTa performance across prediction windows (Section 6.5.1)."""
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
    plt.savefig(output_dir / "per_horizon.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: per_horizon.png/.pdf")


# Figure 5: Confusion Matrix Heatmap

def fig_confusion_matrix(output_dir: Path) -> None:
    """RoBERTa confusion matrix heatmap (Section 6.5.3)."""
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
    plt.savefig(output_dir / "confusion_matrix.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: confusion_matrix.png/.pdf")


# Figure 6: Gaussian Horizon Weighting Curves

def fig_gaussian_horizon(output_dir: Path) -> None:
    """Gaussian horizon weight curves for different prediction windows (Section 4.5.3)."""
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
    plt.savefig(output_dir / "gaussian_horizon.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: gaussian_horizon.png/.pdf")


# Figure 7: Recency Decay Curves

def fig_recency_decay(output_dir: Path) -> None:
    """Recency decay curves for different lambda values (Section 4.5.1)."""
    fig, ax = plt.subplots(figsize=(9, 5))

    # Anchor points from Table 8
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
    plt.savefig(output_dir / "recency_decay.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: recency_decay.png/.pdf")


# Figure 8: Score Distribution by True Class (Simple Bar Chart)

def fig_score_by_class(output_dir: Path) -> None:
    """Grouped bar chart showing mean positive and negative scores by actual class."""
    data = load_primary_results()

    groups = {"negative": {"pos": [], "neg": []}, "neutral": {"pos": [], "neg": []}, "positive": {"pos": [], "neg": []}}
    for c in data["case_results"]:
        actual = c["actual_label"]
        groups[actual]["pos"].append(c["normalized_scores"]["positive"])
        groups[actual]["neg"].append(c["normalized_scores"]["negative"])

    fig, ax = plt.subplots(figsize=(8, 5))

    classes = [
        f"Actually\nNegative\n(n={len(groups['negative']['pos'])})",
        f"Actually\nNeutral\n(n={len(groups['neutral']['pos'])})",
        f"Actually\nPositive\n(n={len(groups['positive']['pos'])})",
    ]
    x = np.arange(len(classes))
    width = 0.3

    pos_means = [np.mean(groups["negative"]["pos"]), np.mean(groups["neutral"]["pos"]), np.mean(groups["positive"]["pos"])]
    neg_means = [np.mean(groups["negative"]["neg"]), np.mean(groups["neutral"]["neg"]), np.mean(groups["positive"]["neg"])]

    bars1 = ax.bar(x - width / 2, pos_means, width, label="Avg Positive Score", color="#4A90D9", edgecolor="white")
    bars2 = ax.bar(x + width / 2, neg_means, width, label="Avg Negative Score", color="#D94A4A", edgecolor="white")

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#4A90D9")
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold", color="#D94A4A")

    ax.set_ylabel("Mean Sentiment Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=10)
    ax.set_ylim(0, 0.55)
    ax.legend(fontsize=10, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "score_by_class.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "score_by_class.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: score_by_class.png/.pdf")


# Figure 9: Evidence Quality vs Accuracy

def fig_quality_accuracy(output_dir: Path) -> None:
    """Bar chart showing prediction accuracy by evidence quality rating."""
    fig, ax = plt.subplots(figsize=(7, 5))

    ratings = ["HIGH\n(0-1 flags)", "MEDIUM\n(2 flags)", "LOW\n(3+ flags)"]
    cases = [108, 75, 9]
    accuracies = [49.1, 25.3, 0.0]
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
    plt.savefig(output_dir / "quality_accuracy.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print("  Saved: quality_accuracy.png/.pdf")


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

    fig_model_comparison(output_dir)
    fig_per_horizon(output_dir)
    fig_confusion_matrix(output_dir)
    fig_score_distributions(output_dir)
    fig_flipset_distribution(output_dir)
    fig_gaussian_horizon(output_dir)
    fig_recency_decay(output_dir)
    fig_score_by_class(output_dir)
    fig_quality_accuracy(output_dir)

    print()
    print("All figures generated.")


if __name__ == "__main__":
    main()
