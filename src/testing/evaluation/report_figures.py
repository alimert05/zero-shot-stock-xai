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
import matplotlib.patches as mpatches
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import EVAL_RESULTS_PATH

FIGURES_DIR = Path(__file__).parent.parent.parent.parent / "figures"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_model_results() -> dict:
    """Load overall metrics for all five models."""
    models = ["deberta", "finbert", "fingpt", "llama", "mistral"]
    results = {}
    for m in models:
        path = EVAL_RESULTS_PATH / f"evaluation_results_{m}.json"
        with open(path, "r", encoding="utf-8") as f:
            results[m] = json.load(f)
    return results


def load_deberta_results() -> dict:
    """Load DeBERTa evaluation results."""
    path = EVAL_RESULTS_PATH / "evaluation_results_deberta.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1: Model Comparison (Grouped Bar)
# ---------------------------------------------------------------------------

def fig_model_comparison(output_dir: Path) -> None:
    """Grouped bar chart comparing all five models."""
    data = load_model_results()

    model_labels = [
        "DeBERTa\nZero-Shot", "FinBERT", "FinGPT",
        "Llama 3.1\n8B", "Mistral\n7B",
    ]
    models = ["deberta", "finbert", "fingpt", "llama", "mistral"]

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
    ax.text(4.5, 0.49, "Majority baseline (48.4%)", fontsize=8, color="#333333", ha="right")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9)
    ax.set_ylim(0, 0.6)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_title("Model Comparison on Holdout Set (192 cases)", fontsize=12, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "model_comparison.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "model_comparison.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: model_comparison.png/.pdf")


# ---------------------------------------------------------------------------
# Figure 2: Score Distribution Overlap (Box Plots)
# ---------------------------------------------------------------------------

def fig_score_distributions(output_dir: Path) -> None:
    """Box plots showing positive score overlap across actual classes."""
    data = load_deberta_results()

    groups = {"negative": [], "neutral": [], "positive": []}
    for c in data["case_results"]:
        groups[c["actual_label"]].append(c["normalized_scores"]["positive"])

    fig, ax = plt.subplots(figsize=(8, 5.5))

    box_data = [groups["negative"], groups["neutral"], groups["positive"]]
    box_labels = [
        f"Actually Negative\n(n={len(groups['negative'])})",
        f"Actually Neutral\n(n={len(groups['neutral'])})",
        f"Actually Positive\n(n={len(groups['positive'])})",
    ]
    colors = ["#D94A4A", "#B0B0B0", "#4A90D9"]

    bp = ax.boxplot(
        box_data, tick_labels=box_labels, patch_artist=True, widths=0.5,
        medianprops=dict(color="black", linewidth=2),
        whiskerprops=dict(color="#555555"),
        capprops=dict(color="#555555"),
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    for i, vals in enumerate(box_data):
        jitter = np.random.default_rng(42).normal(0, 0.04, len(vals))
        ax.scatter(
            np.full(len(vals), i + 1) + jitter, vals,
            alpha=0.3, s=15, color=colors[i], zorder=3,
        )

    means = [np.mean(v) for v in box_data]
    ax.scatter([1, 2, 3], means, marker="D", color="black", s=40, zorder=5, label="Mean")

    ax.set_ylabel("Positive Sentiment Score", fontsize=11)
    ax.set_title(
        "DeBERTa Positive Score Distribution by Actual Class",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "score_distributions.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "score_distributions.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: score_distributions.png/.pdf")


# ---------------------------------------------------------------------------
# Figure 3: Flip-Set Distribution Histogram
# ---------------------------------------------------------------------------

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

    ax.annotate(
        "8.3% of predictions\nflip with 1 removal",
        xy=(0, 16), xytext=(2, 70),
        fontsize=8, color="#D94A4A", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#D94A4A", lw=1.5),
    )
    ax.annotate(
        "51.6% require\n>10 removals",
        xy=(10, 99), xytext=(7.5, 85),
        fontsize=8, color="#5BA85B", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#5BA85B", lw=1.5),
    )

    ax.set_xlabel("Minimum Articles to Remove (Flip-Set Size)", fontsize=11)
    ax.set_ylabel("Number of Cases", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(bins_labels, fontsize=9)
    ax.set_title(
        "Flip-Set Size Distribution Across 192 Holdout Cases",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "flipset_distribution.png", dpi=200, bbox_inches="tight", facecolor="white")
    plt.savefig(output_dir / "flipset_distribution.pdf", bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved: flipset_distribution.png/.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate report figures")
    parser.add_argument("--output", type=str, default=str(FIGURES_DIR),
                        help="Output directory for figures")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating figures in: {output_dir}")
    print()

    fig_model_comparison(output_dir)
    fig_score_distributions(output_dir)
    fig_flipset_distribution(output_dir)

    print()
    print("All figures generated.")


if __name__ == "__main__":
    main()
