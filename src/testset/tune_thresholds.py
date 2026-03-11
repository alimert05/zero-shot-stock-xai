"""Grid search for optimal decision thresholds on tune set scores.

Reads per-case normalised scores from a saved evaluation_results.json
and finds the (tau_pos, tau_neg) thresholds that maximise macro F1.

Decision logic:
    if normalized_scores["positive"] >= tau_pos  -> positive
    elif normalized_scores["negative"] >= tau_neg -> negative
    else                                          -> neutral

This operates purely on saved scores -- no GPU inference needed.

Usage:
    python -m testset.tune_thresholds
    python -m testset.tune_thresholds --results path/to/evaluation_results.json
    python -m testset.tune_thresholds --step 0.005   # finer grid
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PRED_PATH

DEFAULT_RESULTS = PRED_PATH / "evaluation_results_debias_multi_label.json"


# ── Metrics ─────────────────────────────────────────────────────────────

def _compute_metrics(
    actual: list[str],
    predicted: list[str],
) -> dict:
    """Compute accuracy, macro precision/recall/F1, and per-class metrics."""
    labels = ["negative", "neutral", "positive"]

    # Per-class TP, FP, FN
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    correct = 0
    for a, p in zip(actual, predicted):
        if a == p:
            correct += 1
            tp[a] += 1
        else:
            fp[p] += 1
            fn[a] += 1

    accuracy = correct / len(actual) if actual else 0.0

    per_class = {}
    precisions, recalls, f1s = [], [], []

    for lbl in labels:
        prec = tp[lbl] / (tp[lbl] + fp[lbl]) if (tp[lbl] + fp[lbl]) > 0 else 0.0
        rec = tp[lbl] / (tp[lbl] + fn[lbl]) if (tp[lbl] + fn[lbl]) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

        per_class[lbl] = {"precision": prec, "recall": rec, "f1": f1}
        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    return {
        "accuracy": accuracy,
        "macro_precision": sum(precisions) / len(precisions),
        "macro_recall": sum(recalls) / len(recalls),
        "macro_f1": sum(f1s) / len(f1s),
        "per_class": per_class,
        "correct": correct,
        "total": len(actual),
    }


def _apply_thresholds(
    scores: dict[str, float],
    tau_pos: float,
    tau_neg: float,
) -> str:
    """Apply decision thresholds to normalised scores."""
    if scores["positive"] >= tau_pos:
        return "positive"
    elif scores["negative"] >= tau_neg:
        return "negative"
    else:
        return "neutral"


# ── Grid search ─────────────────────────────────────────────────────────

def grid_search(
    case_results: list[dict],
    tau_pos_range: tuple[float, float, float] = (0.33, 0.70, 0.01),
    tau_neg_range: tuple[float, float, float] = (0.20, 0.60, 0.01),
) -> dict:
    """Exhaustive grid search over (tau_pos, tau_neg).

    Parameters
    ----------
    case_results : list[dict]
        Each dict must have 'normalized_scores' and 'actual_label'.
    tau_pos_range : (start, stop, step)
    tau_neg_range : (start, stop, step)

    Returns
    -------
    dict with best thresholds, metrics, and search metadata.
    """
    actual_labels = [c["actual_label"] for c in case_results]
    all_scores = [c["normalized_scores"] for c in case_results]

    best_f1 = -1.0
    best_tau_pos = 0.0
    best_tau_neg = 0.0
    best_metrics = {}
    candidates_checked = 0

    # Build grid
    pos_start, pos_stop, pos_step = tau_pos_range
    neg_start, neg_stop, neg_step = tau_neg_range

    tau_pos = pos_start
    while tau_pos <= pos_stop + 1e-9:
        tau_neg = neg_start
        while tau_neg <= neg_stop + 1e-9:
            predicted = [
                _apply_thresholds(s, tau_pos, tau_neg) for s in all_scores
            ]
            metrics = _compute_metrics(actual_labels, predicted)
            candidates_checked += 1

            if metrics["macro_f1"] > best_f1:
                best_f1 = metrics["macro_f1"]
                best_tau_pos = round(tau_pos, 4)
                best_tau_neg = round(tau_neg, 4)
                best_metrics = metrics

            tau_neg = round(tau_neg + neg_step, 6)
        tau_pos = round(tau_pos + pos_step, 6)

    # Compute positive rate and confusion matrix at best thresholds
    best_predicted = [
        _apply_thresholds(s, best_tau_pos, best_tau_neg) for s in all_scores
    ]
    pos_rate = sum(1 for p in best_predicted if p == "positive") / len(best_predicted)

    confusion = defaultdict(lambda: defaultdict(int))
    for a, p in zip(actual_labels, best_predicted):
        confusion[a][p] += 1

    return {
        "best_tau_pos": best_tau_pos,
        "best_tau_neg": best_tau_neg,
        "macro_f1": round(best_f1, 4),
        "accuracy": round(best_metrics["accuracy"], 4),
        "macro_precision": round(best_metrics["macro_precision"], 4),
        "macro_recall": round(best_metrics["macro_recall"], 4),
        "positive_rate": round(pos_rate, 4),
        "per_class": {
            lbl: {k: round(v, 4) for k, v in m.items()}
            for lbl, m in best_metrics["per_class"].items()
        },
        "confusion_matrix": {a: dict(preds) for a, preds in confusion.items()},
        "candidates_checked": candidates_checked,
    }


# ── Reporting ───────────────────────────────────────────────────────────

def _print_report(result: dict, baseline_metrics: dict | None = None) -> None:
    """Print the grid search results."""
    print("\n" + "=" * 65)
    print("  DECISION THRESHOLD TUNING (grid search, macro F1)")
    print("=" * 65)
    print(f"  Candidates evaluated : {result['candidates_checked']:,}")
    print(f"  Best tau_pos         : {result['best_tau_pos']:.4f}")
    print(f"  Best tau_neg         : {result['best_tau_neg']:.4f}")
    print()

    # Metrics comparison
    if baseline_metrics:
        print(f"  {'Metric':<20} {'Baseline':>10} {'Tuned':>10} {'Delta':>10}")
        print(f"  {'-' * 52}")
        for key, label in [
            ("accuracy", "Accuracy"),
            ("macro_precision", "Macro Precision"),
            ("macro_recall", "Macro Recall"),
            ("macro_f1", "Macro F1"),
            ("positive_rate", "Pos. Rate"),
        ]:
            base_val = baseline_metrics.get(key, 0)
            tuned_val = result.get(key, 0)
            delta = tuned_val - base_val
            sign = "+" if delta >= 0 else ""
            print(f"  {label:<20} {base_val:>10.4f} {tuned_val:>10.4f} {sign}{delta:>9.4f}")
    else:
        print(f"  Accuracy        : {result['accuracy']:.4f}")
        print(f"  Macro Precision : {result['macro_precision']:.4f}")
        print(f"  Macro Recall    : {result['macro_recall']:.4f}")
        print(f"  Macro F1        : {result['best_macro_f1']:.4f}")
        print(f"  Pos. Rate       : {result['positive_rate']:.4f}")

    print()
    print(f"  PER-CLASS METRICS")
    print(f"  {'Class':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print(f"  {'-' * 42}")
    for lbl in ["negative", "neutral", "positive"]:
        m = result["per_class"][lbl]
        print(f"  {lbl:>10} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")

    print()
    print(f"  CONFUSION MATRIX")
    labels = ["negative", "neutral", "positive"]
    header = f"  {'Actual\\Pred':>12}" + "".join(f"{l:>10}" for l in labels)
    print(header)
    print(f"  {'-' * 42}")
    for actual in labels:
        row = f"  {actual:>12}"
        for pred in labels:
            row += f"{result['confusion_matrix'].get(actual, {}).get(pred, 0):>10}"
        print(row)

    print("=" * 65)


# ── Top-N results ──────────────────────────────────────────────────────

def grid_search_top_n(
    case_results: list[dict],
    top_n: int = 10,
    tau_pos_range: tuple[float, float, float] = (0.33, 0.70, 0.01),
    tau_neg_range: tuple[float, float, float] = (0.20, 0.60, 0.01),
) -> list[dict]:
    """Return the top-N threshold combinations by macro F1."""
    actual_labels = [c["actual_label"] for c in case_results]
    all_scores = [c["normalized_scores"] for c in case_results]

    results = []

    pos_start, pos_stop, pos_step = tau_pos_range
    neg_start, neg_stop, neg_step = tau_neg_range

    tau_pos = pos_start
    while tau_pos <= pos_stop + 1e-9:
        tau_neg = neg_start
        while tau_neg <= neg_stop + 1e-9:
            predicted = [
                _apply_thresholds(s, tau_pos, tau_neg) for s in all_scores
            ]
            metrics = _compute_metrics(actual_labels, predicted)
            pos_rate = sum(1 for p in predicted if p == "positive") / len(predicted)

            results.append({
                "tau_pos": round(tau_pos, 4),
                "tau_neg": round(tau_neg, 4),
                "macro_f1": round(metrics["macro_f1"], 4),
                "accuracy": round(metrics["accuracy"], 4),
                "pos_rate": round(pos_rate, 4),
                "neg_recall": round(metrics["per_class"]["negative"]["recall"], 4),
                "neu_recall": round(metrics["per_class"]["neutral"]["recall"], 4),
                "pos_recall": round(metrics["per_class"]["positive"]["recall"], 4),
            })

            tau_neg = round(tau_neg + neg_step, 6)
        tau_pos = round(tau_pos + pos_step, 6)

    results.sort(key=lambda x: x["macro_f1"], reverse=True)
    return results[:top_n]


def _print_top_n(top_results: list[dict]) -> None:
    """Print the top-N threshold combinations."""
    print(f"\n  TOP {len(top_results)} THRESHOLD COMBINATIONS (by macro F1)")
    print(f"  {'-' * 85}")
    header = (
        f"  {'#':>3} {'tau_pos':>8} {'tau_neg':>8} {'F1':>7} {'Acc':>7} "
        f"{'Pos%':>7} {'NegR':>7} {'NeuR':>7} {'PosR':>7}"
    )
    print(header)
    print(f"  {'-' * 85}")
    for i, r in enumerate(top_results, 1):
        print(
            f"  {i:>3} {r['tau_pos']:>8.4f} {r['tau_neg']:>8.4f} "
            f"{r['macro_f1']:>7.4f} {r['accuracy']:>7.4f} "
            f"{r['pos_rate']:>7.4f} {r['neg_recall']:>7.4f} "
            f"{r['neu_recall']:>7.4f} {r['pos_recall']:>7.4f}"
        )
    print()


# ── CLI entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Grid search for optimal decision thresholds"
    )
    parser.add_argument(
        "--results", type=str, default=str(DEFAULT_RESULTS),
        help="Path to evaluation_results.json with per-case normalized_scores",
    )
    parser.add_argument(
        "--step", type=float, default=0.01,
        help="Grid step size for both thresholds (default: 0.01)",
    )
    parser.add_argument(
        "--top-n", type=int, default=10,
        help="Show top N threshold combinations (default: 10)",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    case_results = data["case_results"]
    print(f"\n  Loaded {len(case_results)} cases from {results_path.name}")

    # Extract baseline metrics (argmax, no thresholds)
    baseline_metrics = {
        "accuracy": data["overall_metrics"]["accuracy"],
        "macro_precision": data["overall_metrics"]["macro_precision"],
        "macro_recall": data["overall_metrics"]["macro_recall"],
        "macro_f1": data["overall_metrics"]["macro_f1"],
        "positive_rate": data["metadata"]["positive_rate"],
    }

    # Grid search
    tau_pos_range = (0.33, 0.70, args.step)
    tau_neg_range = (0.20, 0.60, args.step)

    print(f"  Grid: tau_pos [{tau_pos_range[0]:.2f}, {tau_pos_range[1]:.2f}] "
          f"step {tau_pos_range[2]}, "
          f"tau_neg [{tau_neg_range[0]:.2f}, {tau_neg_range[1]:.2f}] "
          f"step {tau_neg_range[2]}")
    print(f"  Searching ...")

    best = grid_search(case_results, tau_pos_range, tau_neg_range)
    _print_report(best, baseline_metrics)

    # Top-N
    top = grid_search_top_n(case_results, args.top_n, tau_pos_range, tau_neg_range)
    _print_top_n(top)

    # Save results
    output_path = results_path.parent / "threshold_tuning_results.json"
    save_data = {
        "source_results": str(results_path),
        "grid": {
            "tau_pos_range": list(tau_pos_range),
            "tau_neg_range": list(tau_neg_range),
        },
        "baseline": baseline_metrics,
        "best": best,
        "top_n": top,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {output_path}\n")


if __name__ == "__main__":
    main()
