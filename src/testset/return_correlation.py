"""
Return Correlation Analysis
============================
Measures how well the pipeline's sentiment predictions correlate
with actual stock price movements on the holdout set.

Metrics computed:
  - Directional accuracy (did prediction match price direction?)
  - Spearman ρ  (rank correlation: polarity score vs actual return)
  - Pearson  r  (linear correlation: polarity score vs actual return)
  - Per-class mean return (avg return when predicted pos / neg / neu)
  - Per-window breakdown (same metrics split by prediction horizon)
  - Confidence-weighted correlation (higher confidence → better signal?)

Usage:
    python -m src.testset.return_correlation
    python -m src.testset.return_correlation --eval path/to/evaluation_results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

# ── paths ───────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "test_results"

DEFAULT_HOLDOUT = DATA / "holdout_set.json"
DEFAULT_EVAL    = DATA / "evaluation_results_fingpt.json"
OUTPUT_FILE     = DATA / "return_correlation_results_fingpt.json"


# ── helpers ─────────────────────────────────────────────────────────────

def _load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _polarity(scores: dict[str, float]) -> float:
    """Sentiment polarity = positive − negative  (range −1 … +1)."""
    return scores.get("positive", 0.0) - scores.get("negative", 0.0)


def _direction(value: float) -> int:
    """Map a continuous value to +1 / −1 / 0."""
    if value > 0:
        return 1
    elif value < 0:
        return -1
    return 0


def _safe_corr(x: list[float], y: list[float]):
    """Return (statistic, p-value) or (None, None) if not enough data."""
    if len(x) < 3:
        return None, None
    result = stats.spearmanr(x, y)
    return float(np.nan_to_num(result.statistic)), \
           float(np.nan_to_num(result.pvalue))


def _safe_pearson(x: list[float], y: list[float]):
    if len(x) < 3:
        return None, None
    r, p = stats.pearsonr(x, y)
    return float(np.nan_to_num(r)), float(np.nan_to_num(p))


# ── core analysis ──────────────────────────────────────────────────────

def analyse(holdout_path: Path, eval_path: Path) -> dict:
    """Run the full return-correlation analysis and return a results dict."""
    holdout = _load_json(holdout_path)
    evaluation = _load_json(eval_path)

    # Build lookup:  case_id → ground-truth fields
    gt_lookup: dict[str, dict] = {}
    for case in holdout["test_cases"]:
        gt_lookup[case["id"]] = case

    # Merge predictions with ground truth
    merged: list[dict] = []
    for pred in evaluation["case_results"]:
        cid = pred["id"]
        gt = gt_lookup.get(cid)
        if gt is None:
            continue  # skip cases not in holdout
        if pred.get("error") or pred.get("issue"):
            continue  # skip failed cases

        scores = pred["normalized_scores"]
        polarity = _polarity(scores)
        confidence = pred["final_confidence"]
        actual_ret = gt["actual_pct_change"]

        merged.append({
            "id":                   cid,
            "ticker":               pred["ticker"],
            "sector":               pred.get("sector", gt.get("sector")),
            "window":               pred["prediction_window_days"],
            "market_period":        pred.get("market_period", gt.get("market_period")),
            "predicted_label":      pred["predicted_label"],
            "actual_label":         gt["actual_label"],
            "polarity_score":       polarity,
            "confidence":           confidence,
            "actual_pct_change":    actual_ret,
            "positive_score":       scores.get("positive", 0),
            "negative_score":       scores.get("negative", 0),
            "neutral_score":        scores.get("neutral", 0),
        })

    n = len(merged)
    if n == 0:
        print("ERROR: No cases matched between holdout and evaluation.")
        sys.exit(1)

    # ── 1. Overall correlations ────────────────────────────────────────
    polarities = [m["polarity_score"] for m in merged]
    returns    = [m["actual_pct_change"] for m in merged]

    spearman_rho, spearman_p = _safe_corr(polarities, returns)
    pearson_r, pearson_p     = _safe_pearson(polarities, returns)

    # ── 2. Directional accuracy ────────────────────────────────────────
    #    Only for non-neutral predictions (neutral has no direction)
    dir_correct = 0
    dir_total   = 0
    for m in merged:
        if m["predicted_label"] == "neutral":
            continue
        pred_dir = 1 if m["predicted_label"] == "positive" else -1
        actual_dir = _direction(m["actual_pct_change"])
        if actual_dir == 0:
            continue  # actual return exactly 0 — skip
        dir_total += 1
        if pred_dir == actual_dir:
            dir_correct += 1

    directional_accuracy = dir_correct / dir_total if dir_total > 0 else None

    # ── 2b. Directional-only correlations (exclude neutral predictions) ─
    directional_cases = [m for m in merged if m["predicted_label"] != "neutral"]
    dir_polarities = [m["polarity_score"] for m in directional_cases]
    dir_returns    = [m["actual_pct_change"] for m in directional_cases]

    dir_spearman_rho, dir_spearman_p = _safe_corr(dir_polarities, dir_returns)
    dir_pearson_r, dir_pearson_p     = _safe_pearson(dir_polarities, dir_returns)

    # ── 3. Per-class mean return ───────────────────────────────────────
    class_returns: dict[str, list[float]] = defaultdict(list)
    for m in merged:
        class_returns[m["predicted_label"]].append(m["actual_pct_change"])

    per_class_mean_return = {}
    for label in ["positive", "negative", "neutral"]:
        rets = class_returns.get(label, [])
        per_class_mean_return[label] = {
            "count": len(rets),
            "mean_return_pct": round(np.mean(rets) * 100, 4) if rets else None,
            "median_return_pct": round(float(np.median(rets)) * 100, 4) if rets else None,
            "std_return_pct": round(float(np.std(rets)) * 100, 4) if rets else None,
        }

    # ── 4. Confidence-weighted correlation ─────────────────────────────
    #    Weight polarity by confidence → does higher confidence = better?
    weighted_polarities = [m["polarity_score"] * m["confidence"] for m in merged]
    conf_spearman_rho, conf_spearman_p = _safe_corr(weighted_polarities, returns)

    # Split into high-confidence vs low-confidence halves
    sorted_by_conf = sorted(merged, key=lambda m: m["confidence"])
    mid = len(sorted_by_conf) // 2
    low_conf  = sorted_by_conf[:mid]
    high_conf = sorted_by_conf[mid:]

    high_pol = [m["polarity_score"] for m in high_conf]
    high_ret = [m["actual_pct_change"] for m in high_conf]
    low_pol  = [m["polarity_score"] for m in low_conf]
    low_ret  = [m["actual_pct_change"] for m in low_conf]

    high_spearman, high_p = _safe_corr(high_pol, high_ret)
    low_spearman, low_p   = _safe_corr(low_pol, low_ret)

    # ── 5. Per-window breakdown ────────────────────────────────────────
    window_groups: dict[int, list[dict]] = defaultdict(list)
    for m in merged:
        window_groups[m["window"]].append(m)

    per_window = {}
    for w in sorted(window_groups.keys()):
        group = window_groups[w]
        w_pol = [m["polarity_score"] for m in group]
        w_ret = [m["actual_pct_change"] for m in group]
        w_spearman, w_sp = _safe_corr(w_pol, w_ret)
        w_pearson, w_pp  = _safe_pearson(w_pol, w_ret)

        # directional accuracy for this window
        w_dir_correct = 0
        w_dir_total = 0
        for m in group:
            if m["predicted_label"] == "neutral":
                continue
            pred_dir = 1 if m["predicted_label"] == "positive" else -1
            actual_dir = _direction(m["actual_pct_change"])
            if actual_dir == 0:
                continue
            w_dir_total += 1
            if pred_dir == actual_dir:
                w_dir_correct += 1

        # per-class for this window
        w_class_ret: dict[str, list[float]] = defaultdict(list)
        for m in group:
            w_class_ret[m["predicted_label"]].append(m["actual_pct_change"])

        per_window[f"W={w}"] = {
            "n": len(group),
            "spearman_rho": w_spearman,
            "spearman_p": w_sp,
            "pearson_r": w_pearson,
            "pearson_p": w_pp,
            "directional_accuracy": w_dir_correct / w_dir_total if w_dir_total > 0 else None,
            "directional_n": w_dir_total,
            "mean_return_by_class": {
                label: round(np.mean(rets) * 100, 4) if rets else None
                for label, rets in w_class_ret.items()
            },
        }

    # ── 6. Per-ticker breakdown ────────────────────────────────────────
    ticker_groups: dict[str, list[dict]] = defaultdict(list)
    for m in merged:
        ticker_groups[m["ticker"]].append(m)

    per_ticker = {}
    for t in sorted(ticker_groups.keys()):
        group = ticker_groups[t]
        t_pol = [m["polarity_score"] for m in group]
        t_ret = [m["actual_pct_change"] for m in group]
        t_spearman, t_sp = _safe_corr(t_pol, t_ret)

        t_dir_correct = 0
        t_dir_total = 0
        for m in group:
            if m["predicted_label"] == "neutral":
                continue
            pred_dir = 1 if m["predicted_label"] == "positive" else -1
            actual_dir = _direction(m["actual_pct_change"])
            if actual_dir == 0:
                continue
            t_dir_total += 1
            if pred_dir == actual_dir:
                t_dir_correct += 1

        per_ticker[t] = {
            "n": len(group),
            "spearman_rho": t_spearman,
            "spearman_p": t_sp,
            "directional_accuracy": t_dir_correct / t_dir_total if t_dir_total > 0 else None,
            "directional_n": t_dir_total,
        }

    # ── 7. Per-market-period breakdown ─────────────────────────────────
    period_groups: dict[str, list[dict]] = defaultdict(list)
    for m in merged:
        period_groups[m["market_period"]].append(m)

    per_period = {}
    for p in sorted(period_groups.keys()):
        group = period_groups[p]
        p_pol = [m["polarity_score"] for m in group]
        p_ret = [m["actual_pct_change"] for m in group]
        p_spearman, p_sp = _safe_corr(p_pol, p_ret)

        per_period[p] = {
            "n": len(group),
            "spearman_rho": p_spearman,
            "spearman_p": p_sp,
        }

    # ── Assemble results ───────────────────────────────────────────────
    results = {
        "metadata": {
            "holdout_file": str(holdout_path),
            "evaluation_file": str(eval_path),
            "total_cases_matched": n,
            "model": evaluation["metadata"].get("sentiment_model", "unknown"),
        },
        "overall_all_cases": {
            "n": n,
            "spearman_rho": spearman_rho,
            "spearman_p": spearman_p,
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
        },
        "directional_only": {
            "n": len(directional_cases),
            "spearman_rho": dir_spearman_rho,
            "spearman_p": dir_spearman_p,
            "pearson_r": dir_pearson_r,
            "pearson_p": dir_pearson_p,
            "directional_accuracy": directional_accuracy,
            "directional_n": dir_total,
        },
        "per_class_mean_return": per_class_mean_return,
        "confidence_analysis": {
            "weighted_spearman_rho": conf_spearman_rho,
            "weighted_spearman_p": conf_spearman_p,
            "high_confidence_half": {
                "n": len(high_conf),
                "spearman_rho": high_spearman,
                "spearman_p": high_p,
            },
            "low_confidence_half": {
                "n": len(low_conf),
                "spearman_rho": low_spearman,
                "spearman_p": low_p,
            },
        },
        "per_window": per_window,
        "per_ticker": per_ticker,
        "per_market_period": per_period,
    }

    return results


# ── pretty print ───────────────────────────────────────────────────────

def print_report(results: dict) -> None:
    """Print a formatted summary of return-correlation results to stdout."""
    ov_all = results["overall_all_cases"]
    ov_dir = results["directional_only"]
    pcr = results["per_class_mean_return"]
    ca = results["confidence_analysis"]

    print("\n" + "=" * 70)
    print("  STOCK RETURN CORRELATION ANALYSIS")
    print("=" * 70)
    print(f"  Model: {results['metadata']['model']}")
    print(f"  Cases: {results['metadata']['total_cases_matched']}")
    print("=" * 70)

    # All cases
    print("\n-- All Cases (n=%d) -------------------------------------" % ov_all["n"])
    _rho = f"{ov_all['spearman_rho']:.4f}" if ov_all["spearman_rho"] is not None else "N/A"
    _sp  = f"{ov_all['spearman_p']:.4f}" if ov_all["spearman_p"] is not None else "N/A"
    _pr  = f"{ov_all['pearson_r']:.4f}" if ov_all["pearson_r"] is not None else "N/A"
    _pp  = f"{ov_all['pearson_p']:.4f}" if ov_all["pearson_p"] is not None else "N/A"
    print(f"  Spearman rho (polarity vs return): {_rho}  (p = {_sp})")
    print(f"  Pearson  r   (polarity vs return): {_pr}  (p = {_pp})")

    # Directional only
    print("\n-- Directional Only (n=%d, excludes neutral predictions) -" % ov_dir["n"])
    if ov_dir["spearman_rho"] is not None:
        print(f"  Spearman rho (polarity vs return): {ov_dir['spearman_rho']:.4f}  (p = {ov_dir['spearman_p']:.4f})")
        print(f"  Pearson  r   (polarity vs return): {ov_dir['pearson_r']:.4f}  (p = {ov_dir['pearson_p']:.4f})")
    else:
        print("  Not enough directional cases for correlation")
    if ov_dir["directional_accuracy"] is not None:
        print(f"  Directional accuracy:              {ov_dir['directional_accuracy']:.1%}  ({ov_dir['directional_n']} cases with non-zero return)")
    else:
        print("  Directional accuracy:              N/A")

    # Per-class mean return
    print("\n-- Per-Class Mean Return --------------------------------")
    print(f"  {'Class':<12} {'Count':>6} {'Mean %':>10} {'Median %':>10} {'Std %':>10}")
    print(f"  {'-'*50}")
    for label in ["positive", "negative", "neutral"]:
        d = pcr[label]
        cnt = d["count"]
        mean = f"{d['mean_return_pct']:+.4f}" if d["mean_return_pct"] is not None else "N/A"
        med  = f"{d['median_return_pct']:+.4f}" if d["median_return_pct"] is not None else "N/A"
        std  = f"{d['std_return_pct']:.4f}" if d["std_return_pct"] is not None else "N/A"
        print(f"  {label:<12} {cnt:>6} {mean:>10} {med:>10} {std:>10}")

    # Confidence analysis
    print("\n-- Confidence Analysis ----------------------------------")
    _wr = f"{ca['weighted_spearman_rho']:.4f}" if ca["weighted_spearman_rho"] is not None else "N/A"
    _wp = f"{ca['weighted_spearman_p']:.4f}" if ca["weighted_spearman_p"] is not None else "N/A"
    print(f"  Weighted polarity Spearman rho:    {_wr}  (p = {_wp})")
    hc = ca["high_confidence_half"]
    lc = ca["low_confidence_half"]
    _hr = f"{hc['spearman_rho']:.4f}" if hc["spearman_rho"] is not None else "N/A"
    _hp = f"{hc['spearman_p']:.4f}" if hc["spearman_p"] is not None else "N/A"
    _lr = f"{lc['spearman_rho']:.4f}" if lc["spearman_rho"] is not None else "N/A"
    _lp = f"{lc['spearman_p']:.4f}" if lc["spearman_p"] is not None else "N/A"
    print(f"  High-confidence half (n={hc['n']}):     rho = {_hr}  (p = {_hp})")
    print(f"  Low-confidence  half (n={lc['n']}):     rho = {_lr}  (p = {_lp})")

    # Per-window
    print("\n-- Per-Window Breakdown ---------------------------------")
    print(f"  {'Window':<8} {'N':>4} {'rho':>8} {'p':>8} {'Dir.Acc':>8} {'DirN':>5}")
    print(f"  {'-'*43}")
    for w, d in results["per_window"].items():
        rho = f"{d['spearman_rho']:.4f}" if d["spearman_rho"] is not None else "N/A"
        p   = f"{d['spearman_p']:.4f}" if d["spearman_p"] is not None else "N/A"
        da  = f"{d['directional_accuracy']:.1%}" if d["directional_accuracy"] is not None else "N/A"
        print(f"  {w:<8} {d['n']:>4} {rho:>8} {p:>8} {da:>8} {d['directional_n']:>5}")

    # Per-ticker
    print("\n-- Per-Ticker Breakdown ---------------------------------")
    print(f"  {'Ticker':<8} {'N':>4} {'rho':>8} {'p':>8} {'Dir.Acc':>8} {'DirN':>5}")
    print(f"  {'-'*43}")
    for t, d in results["per_ticker"].items():
        rho = f"{d['spearman_rho']:.4f}" if d["spearman_rho"] is not None else "N/A"
        p   = f"{d['spearman_p']:.4f}" if d["spearman_p"] is not None else "N/A"
        da  = f"{d['directional_accuracy']:.1%}" if d["directional_accuracy"] is not None else "N/A"
        print(f"  {t:<8} {d['n']:>4} {rho:>8} {p:>8} {da:>8} {d['directional_n']:>5}")

    # Per-market-period
    print("\n-- Per-Market-Period Breakdown --------------------------")
    print(f"  {'Period':<20} {'N':>4} {'rho':>8} {'p':>8}")
    print(f"  {'-'*42}")
    for p_name, d in results["per_market_period"].items():
        rho = f"{d['spearman_rho']:.4f}" if d["spearman_rho"] is not None else "N/A"
        p   = f"{d['spearman_p']:.4f}" if d["spearman_p"] is not None else "N/A"
        print(f"  {p_name:<20} {d['n']:>4} {rho:>8} {p:>8}")

    print("\n" + "=" * 70)


# ── main ───────────────────────────────────────────────────────────────

def main():
    """CLI entry point: load data, run analysis, print report, and save JSON."""
    parser = argparse.ArgumentParser(description="Return correlation analysis")
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT,
                        help="Path to holdout_set.json")
    parser.add_argument("--eval", type=Path, default=DEFAULT_EVAL,
                        help="Path to evaluation_results.json")
    parser.add_argument("--output", type=Path, default=OUTPUT_FILE,
                        help="Path to save JSON results")
    args = parser.parse_args()

    print(f"Loading holdout set: {args.holdout}")
    print(f"Loading evaluation:  {args.eval}")

    results = analyse(args.holdout, args.eval)
    print_report(results)

    # Save JSON
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
