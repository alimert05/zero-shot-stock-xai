"""Split the 480-case test set into tune and holdout subsets.

Uses ticker-level isolation: all 24 cases for a given company go entirely
into one subset.  This prevents data leakage — tuned pipeline parameters
are validated on companies the tuning process never saw.

The optimal ticker assignment is found by exhaustive search over all valid
sector-constrained combinations, minimising the Sum of Absolute Deviations
(SAD) between holdout label proportions and overall distribution.

Constraints:
    - Every sector must have >= 1 ticker in both tune and holdout
    - Holdout contains exactly 8 tickers (40%)

Split:
    Tune   : 12 tickers, 288 cases (60%)
    Holdout:  8 tickers, 192 cases (40%)

Usage:
    python -m testset.split_test_set
    python -m testset.split_test_set --test-set path/to/test_set.json
    python -m testset.split_test_set --find-split   # only show search results
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from itertools import combinations, product
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PRED_PATH

TEST_SET_PATH = PRED_PATH / "test_set.json"
HOLDOUT_SIZE = 8

# -- Optimal ticker assignment (verified by exhaustive search) -----------
# Found by find_optimal_split(): minimises label-proportion SAD
# across 36,288 sector-constrained candidates.  SAD = 0.004167.
#
# Sector allocation: Tech=4, Finance=1, Healthcare=1, Consumer=1, Energy=1
# Tune  (12): AAPL, AMZN, CVX, GS, JNJ, JPM, MCD, META, NKE, PFE, V, WMT
# Holdout (8): BAC, GOOGL, KO, MSFT, NVDA, TSLA, UNH, XOM

HOLDOUT_TICKERS = frozenset({
    # Tech (4)
    "GOOGL", "MSFT", "NVDA", "TSLA",
    # Finance (1)
    "BAC",
    # Healthcare (1)
    "UNH",
    # Consumer (1)
    "KO",
    # Energy (1)
    "XOM",
})


# ── Exhaustive search ───────────────────────────────────────────────────

def find_optimal_split(
    test_set_path: str,
    holdout_size: int = HOLDOUT_SIZE,
) -> tuple[frozenset[str], dict]:
    """Find the holdout ticker set that minimises label-proportion deviation.

    Performs an exhaustive search over all valid sector-constrained
    combinations of tickers.

    Constraints
    -----------
    - Every sector must have >= 1 ticker in both tune and holdout.
    - Holdout contains exactly *holdout_size* tickers.

    Objective
    ---------
    Minimise::

        SAD = sum_{label} |holdout_proportion(label) - overall_proportion(label)|

    Since tune = overall - holdout, minimising holdout deviation
    simultaneously minimises tune deviation.

    Returns
    -------
    (best_holdout_tickers, search_info)
        search_info contains the objective value, proportions, and
        number of candidates evaluated.
    """
    with open(test_set_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_cases = data["test_cases"]

    # Per-ticker label counts and sector mapping
    ticker_labels: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ticker_sector: dict[str, str] = {}
    for c in all_cases:
        ticker_labels[c["ticker"]][c["actual_label"]] += 1
        ticker_sector[c["ticker"]] = c.get("sector", "Unknown")

    # Overall label proportions (optimisation target)
    total_cases = len(all_cases)
    overall_counts: dict[str, int] = defaultdict(int)
    for c in all_cases:
        overall_counts[c["actual_label"]] += 1
    overall_props = {lbl: cnt / total_cases for lbl, cnt in overall_counts.items()}
    labels = sorted(overall_props.keys())

    # Group tickers by sector (sorted for deterministic enumeration)
    sectors: dict[str, list[str]] = defaultdict(list)
    for t in sorted(ticker_sector):
        sectors[ticker_sector[t]].append(t)
    sector_names = sorted(sectors.keys())
    sector_tickers = [sectors[s] for s in sector_names]
    sector_sizes = [len(ts) for ts in sector_tickers]

    # ── Enumerate valid holdout-count allocations per sector ──
    # Each sector contributes k in [1, size-1] tickers to holdout,
    # such that sum(k_i) == holdout_size.
    def _enum_allocations(idx: int, remaining: int):
        if idx == len(sector_sizes):
            if remaining == 0:
                yield ()
            return
        lo, hi = 1, sector_sizes[idx] - 1
        for k in range(lo, min(hi, remaining) + 1):
            for rest in _enum_allocations(idx + 1, remaining - k):
                yield (k,) + rest

    best_score = float("inf")
    best_holdout: frozenset[str] | None = None
    best_allocation: dict[str, int] = {}
    best_holdout_props: dict[str, float] = {}
    candidates_checked = 0

    for allocation in _enum_allocations(0, holdout_size):
        # Per-sector ticker combinations for this allocation
        per_sector_combos = [
            list(combinations(sector_tickers[i], allocation[i]))
            for i in range(len(allocation))
        ]

        for combo in product(*per_sector_combos):
            holdout_set: list[str] = []
            for sector_combo in combo:
                holdout_set.extend(sector_combo)

            # Holdout label proportions
            holdout_total = sum(
                sum(ticker_labels[t].values()) for t in holdout_set
            )
            holdout_label_counts: dict[str, int] = defaultdict(int)
            for t in holdout_set:
                for lbl, cnt in ticker_labels[t].items():
                    holdout_label_counts[lbl] += cnt

            # SAD objective
            score = sum(
                abs(holdout_label_counts.get(lbl, 0) / holdout_total - overall_props[lbl])
                for lbl in labels
            )

            candidates_checked += 1

            if score < best_score:
                best_score = score
                best_holdout = frozenset(holdout_set)
                best_allocation = {
                    sector_names[i]: allocation[i]
                    for i in range(len(sector_names))
                }
                best_holdout_props = {
                    lbl: holdout_label_counts.get(lbl, 0) / holdout_total
                    for lbl in labels
                }

    # Compute tune proportions for the best split
    tune_tickers = set(ticker_sector.keys()) - best_holdout
    tune_total = sum(sum(ticker_labels[t].values()) for t in tune_tickers)
    tune_label_counts: dict[str, int] = defaultdict(int)
    for t in tune_tickers:
        for lbl, cnt in ticker_labels[t].items():
            tune_label_counts[lbl] += cnt
    best_tune_props = {
        lbl: tune_label_counts.get(lbl, 0) / tune_total for lbl in labels
    }

    search_info = {
        "candidates_checked": candidates_checked,
        "best_score_sad": round(best_score, 6),
        "sector_allocation": best_allocation,
        "overall_proportions": {lbl: round(overall_props[lbl], 4) for lbl in labels},
        "holdout_proportions": {lbl: round(v, 4) for lbl, v in best_holdout_props.items()},
        "tune_proportions": {lbl: round(v, 4) for lbl, v in best_tune_props.items()},
        "holdout_tickers": sorted(best_holdout),
        "tune_tickers": sorted(tune_tickers),
    }

    return best_holdout, search_info


# ── Build / split helpers ───────────────────────────────────────────────

def _build_metadata(
    cases: list[dict],
    original_metadata: dict,
    split_name: str,
    search_info: dict | None = None,
) -> dict:
    """Build metadata dict for a split subset."""
    tickers = sorted({c["ticker"] for c in cases})
    sectors = sorted({c.get("sector", "Unknown") for c in cases})
    windows = sorted({c["prediction_window_days"] for c in cases})

    label_dist = defaultdict(int)
    per_window = defaultdict(int)
    per_company = defaultdict(int)
    per_sector = defaultdict(int)

    for c in cases:
        label_dist[c["actual_label"]] += 1
        per_window[str(c["prediction_window_days"])] += 1
        per_company[c["ticker"]] += 1
        per_sector[c.get("sector", "Unknown")] += 1

    meta = {
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "split": split_name,
        "parent_test_set": "test_set.json",
        "split_method": "ticker_level_isolation",
        "split_objective": "minimise_label_SAD",
        "description": (
            f"{split_name.capitalize()} subset - ticker-level split for "
            f"pipeline parameter {'exploration' if split_name == 'tune' else 'final evaluation'}"
        ),
        "ground_truth_source": original_metadata.get("ground_truth_source", "yfinance"),
        "neutral_threshold_method": original_metadata.get("neutral_threshold_method"),
        "neutral_threshold_k": original_metadata.get("neutral_threshold_k"),
        "ewma_lambda": original_metadata.get("ewma_lambda"),
        "companies": tickers,
        "sectors": sectors,
        "prediction_windows": windows,
        "total_cases": len(cases),
        "label_distribution": dict(sorted(label_dist.items())),
        "cases_per_window": dict(sorted(per_window.items())),
        "cases_per_company": dict(sorted(per_company.items())),
        "cases_per_sector": dict(sorted(per_sector.items())),
    }

    if search_info is not None:
        meta["split_search"] = {
            "candidates_checked": search_info["candidates_checked"],
            "best_score_sad": search_info["best_score_sad"],
            "sector_allocation": search_info["sector_allocation"],
        }

    return meta


def split_test_set(
    test_set_path: str,
    holdout_tickers: frozenset[str] | None = None,
    search_info: dict | None = None,
) -> tuple[dict, dict]:
    """Split the test set into tune and holdout subsets.

    Parameters
    ----------
    test_set_path : str
        Path to the full test_set.json.
    holdout_tickers : frozenset[str] | None
        Tickers to assign to holdout.  If *None*, uses the module-level
        ``HOLDOUT_TICKERS`` constant.
    search_info : dict | None
        Output from ``find_optimal_split()`` to embed in metadata.

    Returns
    -------
    (tune_data, holdout_data) -- each is a full JSON-serialisable dict
    with ``metadata`` and ``test_cases`` keys.
    """
    if holdout_tickers is None:
        holdout_tickers = HOLDOUT_TICKERS

    with open(test_set_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    original_metadata = data.get("metadata", {})
    all_cases = data["test_cases"]

    tune_cases = [c for c in all_cases if c["ticker"] not in holdout_tickers]
    holdout_cases = [c for c in all_cases if c["ticker"] in holdout_tickers]

    tune_data = {
        "metadata": _build_metadata(tune_cases, original_metadata, "tune", search_info),
        "test_cases": tune_cases,
    }
    holdout_data = {
        "metadata": _build_metadata(holdout_cases, original_metadata, "holdout", search_info),
        "test_cases": holdout_cases,
    }

    return tune_data, holdout_data




def _print_search_report(search_info: dict) -> None:
    """Print the exhaustive search results."""
    print("\n" + "=" * 65)
    print("  OPTIMAL SPLIT SEARCH (exhaustive, sector-constrained)")
    print("=" * 65)
    print(f"  Candidates evaluated : {search_info['candidates_checked']:,}")
    print(f"  Objective (SAD)      : {search_info['best_score_sad']:.6f}")
    print(f"  Sector allocation    : {search_info['sector_allocation']}")
    print()

    # Proportions table
    labels = sorted(search_info["overall_proportions"].keys())
    header = f"  {'Label':>10}  {'Overall':>8}  {'Holdout':>8}  {'Tune':>8}  {'|H-O|':>8}"
    print(header)
    print(f"  {'-' * 52}")
    for lbl in labels:
        o = search_info["overall_proportions"][lbl]
        h = search_info["holdout_proportions"][lbl]
        t = search_info["tune_proportions"][lbl]
        d = abs(h - o)
        print(f"  {lbl:>10}  {o:>8.4f}  {h:>8.4f}  {t:>8.4f}  {d:>8.4f}")

    print()
    print(f"  Holdout tickers ({len(search_info['holdout_tickers'])}): "
          f"{', '.join(search_info['holdout_tickers'])}")
    print(f"  Tune tickers    ({len(search_info['tune_tickers'])}): "
          f"{', '.join(search_info['tune_tickers'])}")
    print("=" * 65)


def _print_split_report(
    tune_data: dict,
    holdout_data: dict,
    tune_path: Path,
    holdout_path: Path,
) -> None:
    """Print a summary of the generated split files."""
    print("\n" + "=" * 65)
    print("  TEST SET SPLIT (ticker-level isolation)")
    print("=" * 65)

    for name, data, path in [
        ("TUNE", tune_data, tune_path),
        ("HOLDOUT", holdout_data, holdout_path),
    ]:
        meta = data["metadata"]
        total = meta["total_cases"]
        labels = meta["label_distribution"]

        pos = labels.get("positive", 0)
        neg = labels.get("negative", 0)
        neu = labels.get("neutral", 0)

        print(f"\n  {name} ({total} cases) -> {path.name}")
        print(f"  {'-' * 55}")
        print(f"    Tickers : {', '.join(meta['companies'])}")
        print(f"    Sectors : {', '.join(meta['sectors'])}")
        print(
            f"    Labels  : pos={pos} ({pos/total:.1%})  "
            f"neg={neg} ({neg/total:.1%})  "
            f"neu={neu} ({neu/total:.1%})"
        )

    # Overall comparison
    total_all = tune_data["metadata"]["total_cases"] + holdout_data["metadata"]["total_cases"]
    t_labels = tune_data["metadata"]["label_distribution"]
    h_labels = holdout_data["metadata"]["label_distribution"]

    overall_pos = t_labels.get("positive", 0) + h_labels.get("positive", 0)
    overall_neg = t_labels.get("negative", 0) + h_labels.get("negative", 0)
    overall_neu = t_labels.get("neutral", 0) + h_labels.get("neutral", 0)

    print(f"\n  OVERALL ({total_all} cases)")
    print(f"  {'-' * 55}")
    print(
        f"    Labels  : pos={overall_pos} ({overall_pos/total_all:.1%})  "
        f"neg={overall_neg} ({overall_neg/total_all:.1%})  "
        f"neu={overall_neu} ({overall_neu/total_all:.1%})"
    )
    print("=" * 65 + "\n")


# ── CLI entry point ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Split test set into tune and holdout subsets"
    )
    parser.add_argument(
        "--test-set", type=str, default=str(TEST_SET_PATH),
        help="Path to the full test_set.json",
    )
    parser.add_argument(
        "--find-split", action="store_true",
        help="Run the exhaustive search and print results without writing files",
    )
    args = parser.parse_args()

    test_set_path = Path(args.test_set)

    # ── Step 1: Run exhaustive search ──
    print("\n  Running exhaustive split search ...")
    optimal_holdout, search_info = find_optimal_split(str(test_set_path))
    _print_search_report(search_info)

    # ── Step 2: Verify the hardcoded constant matches ──
    if optimal_holdout == HOLDOUT_TICKERS:
        print("  [OK] HOLDOUT_TICKERS matches exhaustive search result.\n")
    else:
        print("  [WARN] HOLDOUT_TICKERS does NOT match search result!")
        print(f"         Hardcoded : {sorted(HOLDOUT_TICKERS)}")
        print(f"         Optimal   : {sorted(optimal_holdout)}")
        print("         Using search result for this run.\n")

    if args.find_split:
        return

    # ── Step 3: Generate split files ──
    output_dir = test_set_path.parent
    tune_path = output_dir / "tune_set.json"
    holdout_path = output_dir / "holdout_set.json"

    tune_data, holdout_data = split_test_set(
        str(test_set_path),
        holdout_tickers=optimal_holdout,
        search_info=search_info,
    )

    with open(tune_path, "w", encoding="utf-8") as f:
        json.dump(tune_data, f, indent=2, ensure_ascii=False)

    with open(holdout_path, "w", encoding="utf-8") as f:
        json.dump(holdout_data, f, indent=2, ensure_ascii=False)

    _print_split_report(tune_data, holdout_data, tune_path, holdout_path)

    print(f"  Saved: {tune_path}")
    print(f"  Saved: {holdout_path}\n")


if __name__ == "__main__":
    main()
