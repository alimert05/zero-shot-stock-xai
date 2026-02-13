"""Generate evaluation test set with pre-computed ground truth from yfinance.

Creates ~120 test cases covering:
    - 5 companies (AAPL, MSFT, GOOGL, AMZN, NVDA)
    - 6 prediction windows (1, 3, 5, 7, 14, 31 days)
    - 4 market periods per company×window

Ground truth labels (positive/negative/neutral) are fetched from yfinance
and stored in the JSON so evaluations are deterministic and fast.

Usage:
    python -m testset.generate_testset
"""

from __future__ import annotations

import json
import math
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import yfinance as yf

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PRED_PATH

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# ── Configuration ──

COMPANIES = [
    {"name": "Apple", "ticker": "AAPL"},
    {"name": "Microsoft", "ticker": "MSFT"},
    {"name": "Alphabet", "ticker": "GOOGL"},
    {"name": "Amazon", "ticker": "AMZN"},
    {"name": "Nvidia", "ticker": "NVDA"},
]

PREDICTION_WINDOWS = [1, 3, 5, 7, 14, 31]

# 4 market periods with distinct characteristics
MARKET_PERIODS = [
    {
        "label": "bull_run",
        "description": "Q1 2024 bull market rally",
        "base_dates": [
            "08-01-2024",
            "22-01-2024",
            "05-02-2024",
            "19-02-2024",
            "04-03-2024",
        ],
    },
    {
        "label": "volatile",
        "description": "Aug 2024 sell-off / volatility",
        "base_dates": [
            "01-08-2024",
            "05-08-2024",
            "12-08-2024",
            "19-08-2024",
            "26-08-2024",
        ],
    },
    {
        "label": "earnings_season",
        "description": "Q3 2024 earnings season (Oct-Nov)",
        "base_dates": [
            "14-10-2024",
            "21-10-2024",
            "28-10-2024",
            "04-11-2024",
            "11-11-2024",
        ],
    },
    {
        "label": "recent_stable",
        "description": "Late 2024 / early 2025 period",
        "base_dates": [
            "02-12-2024",
            "09-12-2024",
            "16-12-2024",
            "23-12-2024",
            "06-01-2025",
        ],
    },
]

NEUTRAL_THRESHOLD = 0.003
MAX_LOOKAHEAD_DAYS = 10
OUTPUT_PATH = PRED_PATH / "test_set.json"


# ── Ground Truth Fetching ──

def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%d-%m-%Y")


def _next_open_day_close(
    ticker: str, day: datetime, max_lookahead: int = MAX_LOOKAHEAD_DAYS
) -> tuple[datetime, float]:
    """Find the next trading day close price on or after `day`."""
    df = yf.download(
        ticker,
        start=day.strftime("%Y-%m-%d"),
        end=(day + timedelta(days=max_lookahead + 1)).strftime("%Y-%m-%d"),
        interval="1d",
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise ValueError(f"No price data for {ticker} around {day.date()}")

    df = df.sort_index()
    for idx in df.index:
        if idx.date() >= day.date():
            close_val = df.loc[idx, "Close"]
            if hasattr(close_val, "iloc"):
                close_val = close_val.iloc[0]
            return idx.to_pydatetime(), float(close_val)

    raise ValueError(
        f"No open day found for {ticker} within {max_lookahead} days after {day.date()}"
    )


def get_ground_truth(
    ticker: str,
    start_date: str,
    end_date: str,
    neutral_threshold: float = NEUTRAL_THRESHOLD,
) -> dict:
    """Compute actual label and metadata from yfinance price data."""
    s0 = _parse_date(start_date)
    e0 = _parse_date(end_date)

    s_dt, s_px = _next_open_day_close(ticker, s0)
    e_dt, e_px = _next_open_day_close(ticker, e0)

    pct = (e_px - s_px) / s_px if s_px else 0.0

    if pct > neutral_threshold:
        label = "positive"
    elif pct < -neutral_threshold:
        label = "negative"
    else:
        label = "neutral"

    return {
        "actual_label": label,
        "actual_pct_change": round(pct, 6),
        "used_start_date": s_dt.strftime("%Y-%m-%d"),
        "used_end_date": e_dt.strftime("%Y-%m-%d"),
        "used_start_close": round(s_px, 2),
        "used_end_close": round(e_px, 2),
    }


# ── Test Case Generation ──

def _compute_backward_days(prediction_window: int) -> int:
    """Mirror the backward lookup formula from fetcher.py."""
    return max(7, min(90, math.ceil(5 * math.sqrt(prediction_window))))


def generate_test_cases() -> list[dict]:
    """Generate all test cases with ground truth."""
    test_cases = []
    case_id = 0

    for company in COMPANIES:
        ticker = company["ticker"]
        name = company["name"]

        for window in PREDICTION_WINDOWS:
            for period in MARKET_PERIODS:
                company_idx = COMPANIES.index(company)
                date_idx = company_idx % len(period["base_dates"])
                base_date_str = period["base_dates"][date_idx]

                start_dt = _parse_date(base_date_str)
                end_dt = start_dt + timedelta(days=window)

                start_date = start_dt.strftime("%d-%m-%Y")
                end_date = end_dt.strftime("%d-%m-%Y")

                case_id += 1
                test_id = f"{ticker}_W{window}_{period['label']}_{case_id:03d}"

                logger.info(
                    "Generating case %s: %s %s→%s (W=%d)",
                    test_id, ticker, start_date, end_date, window,
                )

                try:
                    ground_truth = get_ground_truth(ticker, start_date, end_date)
                except Exception as exc:
                    logger.warning("Skipping %s: %s", test_id, exc)
                    continue

                test_case = {
                    "id": test_id,
                    "company_name": name,
                    "ticker": ticker,
                    "start_date": start_date,
                    "end_date": end_date,
                    "prediction_window_days": window,
                    "expected_backward_days": _compute_backward_days(window),
                    "market_period": period["label"],
                    "market_period_description": period["description"],
                    **ground_truth,
                }
                test_cases.append(test_case)

    return test_cases


def generate_and_save() -> None:
    """Generate the full test set and save to JSON."""
    logger.info("Starting test set generation...")

    test_cases = generate_test_cases()

    label_counts = {}
    window_counts = {}
    company_counts = {}
    for case in test_cases:
        label = case["actual_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        w = case["prediction_window_days"]
        window_counts[w] = window_counts.get(w, 0) + 1
        t = case["ticker"]
        company_counts[t] = company_counts.get(t, 0) + 1

    output = {
        "metadata": {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Evaluation test set for news-based stock prediction pipeline",
            "ground_truth_source": "yfinance",
            "neutral_threshold": NEUTRAL_THRESHOLD,
            "companies": [c["ticker"] for c in COMPANIES],
            "prediction_windows": PREDICTION_WINDOWS,
            "total_cases": len(test_cases),
            "label_distribution": label_counts,
            "cases_per_window": window_counts,
            "cases_per_company": company_counts,
        },
        "test_cases": test_cases,
    }

    PRED_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("=" * 50)
    logger.info("Test set generated: %d cases", len(test_cases))
    logger.info("Label distribution: %s", label_counts)
    logger.info("Cases per window: %s", window_counts)
    logger.info("Cases per company: %s", company_counts)
    logger.info("Saved to: %s", OUTPUT_PATH)
    logger.info("=" * 50)


if __name__ == "__main__":
    generate_and_save()

