"""Generate evaluation test set with pre-computed ground truth from yfinance.

<<<<<<< HEAD
Creates ~120 test cases covering:
    - 5 companies (AAPL, MSFT, GOOGL, AMZN, NVDA)
=======
Creates ~480 test cases covering:
    - 20 companies across 5 sectors (Tech, Finance, Healthcare, Energy, Consumer)
>>>>>>> fix/positive_bias
    - 6 prediction windows (1, 3, 5, 7, 14, 31 days)
    - 4 market periods per company×window

Ground truth labels (positive/negative/neutral) are fetched from yfinance
and stored in the JSON so evaluations are deterministic and fast.

<<<<<<< HEAD
Neutral threshold is computed per-company as k × σ_daily (volatility-scaled),
following López de Prado (2018) "Advances in Financial Machine Learning" and
Bollinger (1983). A single fixed threshold fails across different volatility
regimes — Apple (σ ≈ 1.5%/day) needs a much wider neutral band than a
low-volatility utility stock.
=======
Neutral threshold is computed per test case as:

    threshold = k × EWMA_sigma(as of start_date) × sqrt(window_days)

This is a point-in-time, volatility-scaled neutral band:
    - point-in-time: only data available before the case start date is used
    - EWMA: recent volatility is weighted more heavily
    - sqrt(window): horizon-aware scaling of daily volatility
>>>>>>> fix/positive_bias

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
<<<<<<< HEAD
    {"name": "Apple", "ticker": "AAPL"},
    {"name": "Microsoft", "ticker": "MSFT"},
    {"name": "Alphabet", "ticker": "GOOGL"},
    {"name": "Amazon", "ticker": "AMZN"},
    {"name": "Nvidia", "ticker": "NVDA"},
=======
    # Tech (7)
    {"name": "Apple", "ticker": "AAPL", "sector": "Tech"},
    {"name": "Microsoft", "ticker": "MSFT", "sector": "Tech"},
    {"name": "Alphabet", "ticker": "GOOGL", "sector": "Tech"},
    {"name": "Amazon", "ticker": "AMZN", "sector": "Tech"},
    {"name": "Nvidia", "ticker": "NVDA", "sector": "Tech"},
    {"name": "Meta Platforms", "ticker": "META", "sector": "Tech"},
    {"name": "Tesla", "ticker": "TSLA", "sector": "Tech"},
    # Finance (4)
    {"name": "JPMorgan Chase", "ticker": "JPM", "sector": "Finance"},
    {"name": "Bank of America", "ticker": "BAC", "sector": "Finance"},
    {"name": "Goldman Sachs", "ticker": "GS", "sector": "Finance"},
    {"name": "Visa", "ticker": "V", "sector": "Finance"},
    # Healthcare (3)
    {"name": "Johnson & Johnson", "ticker": "JNJ", "sector": "Healthcare"},
    {"name": "Pfizer", "ticker": "PFE", "sector": "Healthcare"},
    {"name": "UnitedHealth Group", "ticker": "UNH", "sector": "Healthcare"},
    # Energy (2)
    {"name": "Exxon Mobil", "ticker": "XOM", "sector": "Energy"},
    {"name": "Chevron", "ticker": "CVX", "sector": "Energy"},
    # Consumer (4)
    {"name": "Walmart", "ticker": "WMT", "sector": "Consumer"},
    {"name": "Coca-Cola", "ticker": "KO", "sector": "Consumer"},
    {"name": "McDonald's", "ticker": "MCD", "sector": "Consumer"},
    {"name": "Nike", "ticker": "NKE", "sector": "Consumer"},
>>>>>>> fix/positive_bias
]

PREDICTION_WINDOWS = [1, 3, 5, 7, 14, 31]

<<<<<<< HEAD
# 4 market periods with distinct characteristics
MARKET_PERIODS = [
    {
        "label": "bull_run",
        "description": "Q1 2025 market period (Mar–Apr)",
        "base_dates": [
            "03-03-2025",
            "10-03-2025",
            "17-03-2025",
            "24-03-2025",
            "07-04-2025",
=======
MARKET_PERIODS = [
    {
        "label": "bull_run",
        "description": "Q1 2025 market period (Apr–May)",
        "base_dates": [
            "03-04-2025",
            "10-04-2025",
            "17-04-2025",
            "24-04-2025",
            "07-05-2025",
>>>>>>> fix/positive_bias
        ],
    },
    {
        "label": "volatile",
        "description": "Aug 2025 sell-off / volatility",
        "base_dates": [
            "01-08-2025",
            "05-08-2025",
            "12-08-2025",
            "19-08-2025",
            "26-08-2025",
        ],
    },
    {
        "label": "earnings_season",
        "description": "Q3 2025 earnings season (Oct-Nov)",
        "base_dates": [
            "14-10-2025",
            "21-10-2025",
            "28-10-2025",
            "04-11-2025",
            "11-11-2025",
        ],
    },
    {
        "label": "recent_stable",
<<<<<<< HEAD
        "description": "Late 2024 / early 2026 period",
=======
        "description": "Late 2025 / early 2026 period",
>>>>>>> fix/positive_bias
        "base_dates": [
            "02-12-2025",
            "09-12-2025",
            "16-12-2025",
            "23-12-2025",
            "06-01-2026",
        ],
    },
]

<<<<<<< HEAD
VOLATILITY_K             = 0.5   # |return| must exceed k×σ to be directional
VOLATILITY_LOOKBACK_DAYS = 60    # calendar days of history used to estimate σ_daily
FALLBACK_NEUTRAL_THRESHOLD = 0.005  # used only when yfinance data is unavailable
=======
VOLATILITY_K = 0.5
EWMA_LAMBDA = 0.94
EWMA_LOOKBACK_CALENDAR_DAYS = 180
MIN_RETURNS_FOR_EWMA = 20
FALLBACK_NEUTRAL_THRESHOLD = 0.005
>>>>>>> fix/positive_bias

MAX_LOOKAHEAD_DAYS = 10
OUTPUT_PATH = PRED_PATH / "test_set.json"


<<<<<<< HEAD
def compute_volatility_threshold(
    ticker: str,
    k: float = VOLATILITY_K,
    lookback_days: int = VOLATILITY_LOOKBACK_DAYS,
) -> float:
    try:
        hist = yf.download(
            ticker,
            period=f"{lookback_days}d",
=======
# ── Volatility Thresholding ──

def compute_point_in_time_ewma_threshold(
    ticker: str,
    asof_date: datetime,
    horizon_days: int,
    k: float = VOLATILITY_K,
    ewma_lambda: float = EWMA_LAMBDA,
    lookback_calendar_days: int = EWMA_LOOKBACK_CALENDAR_DAYS,
) -> float:
    """
    Compute point-in-time EWMA daily volatility as of `asof_date`,
    then scale it to the requested horizon with sqrt(window_days).

    Uses only data strictly before asof_date.
    """
    try:
        lookback_start = asof_date - timedelta(days=lookback_calendar_days)

        hist = yf.download(
            ticker,
            start=lookback_start.strftime("%Y-%m-%d"),
            end=asof_date.strftime("%Y-%m-%d"),
>>>>>>> fix/positive_bias
            interval="1d",
            progress=False,
            auto_adjust=True,
        )
<<<<<<< HEAD
        if hist is None or hist.empty:
            raise ValueError("No price data returned")
=======

        if hist is None or hist.empty:
            raise ValueError("No historical data returned")
>>>>>>> fix/positive_bias

        closes = hist["Close"]
        if hasattr(closes, "squeeze"):
            closes = closes.squeeze()

        daily_returns = closes.pct_change().dropna()
<<<<<<< HEAD
        if len(daily_returns) < 10:
            raise ValueError(f"Too few data points ({len(daily_returns)})")

        sigma = float(daily_returns.std())
        threshold = k * sigma

        logger.info(
            "  %s: σ_daily=%.4f  →  threshold = %.2f × %.4f = %.4f  (%.2f%%)",
            ticker, sigma, k, sigma, threshold, threshold * 100,
=======

        if len(daily_returns) < MIN_RETURNS_FOR_EWMA:
            raise ValueError(f"Too few return observations ({len(daily_returns)})")

        # RiskMetrics-style EWMA variance recursion
        returns_list = [float(x) for x in daily_returns.tolist()]
        ewma_var = returns_list[0] ** 2
        for r in returns_list[1:]:
            ewma_var = (ewma_lambda * ewma_var) + ((1.0 - ewma_lambda) * (r ** 2))

        sigma_daily = math.sqrt(max(ewma_var, 0.0))
        sigma_horizon = sigma_daily * math.sqrt(max(horizon_days, 1))
        threshold = k * sigma_horizon

        logger.info(
            "  %s @ %s: EWMA σ_1d=%.4f, σ_%dd=%.4f, threshold=%.4f (k=%.2f, λ=%.2f)",
            ticker,
            asof_date.strftime("%Y-%m-%d"),
            sigma_daily,
            horizon_days,
            sigma_horizon,
            threshold,
            k,
            ewma_lambda,
>>>>>>> fix/positive_bias
        )
        return threshold

    except Exception as exc:
        logger.warning(
<<<<<<< HEAD
            "Could not compute volatility threshold for %s (%s) — using fallback %.4f",
            ticker, exc, FALLBACK_NEUTRAL_THRESHOLD,
=======
            "Could not compute point-in-time EWMA threshold for %s @ %s (%s) — "
            "using fallback %.4f",
            ticker,
            asof_date.strftime("%Y-%m-%d"),
            exc,
            FALLBACK_NEUTRAL_THRESHOLD,
>>>>>>> fix/positive_bias
        )
        return FALLBACK_NEUTRAL_THRESHOLD


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
    neutral_threshold: float = FALLBACK_NEUTRAL_THRESHOLD,
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


<<<<<<< HEAD
def generate_test_cases(ticker_thresholds: dict[str, float]) -> list[dict]:
    """Generate all test cases with ground truth using per-company thresholds."""
=======
def generate_test_cases() -> list[dict]:
    """
    Generate all test cases with point-in-time, horizon-scaled ground truth labels.
    """
>>>>>>> fix/positive_bias
    test_cases = []
    case_id = 0

    for company in COMPANIES:
        ticker = company["ticker"]
        name = company["name"]
<<<<<<< HEAD
        neutral_threshold = ticker_thresholds[ticker]
=======
>>>>>>> fix/positive_bias

        for window in PREDICTION_WINDOWS:
            for period in MARKET_PERIODS:
                company_idx = COMPANIES.index(company)
                date_idx = company_idx % len(period["base_dates"])
                base_date_str = period["base_dates"][date_idx]

                start_dt = _parse_date(base_date_str)
                end_dt = start_dt + timedelta(days=window)

                start_date = start_dt.strftime("%d-%m-%Y")
                end_date = end_dt.strftime("%d-%m-%Y")

<<<<<<< HEAD
=======
                neutral_threshold = compute_point_in_time_ewma_threshold(
                    ticker=ticker,
                    asof_date=start_dt,
                    horizon_days=window,
                )

>>>>>>> fix/positive_bias
                case_id += 1
                test_id = f"{ticker}_W{window}_{period['label']}_{case_id:03d}"

                logger.info(
                    "Generating case %s: %s %s→%s (W=%d, threshold=%.4f)",
                    test_id, ticker, start_date, end_date, window, neutral_threshold,
                )

                try:
                    ground_truth = get_ground_truth(
<<<<<<< HEAD
                        ticker, start_date, end_date,
=======
                        ticker,
                        start_date,
                        end_date,
>>>>>>> fix/positive_bias
                        neutral_threshold=neutral_threshold,
                    )
                except Exception as exc:
                    logger.warning("Skipping %s: %s", test_id, exc)
                    continue

                test_case = {
                    "id": test_id,
                    "company_name": name,
                    "ticker": ticker,
<<<<<<< HEAD
=======
                    "sector": company["sector"],
>>>>>>> fix/positive_bias
                    "start_date": start_date,
                    "end_date": end_date,
                    "prediction_window_days": window,
                    "expected_backward_days": _compute_backward_days(window),
                    "market_period": period["label"],
                    "market_period_description": period["description"],
                    "neutral_threshold_used": round(neutral_threshold, 6),
<<<<<<< HEAD
=======
                    "threshold_method": "point_in_time_ewma_sigma_sqrt_window",
                    "ewma_lambda_used": EWMA_LAMBDA,
                    "volatility_k_used": VOLATILITY_K,
>>>>>>> fix/positive_bias
                    **ground_truth,
                }
                test_cases.append(test_case)

    return test_cases


def generate_and_save() -> None:
    """Generate the full test set and save to JSON."""
    logger.info("Starting test set generation...")

<<<<<<< HEAD
    # ── Step 1: compute per-company volatility thresholds ──
    logger.info(
        "Computing per-company volatility thresholds  (k=%.2f × σ_daily, lookback=%dd)...",
        VOLATILITY_K, VOLATILITY_LOOKBACK_DAYS,
    )
    ticker_thresholds: dict[str, float] = {}
    for company in COMPANIES:
        ticker = company["ticker"]
        ticker_thresholds[ticker] = compute_volatility_threshold(ticker)

    # ── Step 2: generate test cases ──
    test_cases = generate_test_cases(ticker_thresholds)
=======
    test_cases = generate_test_cases()
>>>>>>> fix/positive_bias

    label_counts: dict[str, int] = {}
    window_counts: dict[int, int] = {}
    company_counts: dict[str, int] = {}
<<<<<<< HEAD
    for case in test_cases:
        label = case["actual_label"]
        label_counts[label] = label_counts.get(label, 0) + 1
        w = case["prediction_window_days"]
        window_counts[w] = window_counts.get(w, 0) + 1
        t = case["ticker"]
        company_counts[t] = company_counts.get(t, 0) + 1
=======
    sector_counts: dict[str, int] = {}

    thresholds_by_window: dict[int, list[float]] = {w: [] for w in PREDICTION_WINDOWS}
    thresholds_by_ticker: dict[str, list[float]] = {c["ticker"]: [] for c in COMPANIES}

    for case in test_cases:
        label = case["actual_label"]
        label_counts[label] = label_counts.get(label, 0) + 1

        w = case["prediction_window_days"]
        window_counts[w] = window_counts.get(w, 0) + 1
        thresholds_by_window[w].append(case["neutral_threshold_used"])

        t = case["ticker"]
        company_counts[t] = company_counts.get(t, 0) + 1
        thresholds_by_ticker[t].append(case["neutral_threshold_used"])

        s = case["sector"]
        sector_counts[s] = sector_counts.get(s, 0) + 1
>>>>>>> fix/positive_bias

    output = {
        "metadata": {
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "description": "Evaluation test set for news-based stock prediction pipeline",
            "ground_truth_source": "yfinance",
<<<<<<< HEAD
            # Volatility-scaled threshold metadata
            "neutral_threshold_method": f"volatility_scaled_k{VOLATILITY_K}",
            "neutral_threshold_k": VOLATILITY_K,
            "neutral_threshold_lookback_days": VOLATILITY_LOOKBACK_DAYS,
            "neutral_thresholds_by_ticker": {
                t: round(v, 6) for t, v in ticker_thresholds.items()
            },
            "companies": [c["ticker"] for c in COMPANIES],
=======
            "neutral_threshold_method": "point_in_time_ewma_sigma_sqrt_window",
            "neutral_threshold_k": VOLATILITY_K,
            "ewma_lambda": EWMA_LAMBDA,
            "ewma_lookback_calendar_days": EWMA_LOOKBACK_CALENDAR_DAYS,
            "companies": [c["ticker"] for c in COMPANIES],
            "sectors": sorted(set(c["sector"] for c in COMPANIES)),
>>>>>>> fix/positive_bias
            "prediction_windows": PREDICTION_WINDOWS,
            "total_cases": len(test_cases),
            "label_distribution": label_counts,
            "cases_per_window": window_counts,
            "cases_per_company": company_counts,
<<<<<<< HEAD
=======
            "cases_per_sector": sector_counts,
            "avg_threshold_by_window": {
                str(w): round(sum(vals) / len(vals), 6) if vals else None
                for w, vals in thresholds_by_window.items()
            },
            "avg_threshold_by_ticker": {
                t: round(sum(vals) / len(vals), 6) if vals else None
                for t, vals in thresholds_by_ticker.items()
            },
>>>>>>> fix/positive_bias
        },
        "test_cases": test_cases,
    }

    PRED_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    logger.info("=" * 60)
    logger.info("Test set generated: %d cases", len(test_cases))
    logger.info("Label distribution: %s", label_counts)
    logger.info("Cases per window:   %s", window_counts)
    logger.info("Cases per company:  %s", company_counts)
<<<<<<< HEAD
    logger.info(
        "Thresholds used:    %s",
        {t: f"{v * 100:.2f}%" for t, v in ticker_thresholds.items()},
=======
    logger.info("Cases per sector:   %s", sector_counts)
    logger.info(
        "Avg threshold by window: %s",
        {
            w: f"{(sum(vals) / len(vals)) * 100:.2f}%"
            for w, vals in thresholds_by_window.items()
            if vals
        },
    )
    logger.info(
        "Avg threshold by ticker: %s",
        {
            t: f"{(sum(vals) / len(vals)) * 100:.2f}%"
            for t, vals in thresholds_by_ticker.items()
            if vals
        },
>>>>>>> fix/positive_bias
    )
    logger.info("Saved to: %s", OUTPUT_PATH)
    logger.info("=" * 60)


if __name__ == "__main__":
<<<<<<< HEAD
    generate_and_save()
=======
    generate_and_save() 
>>>>>>> fix/positive_bias
