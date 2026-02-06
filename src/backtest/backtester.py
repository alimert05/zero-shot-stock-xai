from datetime import datetime, timedelta
import json
import yfinance as yf
from config import PRED_JSON_PATH, JSON_PATH, SENTIMENT_JSON_PATH


def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%d-%m-%Y")


def _next_open_day_close(ticker: str, day: datetime, max_lookahead_days: int = 10) -> tuple[datetime, float]:
    df = yf.download(
        ticker,
        start=day.strftime("%Y-%m-%d"),
        end=(day + timedelta(days=max_lookahead_days + 1)).strftime("%Y-%m-%d"),
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

    raise ValueError(f"No open day found for {ticker} within {max_lookahead_days} days after {day.date()}")


def get_real_label_yfinance(ticker: str, start_date: str, end_date: str, neutral_threshold: float = 0.003, max_lookahead_days: int = 10,) -> tuple[str, dict, str | None]:
    s0 = _parse_date(start_date)
    e0 = _parse_date(end_date)
    window_days = (e0.date() - s0.date()).days

    s_dt, s_px = _next_open_day_close(ticker, s0, max_lookahead_days=max_lookahead_days)
    e_dt, e_px = _next_open_day_close(ticker, e0, max_lookahead_days=max_lookahead_days)

    pct = (e_px - s_px) / s_px if s_px else 0.0

    if pct > neutral_threshold:
        label = "positive"
    elif pct < -neutral_threshold:
        label = "negative"
    else:
        label = "neutral"

    warn = None
    shifted = (s_dt.date() != s0.date()) or (e_dt.date() != e0.date())
    if window_days <= 1 and shifted:
        warn = (
            f"Warning: 1-day window requested ({start_date}->{end_date}) but market was closed, "
            f"shifted to ({s_dt.date()}->{e_dt.date()})."
        )

    meta = {
        "ticker": ticker,
        "requested_start": start_date,
        "requested_end": end_date,
        "used_start": s_dt.strftime("%Y-%m-%d"),
        "used_end": e_dt.strftime("%Y-%m-%d"),
        "start_close": s_px,
        "end_close": e_px,
        "pct_change": pct,
        "neutral_threshold": neutral_threshold,
    }
    return label, meta, warn


def store_predictions_jsonl(filepath: str, record: dict) -> None:
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def evaluate_one(predicted_label: str, actual_label: str) -> dict:
    return {
        "predicted": predicted_label,
        "actual": actual_label,
        "correct": predicted_label == actual_label,
    }

with open(JSON_PATH, 'r', encoding="utf-8") as f:
    query_data = json.load(f)


company_name = query_data["query"]
ticker = query_data["ticker"] 
start_date = query_data["start_date"]
end_date = query_data["end_date"]
 

actual_label, meta, warn = get_real_label_yfinance(ticker, start_date, end_date)
if warn:
    print(warn)

with open(SENTIMENT_JSON_PATH, 'r', encoding="utf-8") as f:
    sentiment_data = json.load(f)

predicted_label = sentiment_data["final_label"]

pred_record = {
    "company_name": company_name,
    "ticker": ticker,
    "start_date": start_date,
    "end_date": end_date,
    "predicted_label": predicted_label,
}
store_predictions_jsonl(PRED_JSON_PATH, pred_record)

def run_backtest():

    report = evaluate_one(predicted_label, actual_label)
    print("─" * 35)
    print("Actual:", actual_label.capitalize())
    print("─" * 35)
    print("Pred:", predicted_label.capitalize())
    print("─" * 35)
    print("Correct:", report["correct"])
    print("─" * 35)
    print(f"Change: {round(float(meta["pct_change"]), 5) * 100}%", warn if warn is not None else "")
    print("─" * 35)

