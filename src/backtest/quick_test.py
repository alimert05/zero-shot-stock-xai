from .backtester import get_real_label_yfinance

label, meta, warn = get_real_label_yfinance("AAPL", "01-01-2025", "01-02-2025")
print(label, meta["pct_change"], warn if warn is not None else "")
