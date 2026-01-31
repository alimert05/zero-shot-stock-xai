from .backtester import get_real_label_yfinance

label, meta, warn = get_real_label_yfinance("LLY", "2025-01-01", "2025-02-01")
print(label, meta["pct_change"], warn)
