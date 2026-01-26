from .backtester import get_real_label_yfinance

label, meta, warn = get_real_label_yfinance("AAPL", "2023-01-01", "2023-01-10")
print(label, meta["pct_change"], warn)
