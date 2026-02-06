import atexit
from fetcher.fetcher import Fetcher
from predictor.finbert import run_sentiment_prediction
from backtest.backtester import run_backtest
from config import JSON_PATH, SENTIMENT_OUTPUT_PATH

def main() -> None:

    fetcher = Fetcher()
    fetcher.run_fetcher()

    run_sentiment_prediction(
        articles_json_path=str(JSON_PATH),
        output_path=str(SENTIMENT_OUTPUT_PATH),
    )

    # Not working well yet
    run_backtest()

if __name__ == "__main__":
    main()
