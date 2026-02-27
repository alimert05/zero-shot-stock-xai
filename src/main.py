import atexit
from fetcher.fetcher import Fetcher
from predictor.finbert import run_sentiment_prediction as run_finbert
from predictor.fingpt import run_sentiment_prediction as run_fingpt
from predictor.zero_shot import run_sentiment_prediction as run_zero_shot
from backtest.backtester import run_backtest
from config import JSON_PATH, SENTIMENT_MODEL, FINBERT_PREDS, FINGPT_PREDS, ZEROSHOT_PREDS, MODEL_NAME, XAI_ENABLED, XAI_OUTPUT_PATH
from xai import run_xai

def main() -> None:

    # fetcher = Fetcher()
    # fetcher.run_fetcher()

    # prediction_result = None

    # if SENTIMENT_MODEL == "fingpt":
    #     prediction_result = run_fingpt(articles_json_path=str(JSON_PATH), output_path=str(FINGPT_PREDS))
    # elif SENTIMENT_MODEL == "ProsusAI/finbert":
    #     prediction_result = run_finbert(articles_json_path=str(JSON_PATH), output_path=str(FINBERT_PREDS))
    # elif SENTIMENT_MODEL == "zero-shot":
    #     if MODEL_NAME == "facebook/bart-large-mnli":
    #         path = "BART Large MNLI"
    #     if MODEL_NAME == "roberta-large-mnli":
    #         path = "RoBERTa Large MNLI"
    #     if MODEL_NAME == "microsoft/deberta-large-mnli":
    #         path = "DeBERTa Large MNLI"
    #     prediction_result = run_zero_shot(articles_json_path=str(JSON_PATH), output_path=str(ZEROSHOT_PREDS))

    # if XAI_ENABLED and prediction_result is not None:
    #     run_xai(
    #         prediction_result=prediction_result,
    #         articles_json_path=str(JSON_PATH),
    #         output_path=str(XAI_OUTPUT_PATH),
    #     )

    run_backtest()

if __name__ == "__main__":
    main()
