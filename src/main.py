"""Main entry point for the sentiment prediction pipeline.

Fetches news articles, runs the configured sentiment model, and optionally
generates an XAI explainability report.
"""

from __future__ import annotations

import logging

from news_retriever.retriever import Fetcher
from predictors.finbert import run_sentiment_prediction as run_finbert
from predictors.fingpt import run_sentiment_prediction as run_fingpt
from predictors.zero_shot import run_sentiment_prediction as run_zero_shot
from predictors.ollama_predictor import run_sentiment_prediction as run_ollama
from config import JSON_PATH, SENTIMENT_MODEL, FINBERT_PREDS, FINGPT_PREDS, ZEROSHOT_PREDS, OLLAMA_PREDS, MODEL_NAME, XAI_ENABLED, XAI_OUTPUT_PATH
from xai import run_xai

logger = logging.getLogger(__name__)


def main() -> None:
    """Fetch articles, run sentiment prediction, and generate XAI output."""
    fetcher = Fetcher()
    has_articles = fetcher.run_fetcher()
    if not has_articles:
        logger.warning("No articles fetched for the selected date window. Stopping pipeline safely.")
        return

    prediction_result = None

    if SENTIMENT_MODEL == "fingpt":
        prediction_result = run_fingpt(articles_json_path=str(JSON_PATH), output_path=str(FINGPT_PREDS))
    elif SENTIMENT_MODEL == "ProsusAI/finbert":
        prediction_result = run_finbert(articles_json_path=str(JSON_PATH), output_path=str(FINBERT_PREDS))
    elif SENTIMENT_MODEL == "zero-shot":
        if MODEL_NAME == "facebook/bart-large-mnli":
            path = "BART Large MNLI"
        if MODEL_NAME == "roberta-large-mnli":
            path = "RoBERTa Large MNLI"
        if MODEL_NAME == "microsoft/deberta-large-mnli":
            path = "DeBERTa Large MNLI"
        prediction_result = run_zero_shot(articles_json_path=str(JSON_PATH), output_path=str(ZEROSHOT_PREDS))
    elif SENTIMENT_MODEL in ("ollama-llama3", "ollama-mistral"):
        prediction_result = run_ollama(articles_json_path=str(JSON_PATH), output_path=str(OLLAMA_PREDS))

    if XAI_ENABLED and prediction_result is not None:
        run_xai(
            prediction_result=prediction_result,
            articles_json_path=str(JSON_PATH),
            output_path=str(XAI_OUTPUT_PATH),
        )

    # run_backtest()

if __name__ == "__main__":
    main()
