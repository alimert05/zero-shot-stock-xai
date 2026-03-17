from __future__ import annotations

import json
import logging
from typing import Any

from predictor.abstention import apply_abstention
from predictor.common import title_matches, build_input_text, print_summary
from config import SENTIMENT_DEVICE

logger = logging.getLogger(__name__)

_sentiment_pipeline = None


def _get_sentiment_pipeline():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline

            logger.info("Loading FinBERT sentiment model...")
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=SENTIMENT_DEVICE,
            )
            logger.info("FinBERT model loaded successfully")
        except Exception as exc:
            logger.error("Failed to load FinBERT model: %s", exc)
            raise
    return _sentiment_pipeline


FINBERT_LABEL_MAP = {
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
}


def _classify_sentiment(text: str) -> dict[str, float]:
    pipe = _get_sentiment_pipeline()

    results = pipe(text, top_k=None, truncation=True, max_length=512)

    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for item in results:
        label = FINBERT_LABEL_MAP.get(item["label"], item["label"])
        scores[label] = item["score"]

    return scores


def predict_sentiment(
    articles_json_path: str,
    company_name: str | None = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    with open(articles_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    query = data.get("query", "")
    json_ticker = data.get("ticker")

    if not company_name:
        company_name = query
    if not ticker:
        ticker = json_ticker
    if not company_name:
        raise ValueError("company_name must be provided or present in articles.json query field")

    logger.info(
        "Running FinBERT sentiment on %d articles (company=%s, ticker=%s)",
        len(articles), company_name, ticker,
    )

    weighted_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    total_weight = 0.0
    article_sentiments: list[dict] = []
    article_weights: list[float] = []

    for i, article in enumerate(articles):
        title = article.get("title", "")
        final_weight = float(article.get("final_weight", 1.0))

        include_title = title_matches(title, company_name, ticker)
        text = build_input_text(article, include_title=include_title, company_name=company_name)

        if not text:
            logger.debug("Skipping article (no title and no content): %s", title[:80])
            continue

        scores = _classify_sentiment(text)

        for label in weighted_scores:
            weighted_scores[label] += scores[label] * final_weight
        total_weight += final_weight
        article_weights.append(final_weight)

        content_raw = article.get("content") or ""
        if include_title:
            source_label = "headline+content"
        elif content_raw.strip():
            source_label = "content-only"
        else:
            source_label = "title-fallback"
        article_sentiments.append({
            "title": title,
            "final_weight": final_weight,
            "input_source": source_label,
            "raw_scores": scores,
            "weighted_scores": {
                k: round(v * final_weight, 4) for k, v in scores.items()
            },
        })

        logger.info(
            "[%d/%d] (%s) %s -> pos=%.4f neg=%.4f neu=%.4f (w=%.3f)",
            i + 1, len(articles), source_label, title[:50],
            scores["positive"], scores["negative"], scores["neutral"],
            final_weight,
        )

    if total_weight > 0:
        normalized_scores = {
            k: round(v / total_weight, 4) for k, v in weighted_scores.items()
        }
    else:
        normalized_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    abstention = apply_abstention(normalized_scores, article_weights)
    final_label = abstention["final_label"]

    threshold_gap = abstention["abstention_test"].get("threshold_gap")
    if threshold_gap:
        final_confidence = threshold_gap["score"]
    else:
        final_confidence = normalized_scores[final_label]

    result = {
        "query": query,
        "company_name": company_name,
        "ticker": ticker,
        "articles_analyzed": len(article_sentiments),
        "articles_total": len(articles),
        "total_weight": round(total_weight, 4),
        "weighted_scores": {
            k: round(v, 4) for k, v in weighted_scores.items()
        },
        "normalized_scores": normalized_scores,
        "final_label": final_label,
        "final_confidence": final_confidence,
        "article_details": article_sentiments,
        "abstention_test": abstention["abstention_test"],
    }

    logger.info(
        "Sentiment prediction complete: label=%s confidence=%.4f (%d articles analyzed)",
        final_label, normalized_scores[final_label], len(article_sentiments),
    )

    return result


def run_sentiment_prediction(
    articles_json_path: str,
    output_path: str | None = None,
    company_name: str | None = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    result = predict_sentiment(
        articles_json_path=articles_json_path,
        company_name=company_name,
        ticker=ticker,
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Sentiment result saved to %s", output_path)

    print_summary(result, model_label="FinBERT")
    return result
