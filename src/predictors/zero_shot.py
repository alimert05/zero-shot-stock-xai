"""
zero_shot.py - Zero-shot NLI sentiment predictor.

Uses Natural Language Inference to classify financial news articles as
positive, negative, or neutral without task-specific fine-tuning. 
Articles are scored individually and then aggregated via weighted averaging.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from config import (
    SENTIMENT_DEVICE,
    MODEL_NAME,
)
from predictors.abstention import apply_abstention
from predictors.common import title_matches, build_input_text, print_summary, compute_effective_weight

logger = logging.getLogger(__name__)

# Model configuration

_nli_pipeline = None

MODEL_DISPLAY_NAMES = {
    "facebook/bart-large-mnli": "BART Large MNLI",
    "roberta-large-mnli": "RoBERTa Large MNLI",
    "microsoft/deberta-large-mnli": "DeBERTa Large MNLI",
    "MoritzLaurer/deberta-v3-base-zeroshot-v2.0-c": "DeBERTa v3 Base Zeroshot v2.0",
}
_display_model = MODEL_DISPLAY_NAMES.get(MODEL_NAME, MODEL_NAME)


# Model loading


def _get_nli_pipeline():
    """Lazy-load and cache the zero-shot NLI classification pipeline."""
    global _nli_pipeline
    if _nli_pipeline is None:
        try:
            from transformers import pipeline
            
            logger.info("Loading %s zero-shot classifier...", _display_model)
            _nli_pipeline = pipeline(
                "zero-shot-classification",
                model=MODEL_NAME,
                device=SENTIMENT_DEVICE,
            )
            logger.info("%s model loaded successfully", _display_model)
        except Exception as exc:
            logger.error("Failed to load %s model: %s", _display_model, exc)
            raise
    return _nli_pipeline


# Article filtering (delegated to predictors.common)


_CLASS_TO_LABEL = {
    "positive": "bullish",
    "negative": "bearish",
    "neutral": "neutral",
}
_CANDIDATE_LABELS = list(_CLASS_TO_LABEL.values())
_LABEL_TO_CLASS = {v.lower().strip(): k for k, v in _CLASS_TO_LABEL.items()}
_HYPOTHESIS_TEMPLATE = "The implication of this statement is {}."



def _batch_classify_sentiment(
    texts: list[str],
    batch_size: int = 32,
) -> list[dict[str, float]]:
    """Classify multiple texts in a single batched GPU call.

    Instead of N individual forward passes (one per article), this creates
    N × 3 NLI pairs and processes them in chunks of *batch_size*, dramatically
    reducing GPU kernel-launch overhead and Python-loop latency.
    """
    pipe = _get_nli_pipeline()

    results = pipe(
        texts,
        candidate_labels=_CANDIDATE_LABELS,
        hypothesis_template=_HYPOTHESIS_TEMPLATE,
        multi_label=True,
        batch_size=batch_size,
    )

    # Single text -> pipeline returns a dict instead of list
    if isinstance(results, dict):
        results = [results]

    all_scores: list[dict[str, float]] = []
    for result in results:
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for label, score in zip(result["labels"], result["scores"]):
            cls = _LABEL_TO_CLASS.get(label.lower().strip())
            if cls:
                scores[cls] = float(score)
        all_scores.append(scores)

    return all_scores


def predict_sentiment(
    articles_json_path: str,
    company_name: str | None = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    """Score all articles and aggregate into a three-way sentiment prediction."""
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
        "Running %s sentiment on %d articles (company=%s, ticker=%s)",
        _display_model, len(articles), company_name, ticker
    )

    weighted_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    total_weight = 0.0
    article_sentiments: list[dict] = []
    article_weights: list[float] = []

    # Phase 1: Collect texts and metadata (CPU only)
    batch_texts: list[str] = []
    batch_meta: list[dict] = []  # parallel list: article index, weight, source_label

    for i, article in enumerate(articles):
        title = article.get("title", "")
        final_weight = float(article.get("final_weight", 1.0))

        include_title = title_matches(title, company_name, ticker)
        text = build_input_text(
            article, include_title=include_title,
            company_name=company_name, prefix="News about",
        )

        if not text:
            logger.debug("Skipping article (no title and no content): %s", title[:80])
            continue

        content_raw = article.get("content") or ""
        if include_title:
            source_label = "headline+content"
        elif content_raw.strip():
            source_label = "content-only"
        else:
            source_label = "title-fallback"

        coverage_count = int(article.get("coverage_count", 1))
        is_headline_only = not content_raw.strip()

        batch_texts.append(text)
        batch_meta.append({
            "idx": i,
            "title": title,
            "final_weight": final_weight,
            "source_label": source_label,
            "coverage_count": coverage_count,
            "is_headline_only": is_headline_only,
        })

    # Phase 2: Batch GPU inference (single call for all articles)
    if batch_texts:
        all_scores = _batch_classify_sentiment(batch_texts, batch_size=32)
    else:
        all_scores = []

    # Phase 3: Accumulate weighted scores
    for meta, raw in zip(batch_meta, all_scores):
        base_weight = meta["final_weight"]
        effective_weight, coverage_boost, headline_discount = compute_effective_weight(
            base_weight, meta["coverage_count"], meta["is_headline_only"],
        )

        for label in weighted_scores:
            weighted_scores[label] += raw[label] * effective_weight
        total_weight += effective_weight
        article_weights.append(effective_weight)

        detail = {
            "title": meta["title"],
            "base_weight": base_weight,
            "final_weight": round(effective_weight, 4),
            "coverage_boost": round(coverage_boost, 4),
            "headline_discount": round(headline_discount, 4),
            "input_source": meta["source_label"],
            "raw_scores": raw,
            "weighted_scores": {
                k: round(v * effective_weight, 4) for k, v in raw.items()
            },
        }
        article_sentiments.append(detail)

        logger.info(
            "[%d/%d] (%s) %s -> pos=%.4f neg=%.4f neu=%.4f (w=%.3f->%.3f, cov=%.2f, hdl=%.2f)",
            meta["idx"] + 1, len(articles), meta["source_label"],
            meta["title"][:50],
            raw["positive"], raw["negative"], raw["neutral"],
            base_weight, effective_weight, coverage_boost, headline_discount,
        )

    if total_weight > 0:
        normalized_scores = {
            k: round(v / total_weight, 4) for k, v in weighted_scores.items()
        }
    else:
        normalized_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    abstention = apply_abstention(normalized_scores, article_weights)
    final_label = abstention["final_label"]

    # For threshold-triggered neutrals, report the highest directional score
    # as confidence (not the tiny raw neutral score which is misleading).
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
        "enhanced_weighting": True,
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
    """Run prediction and save results to a JSON file."""
    result = predict_sentiment(
        articles_json_path=articles_json_path,
        company_name=company_name,
        ticker=ticker,
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Sentiment result saved to %s", output_path)

    print_summary(result, model_label=_display_model)
    return result