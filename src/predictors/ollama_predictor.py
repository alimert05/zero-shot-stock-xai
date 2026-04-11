"""Ollama-based LLM sentiment prediction (Llama 3.1 8B, Mistral 7B)."""

from __future__ import annotations

import json
import logging
from typing import Any

from predictors.abstention import apply_abstention
from predictors.common import title_matches, build_input_text, print_summary, compute_effective_weight
from config import OLLAMA_SENTIMENT_MODEL, LLM_LABEL_CONFIDENCE, LLM_LABEL_RESIDUAL

logger = logging.getLogger(__name__)

OLLAMA_PROMPT_TEMPLATE = (
    "Instruction: What is the sentiment of this news? "
    "Please choose an answer from {{negative/neutral/positive}}\n"
    "Input: {text}\n"
    "Answer: "
)


# Classification

def _classify_sentiment(text: str) -> dict[str, float]:
    """Classify a single text via Ollama."""
    import ollama

    prompt = OLLAMA_PROMPT_TEMPLATE.format(text=text)

    response = ollama.chat(
        model=OLLAMA_SENTIMENT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.0,
            "num_predict": 10,
        },
    )

    answer = response["message"]["content"].strip().lower()

    if "positive" in answer:
        label = "positive"
    elif "negative" in answer:
        label = "negative"
    elif "neutral" in answer:
        label = "neutral"
    else:
        logger.warning(
            "Ollama (%s) unexpected answer: '%s', defaulting to neutral",
            OLLAMA_SENTIMENT_MODEL, answer,
        )
        label = "neutral"

    scores = {"positive": LLM_LABEL_RESIDUAL, "negative": LLM_LABEL_RESIDUAL, "neutral": LLM_LABEL_RESIDUAL}
    scores[label] = LLM_LABEL_CONFIDENCE
    return scores


# Pipeline-level prediction

def predict_sentiment(
    articles_json_path: str,
    company_name: str | None = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    """Classify articles via Ollama LLM and aggregate into a sentiment prediction."""
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
        "Running Ollama (%s) sentiment on %d articles (company=%s, ticker=%s)",
        OLLAMA_SENTIMENT_MODEL, len(articles), company_name, ticker,
    )

    weighted_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    total_weight = 0.0
    article_sentiments: list[dict] = []
    article_weights: list[float] = []

    for i, article in enumerate(articles):
        title = article.get("title", "")
        base_weight = float(article.get("final_weight", 1.0))
        coverage_count = int(article.get("coverage_count", 1))
        content_raw = article.get("content") or ""
        is_headline_only = not content_raw.strip()

        include_title = title_matches(title, company_name, ticker)
        text = build_input_text(article, include_title=include_title, company_name=company_name)

        if not text:
            logger.debug("Skipping article (no title and no content): %s", title[:80])
            continue

        scores = _classify_sentiment(text)

        effective_weight, coverage_boost, headline_discount = compute_effective_weight(
            base_weight, coverage_count, is_headline_only,
        )

        for label in weighted_scores:
            weighted_scores[label] += scores[label] * effective_weight
        total_weight += effective_weight
        article_weights.append(effective_weight)

        if include_title:
            source_label = "headline+content"
        elif content_raw.strip():
            source_label = "content-only"
        else:
            source_label = "title-fallback"

        article_sentiments.append({
            "title": title,
            "base_weight": base_weight,
            "final_weight": round(effective_weight, 4),
            "coverage_boost": round(coverage_boost, 4),
            "headline_discount": round(headline_discount, 4),
            "input_source": source_label,
            "raw_scores": scores,
            "weighted_scores": {
                k: round(v * effective_weight, 4) for k, v in scores.items()
            },
        })

        logger.info(
            "[%d/%d] (%s) %s -> pos=%.4f neg=%.4f neu=%.4f (w=%.3f)",
            i + 1, len(articles), source_label, title[:50],
            scores["positive"], scores["negative"], scores["neutral"],
            effective_weight,
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

    print_summary(result, model_label=OLLAMA_SENTIMENT_MODEL)
    return result
