from __future__ import annotations

import re
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)

_deberta_classifier = None


def _get_deberta_classifier():
    global _deberta_classifier
    if _deberta_classifier is None:
        try:
            from transformers import pipeline
            from config import SENTIMENT_DEVICE, NOISE_REDUCTION_MODEL

            logger.info("Loading DeBERTa classifier for noise reduction...")
            _deberta_classifier = pipeline(
                "zero-shot-classification",
                model=NOISE_REDUCTION_MODEL,
                device=SENTIMENT_DEVICE,
                token=False
            )
            logger.info("DeBERTa classifier loaded successfully")
        except Exception as exc:
            logger.error("Failed to load DeBERTa classifier: %s", exc)
            raise
    return _deberta_classifier

def _split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]

def _score_sentence_relevance(
    sentences: List[str],
    company_name: str,
    ticker: Optional[str], 
) -> List[tuple[str, float]]:

    if not sentences:
        return []

    classifier = _get_deberta_classifier()

    ticker_part = f" ({ticker})" if ticker else ""
    labels = ["relevant", "irrelevant"]
    hypothesis_template = (
        f"This sentence is {{}} to {company_name}{ticker_part}, including "
        f"products/brands, financial performance (revenue, earnings, losses, margins), "
        f"guidance, operations, lawsuits, regulation, risks, and major announcements."
    )

    results = classifier(
        sentences,
        candidate_labels=labels,
        hypothesis_template=hypothesis_template,
        batch_size=16,
        multi_label=True
    )

    if isinstance(results, dict):
        results = [results]

    scored: List[tuple[str, float]] = []
    for sentence, result in zip(sentences, results):
        relevance_score = 0.0
        for label, score in zip(result["labels"], result["scores"]):
            if label == "relevant":
                relevance_score = score
                break
        scored.append((sentence, relevance_score))

    return scored

def _filter_relevant_sentences(
    scored_sentences: List[tuple[str, float]],
    threshold: float = 0.5,
) -> List[str]:
    return [sent for sent, score in scored_sentences if score >= threshold]

def reduce_content_noise(
    content: str,
    company_name: str,
    ticker: Optional[str] = None,
    relevance_threshold: float = 0.5,
) -> tuple[Optional[str], dict]:

    if not content:
        return None, {"error": "no_content"}

    sentences = _split_into_sentences(content)

    if not sentences:
        return None, {"error": "no_sentences"}

    scored = _score_sentence_relevance(sentences, company_name, ticker)
    relevant = _filter_relevant_sentences(scored, relevance_threshold)

    total_sentences = len(sentences)
    relevant_count = len(relevant)
    relevance_ratio = relevant_count / total_sentences if total_sentences > 0 else 0

    metadata = {
        "total_sentences": total_sentences,
        "relevant_sentences": relevant_count,
        "relevance_ratio": round(relevance_ratio, 4),
        "condensed": False,
    }

    if relevant_count == 0:
        logger.debug("No relevant sentences for %s after filtering", company_name)
        return None, {**metadata, "filter_action": "no_match"}

    filtered_text = " ".join(relevant)

    if relevant_count == total_sentences:
        return filtered_text, {**metadata, "filter_action": "no_change"}
    
    return filtered_text, {**metadata, "filter_action": "filtered"}


def clean_articles_content(
    articles: List[dict],
    company_name: str,
    ticker: Optional[str] = None,
    relevance_threshold: float = 0.5,
) -> List[dict]:

    if not articles:
        return []

    logger.info(
        "Cleaning article content for %s (ticker: %s) — DeBERTa-only pipeline",
        company_name,
        ticker or "None",
    )

    stats = {
        "total_articles": 0,
        "articles_with_content": 0,
        "articles_filtered": 0,
        "articles_no_change": 0,
        "articles_no_content_after": 0,
        "total_sentences_before": 0,
        "total_sentences_after": 0,
    }

    for article in articles:
        content = article.get("content")

        stats["total_articles"] += 1

        if not content:
            continue

        stats["articles_with_content"] += 1

        cleaned_content, metadata = reduce_content_noise(
            content,
            company_name,
            ticker,
            relevance_threshold,
        )

        article["content"] = cleaned_content
        article["content_stats"] = metadata

        stats["total_sentences_before"] += metadata.get("total_sentences", 0)
        stats["total_sentences_after"] += metadata.get("relevant_sentences", 0)

        if metadata.get("filter_action") == "filtered":
            stats["articles_filtered"] += 1
        elif metadata.get("filter_action") == "no_change":
            stats["articles_no_change"] += 1

        if cleaned_content is None:
            stats["articles_no_content_after"] += 1

    logger.info(
        "Content cleaning complete:\n"
        "  Total articles: %d\n"
        "  Articles with content: %d\n"
        "  Articles filtered (DeBERTa only): %d\n"
        "  Articles with no content after: %d\n"
        "  Sentences: %d → %d",
        stats["total_articles"],
        stats["articles_with_content"],
        stats["articles_filtered"],
        stats["articles_no_content_after"],
        stats["total_sentences_before"],
        stats["total_sentences_after"],
    )

    if stats["total_sentences_before"] > 0:
        reduction = (1 - stats["total_sentences_after"] / stats["total_sentences_before"]) * 100
        logger.info("  Noise reduction: %.1f%%", reduction)

    return articles