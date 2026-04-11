"""Impact horizon weighting functions.

Computes per-article weights based on event-family horizon priors
(Gaussian decay) and combines them with recency weights via geometric mean.
"""

from __future__ import annotations

import logging
import math

from .event_classifier import (
    EVENT_FAMILY_LABELS,
    _get_classifier,
    _build_classification_text,
    _map_classifier_result,
    _fallback_horizon_result,
)

logger = logging.getLogger(__name__)


def _clamp(value: float, low: float, high: float) -> float:
    """Constrain value to the range [low, high]."""
    return max(low, min(high, value))


def _single_horizon_weight(
    days_ago: int,
    impact_horizon_days: int,
    prediction_window_days: int,
    min_weight: float = 0.05,
) -> float:
    """
    Gaussian decay weight for a single horizon, 
    extracted so primary and secondary horizons can be combined.
    """
    W = max(int(prediction_window_days), 1)
    impact_day = int(impact_horizon_days) - int(days_ago)
    mu = W / 2.0
    sigma = max(W / 2.0, 3.0)
    raw = math.exp(-((impact_day - mu) ** 2) / (2.0 * sigma ** 2))
    return max(raw, min_weight)


def calculate_impact_horizon_weight(
    days_ago: int,
    prediction_window_days: int,
    impact_horizon_days: int | None = None,
    primary_horizon_days: int | None = None,
    secondary_horizon_days: int | None = None,
    confidence: float = 1.0,
    min_weight: float = 0.05,
) -> float:
    """Compute a Gaussian horizon weight based on timing alignment with the prediction window."""
    if primary_horizon_days is None:
        if impact_horizon_days is None:
            raise ValueError(
                "Either primary_horizon_days or impact_horizon_days must be provided"
            )
        primary_horizon_days = impact_horizon_days

    primary_weight = _single_horizon_weight(
        days_ago=days_ago,
        impact_horizon_days=primary_horizon_days,
        prediction_window_days=prediction_window_days,
        min_weight=min_weight,
    )

    if secondary_horizon_days is None:
        return primary_weight

    secondary_weight = _single_horizon_weight(
        days_ago=days_ago,
        impact_horizon_days=secondary_horizon_days,
        prediction_window_days=prediction_window_days,
        min_weight=min_weight,
    )

    confidence = _clamp(float(confidence), 0.0, 1.0)
    # Blend primary and secondary horizons based on classifier confidence.
    # High confidence (1.0) -> 85% primary, low confidence (0.0) -> 60% primary.
    # Primary always dominates since even an uncertain classification is more
    # informative than a uniform prior.
    primary_mix = 0.60 + (0.25 * confidence)   # 0.60 .. 0.85
    secondary_mix = 1.0 - primary_mix

    combined = (primary_mix * primary_weight) + (secondary_mix * secondary_weight)
    return max(combined, min_weight)


def calculate_combined_weight(
    recency_weight: float,
    impact_horizon_weight: float,
) -> float:
    """Combine recency and horizon weights via geometric mean."""
    recency_weight = max(float(recency_weight), 0.0)
    impact_horizon_weight = max(float(impact_horizon_weight), 0.0)
    return math.sqrt(recency_weight * impact_horizon_weight)


def _apply_horizon_to_article(
    article: dict,
    horizon_result: dict,
    prediction_window_days: int,
) -> None:
    """Attach impact horizon data, weights, and final_weight to an article dict."""
    raw_days_ago = article.get("days_ago", 0)
    raw_recency_weight = article.get("recency_weight", 1.0)

    try:
        days_ago = int(raw_days_ago) if raw_days_ago is not None else 0
    except (TypeError, ValueError):
        days_ago = 0

    try:
        recency_weight = float(raw_recency_weight)
    except (TypeError, ValueError):
        recency_weight = 1.0

    horizon_weight = calculate_impact_horizon_weight(
        days_ago=days_ago,
        prediction_window_days=prediction_window_days,
        primary_horizon_days=horizon_result["primary_horizon_days"],
        secondary_horizon_days=horizon_result["secondary_horizon_days"],
        confidence=horizon_result["confidence"],
    )

    final_weight = calculate_combined_weight(
        recency_weight=recency_weight,
        impact_horizon_weight=horizon_weight,
    )

    article["impact_horizon"] = {
        "event_type": horizon_result["event_type"],
        "label": horizon_result["label"],
        "category": horizon_result["category"],
        "horizon_days": horizon_result["horizon_days"],
        "confidence": round(horizon_result["confidence"], 4),
        "event_family": horizon_result["event_family"],
        "primary_horizon_label": horizon_result["primary_horizon_label"],
        "primary_horizon_days": horizon_result["primary_horizon_days"],
        "primary_category": horizon_result["primary_category"],
        "secondary_horizon_label": horizon_result["secondary_horizon_label"],
        "secondary_horizon_days": horizon_result["secondary_horizon_days"],
        "secondary_category": horizon_result["secondary_category"],
        "alternative_event_family": horizon_result["alternative_event_family"],
        "alternative_confidence": (
            round(horizon_result["alternative_confidence"], 4)
            if horizon_result["alternative_confidence"] is not None
            else None
        ),
    }
    article["impact_horizon_weight"] = round(horizon_weight, 4)
    article["final_weight"] = round(final_weight, 4)


def add_impact_horizon_data(
    articles: list[dict],
    prediction_window_days: int,
    batch_size: int = 32,
) -> None:
    """Classify articles and attach impact horizon data with batched GPU inference.

    Instead of N individual GPU calls, collects all classifiable texts and
    runs a single batched pipeline call, then post-processes per article.
    """
    logger.info(
        "Adding impact horizon data to %d articles (prediction window: %d days)",
        len(articles),
        prediction_window_days,
    )

    # Phase 1: Partition articles (CPU only)
    gpu_indices: list[int] = []
    gpu_texts: list[str] = []

    for i, article in enumerate(articles):
        title = article.get("title", "")

        if not title:
            raw_recency_weight = article.get("recency_weight", 1.0)
            try:
                recency_weight = float(raw_recency_weight)
            except (TypeError, ValueError):
                recency_weight = 1.0
            # No title to classify — skip event classification,
            # fall back to recency weight only.
            article["impact_horizon"] = None
            article["impact_horizon_weight"] = 1.0
            article["final_weight"] = recency_weight
        else:
            content = article.get("content", "")
            text = _build_classification_text(title, content)
            gpu_indices.append(i)
            gpu_texts.append(text)

    if not gpu_texts:
        logger.info("No articles with titles - skipping impact horizon classification")
        return

    # Phase 2: Batch GPU inference
    classifier = _get_classifier()
    try:
        batch_results = classifier(
            gpu_texts,
            candidate_labels=EVENT_FAMILY_LABELS,
            hypothesis_template="The primary firm-specific event in this news article is {}.",
            multi_label=False,
            batch_size=batch_size,
        )
    except Exception as exc:
        logger.error("Batch impact horizon classification failed: %s - using fallback", exc)
        fallback = _fallback_horizon_result()
        for idx in gpu_indices:
            _apply_horizon_to_article(articles[idx], fallback, prediction_window_days)
        return

    # Single result -> wrap in list
    if isinstance(batch_results, dict):
        batch_results = [batch_results]

    # Phase 3: Post-process each result
    for idx, raw_result in zip(gpu_indices, batch_results):
        try:
            horizon_result = _map_classifier_result(raw_result)
        except Exception as exc:
            logger.warning("Impact horizon mapping failed for article %d: %s", idx, exc)
            horizon_result = _fallback_horizon_result()

        _apply_horizon_to_article(articles[idx], horizon_result, prediction_window_days)

    logger.info(
        "Impact horizon classification complete (%d articles, batch_size=%d)",
        len(gpu_texts), batch_size,
    )
