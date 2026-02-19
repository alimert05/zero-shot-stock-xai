from __future__ import annotations

import logging
from typing import Any

from .utils import herfindahl_index, safe_round, get_dominant_label

logger = logging.getLogger(__name__)


def _compute_contribution_share(
    article: dict[str, Any],
    total_weight: float,
) -> dict[str, float]:
    weighted_scores = article.get("weighted_scores", {})
    if total_weight == 0:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    return {
        label: safe_round(score / total_weight)
        for label, score in weighted_scores.items()
    }


def _run_counterfactual(
    article: dict[str, Any],
    total_weighted: dict[str, float],
    total_weight: float,
    current_label: str,
) -> dict[str, Any]:
    article_weight = article.get("final_weight", 0.0)
    article_weighted_scores = article.get("weighted_scores", {})

    new_total_weight = total_weight - article_weight
    if new_total_weight <= 0:
        return {
            "new_label": "neutral",
            "new_confidence": 0.0,
            "new_normalized": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
            "label_would_change": current_label != "neutral",
        }

    new_normalized = {}
    for label in ["positive", "negative", "neutral"]:
        new_w = total_weighted.get(label, 0.0) - article_weighted_scores.get(label, 0.0)
        new_normalized[label] = safe_round(new_w / new_total_weight)

    new_label = max(new_normalized, key=new_normalized.get)
    new_confidence = safe_round(new_normalized[new_label])

    return {
        "new_label": new_label,
        "new_confidence": new_confidence,
        "new_normalized": new_normalized,
        "label_would_change": new_label != current_label,
    }


def explain_articles(
    merged_articles: list[dict[str, Any]],
    prediction_result: dict[str, Any],
) -> dict[str, Any]:
    total_weight = prediction_result.get("total_weight", 0.0)
    total_weighted = prediction_result.get("weighted_scores", {})
    current_label = prediction_result.get("final_label", "neutral")

    # Sort by final_weight descending
    sorted_articles = sorted(
        merged_articles,
        key=lambda a: a.get("final_weight", 0.0),
        reverse=True,
    )

    ranked = []
    label_flipping = []
    positive_drivers: list[tuple[str, float]] = []
    negative_drivers: list[tuple[str, float]] = []

    all_weights = [a.get("final_weight", 0.0) for a in merged_articles]
    hhi = herfindahl_index(all_weights)

    for rank, article in enumerate(sorted_articles, start=1):
        title = article.get("title", "")
        final_weight = article.get("final_weight", 0.0)
        weight_share = safe_round(final_weight / total_weight) if total_weight > 0 else 0.0
        raw_scores = article.get("raw_scores", {})
        weighted_scores = article.get("weighted_scores", {})
        dominant = get_dominant_label(raw_scores) if raw_scores else current_label

        contribution_share = _compute_contribution_share(article, total_weight)
        counterfactual = _run_counterfactual(article, total_weighted, total_weight, current_label)

        if counterfactual["label_would_change"]:
            label_flipping.append(title)

        pos_contrib = contribution_share.get("positive", 0.0)
        neg_contrib = contribution_share.get("negative", 0.0)
        positive_drivers.append((title, pos_contrib))
        negative_drivers.append((title, neg_contrib))

        ranked.append({
            "rank": rank,
            "title": title,
            "final_weight": safe_round(final_weight),
            "weight_share": weight_share,
            "dominant_sentiment": dominant,
            "raw_scores": {k: safe_round(v) for k, v in raw_scores.items()},
            "weighted_scores": {k: safe_round(v) for k, v in weighted_scores.items()},
            "contribution_share": contribution_share,
            "counterfactual": counterfactual,
        })

    top_positive = [t for t, _ in sorted(positive_drivers, key=lambda x: x[1], reverse=True)[:3]]
    top_negative = [t for t, _ in sorted(negative_drivers, key=lambda x: x[1], reverse=True)[:3]]

    logger.info(
        "Article explanation complete: %d articles, %d label-flipping, HHI=%.4f",
        len(ranked), len(label_flipping), hhi,
    )

    return {
        "ranked_articles": ranked,
        "label_flipping_articles": label_flipping,
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
        "weight_concentration": round(hhi, 4),
    }
