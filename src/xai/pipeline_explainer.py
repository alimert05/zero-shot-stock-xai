from __future__ import annotations

import logging
import math
from typing import Any

from .utils import safe_round

logger = logging.getLogger(__name__)

# Anchors from fetcher/utils.py _compute_ewma_lambda
_EWMA_ANCHORS = [(1, 0.89), (5, 0.92), (10, 0.95), (21, 0.97)]


def _compute_ewma_lambda(prediction_window_days: int) -> float:
    W = prediction_window_days
    if W <= _EWMA_ANCHORS[0][0]:
        return _EWMA_ANCHORS[0][1]
    if W >= _EWMA_ANCHORS[-1][0]:
        return _EWMA_ANCHORS[-1][1]
    for i in range(len(_EWMA_ANCHORS) - 1):
        w_lo, lam_lo = _EWMA_ANCHORS[i]
        w_hi, lam_hi = _EWMA_ANCHORS[i + 1]
        if w_lo <= W <= w_hi:
            t = (W - w_lo) / (w_hi - w_lo)
            return lam_lo + t * (lam_hi - lam_lo)
    return _EWMA_ANCHORS[-1][1]


def _explain_recency_weight(
    recency_weight: float,
    days_ago: int,
    prediction_window_days: int,
) -> dict[str, Any]:
    lam = _compute_ewma_lambda(prediction_window_days)
    interpretation = (
        f"Article is {days_ago} day(s) old; "
        f"EWMA λ={lam:.4f} gives recency weight {recency_weight:.4f} "
        f"(λ^{days_ago} = {lam}^{days_ago})"
    )
    return {
        "days_ago": days_ago,
        "recency_weight": safe_round(recency_weight),
        "ewma_lambda": round(lam, 4),
        "interpretation": interpretation,
    }


def _explain_impact_horizon_weight(
    impact_horizon: dict[str, Any],
    impact_horizon_weight: float,
    days_ago: int,
    prediction_window_days: int,
) -> dict[str, Any]:
    W = prediction_window_days
    horizon_days = impact_horizon.get("horizon_days", 7)
    category = impact_horizon.get("category", "UNKNOWN")
    confidence = impact_horizon.get("confidence", 0.0)

    impact_day = horizon_days - days_ago
    mu = W / 2.0
    sigma = W / 2.0

    event_type = impact_horizon.get("event_type", "unknown")

    interpretation = (
        f"event type: {event_type} → {category} horizon ({horizon_days} days); "
        f"expected impact day = {impact_day} (horizon_days - days_ago); "
        f"Gaussian centre μ={mu}, σ={sigma}; "
        f"weight = exp(-({impact_day}-{mu})²/(2×{sigma}²)) = {impact_horizon_weight:.4f}"
    )
    return {
        "event_type": event_type,
        "horizon_category": category,
        "horizon_days": horizon_days,
        "horizon_confidence": safe_round(confidence),
        "impact_day": safe_round(impact_day),
        "gaussian_mu": safe_round(mu),
        "gaussian_sigma": safe_round(sigma),
        "impact_horizon_weight": safe_round(impact_horizon_weight),
        "interpretation": interpretation,
    }


def _explain_weight_combination(
    recency_weight: float,
    impact_horizon_weight: float,
    final_weight: float,
) -> dict[str, Any]:
    interpretation = (
        f"geometric mean: sqrt({recency_weight:.4f} × {impact_horizon_weight:.4f}) "
        f"= {final_weight:.4f}"
    )
    return {
        "recency_weight": safe_round(recency_weight),
        "impact_horizon_weight": safe_round(impact_horizon_weight),
        "combination_method": "geometric_mean",
        "final_weight": safe_round(final_weight),
        "interpretation": interpretation,
    }


def explain_pipeline(
    merged_articles: list[dict[str, Any]],
    prediction_window_days: int,
) -> dict[str, Any]:
    horizon_distribution = {
        "IMMEDIATE": 0,
        "SHORT_TERM": 0,
        "MEDIUM_TERM": 0,
        "LONG_TERM": 0,
    }
    event_type_distribution: dict[str, int] = {}

    explained_articles = []
    recency_weights = []
    horizon_weights = []
    days_ago_values = []

    for article in merged_articles:
        title = article.get("title", "")
        recency_weight = article.get("recency_weight", 1.0) or 1.0
        impact_horizon_weight = article.get("impact_horizon_weight", 1.0) or 1.0
        final_weight = article.get("final_weight", 1.0) or 1.0
        days_ago = article.get("days_ago") or 0
        impact_horizon = article.get("impact_horizon") or {}

        category = impact_horizon.get("category", "UNKNOWN")
        if category in horizon_distribution:
            horizon_distribution[category] += 1

        event_type = impact_horizon.get("event_type", "unknown")
        event_type_distribution[event_type] = event_type_distribution.get(event_type, 0) + 1

        recency_exp = _explain_recency_weight(recency_weight, days_ago, prediction_window_days)
        horizon_exp = _explain_impact_horizon_weight(
            impact_horizon, impact_horizon_weight, days_ago, prediction_window_days
        )
        combo_exp = _explain_weight_combination(recency_weight, impact_horizon_weight, final_weight)

        explained_articles.append({
            "title": title,
            "recency_explanation": recency_exp,
            "horizon_explanation": horizon_exp,
            "combination_explanation": combo_exp,
            "final_weight": safe_round(final_weight),
        })

        recency_weights.append(recency_weight)
        horizon_weights.append(impact_horizon_weight)
        days_ago_values.append(days_ago)

    n = len(merged_articles) or 1
    avg_days_ago = round(sum(days_ago_values) / n, 2)
    avg_recency = round(sum(recency_weights) / n, 4)
    avg_horizon = round(sum(horizon_weights) / n, 4)

    weight_formula = (
        "final_weight = sqrt(recency_weight * impact_horizon_weight); "
        "recency_weight = lambda^days_ago (EWMA, lambda interpolated by window size); "
        "impact_horizon_weight = exp(-((horizon_days - days_ago - W/2)^2) / (2*(W/2)^2))"
    )

    logger.info(
        "Pipeline explanation complete: %d articles, horizon dist=%s",
        len(merged_articles), horizon_distribution,
    )

    return {
        "prediction_window_days": prediction_window_days,
        "weight_formula": weight_formula,
        "articles": explained_articles,
        "horizon_distribution": horizon_distribution,
        "event_type_distribution": event_type_distribution,
        "avg_days_ago": avg_days_ago,
        "avg_recency_weight": avg_recency,
        "avg_horizon_weight": avg_horizon,
    }

