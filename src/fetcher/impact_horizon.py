from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

_fls_pipeline = None


def _get_fls_pipeline():
    global _fls_pipeline
    if _fls_pipeline is None:
        try:
            from transformers import pipeline
            from config import SENTIMENT_DEVICE

            logger.info("Loading FinBERT FLS classifier...")
            _fls_pipeline = pipeline(
                "text-classification",
                model="yiyanghkust/finbert-fls",
                device=SENTIMENT_DEVICE,
            )
            logger.info("FinBERT FLS model loaded successfully")
        except Exception as exc:
            logger.error("Failed to load FinBERT FLS model: %s", exc)
            raise
    return _fls_pipeline


# FLS label to horizon category (for 1-31 day prediction window)
FLS_TO_HORIZON = {
    "Not FLS": "IMMEDIATE",           # Historical/current → quick price reaction
    "Specific FLS": "SHORT_TERM",     # Specific future event → days to weeks
    "Non-specific FLS": "MEDIUM_TERM", # Vague future → weeks to month
}

# Ideal prediction window range for each horizon (in days)
# Tuned for 1-31 day trading predictions
HORIZON_IDEAL_RANGE = {
    "IMMEDIATE": (1, 5),       # News already out, quick reaction
    "SHORT_TERM": (5, 14),     # Specific upcoming events
    "MEDIUM_TERM": (14, 31),   # Vague forward-looking statements
}


def _calculate_dynamic_weight(
    horizon: str,
    prediction_window_days: int,
    min_weight: float = 0.3,
    max_weight: float = 1.5,
) -> float:
    """
    Calculate weight based on how well the horizon matches the prediction window.
    
    - If horizon perfectly matches prediction window → max_weight
    - If horizon is far from prediction window → min_weight
    - Smooth interpolation in between
    """
    ideal_min, ideal_max = HORIZON_IDEAL_RANGE.get(horizon, (5, 14))
    
    # Check if prediction window falls within ideal range
    if ideal_min <= prediction_window_days <= ideal_max:
        # Perfect match — return max weight
        return max_weight
    
    # Calculate distance from ideal range
    if prediction_window_days < ideal_min:
        distance = ideal_min - prediction_window_days
    else:
        distance = prediction_window_days - ideal_max
    
    # Normalize distance (max distance is ~30 days)
    max_distance = 30
    normalized_distance = min(distance / max_distance, 1.0)
    
    # Exponential decay from max to min weight
    weight = max_weight - (max_weight - min_weight) * (normalized_distance ** 0.7)
    
    return round(max(weight, min_weight), 4)


def classify_impact_horizon(text: str, prediction_window_days: int = 7) -> dict:
    """Classify impact horizon using FinBERT FLS with dynamic weighting."""
    pipe = _get_fls_pipeline()
    
    # Truncate for model (512 token limit)
    truncated = text[:512]
    
    result = pipe(truncated)
    
    fls_label = result[0]["label"]
    fls_score = result[0]["score"]
    
    horizon = FLS_TO_HORIZON.get(fls_label, "SHORT_TERM")
    
    # Calculate dynamic weight based on prediction window
    horizon_weight = _calculate_dynamic_weight(horizon, prediction_window_days)
    
    return {
        "fls_label": fls_label,
        "fls_confidence": round(fls_score, 4),
        "impact_horizon": horizon,
        "horizon_weight": horizon_weight,
        "prediction_window_days": prediction_window_days,
    }


def add_impact_horizon_data(
    articles: list[dict],
    prediction_window_days: int = 7,
    combine_method: str = "weighted_avg",
) -> list[dict]:
    """Add impact horizon data to articles using FinBERT FLS with dynamic weighting."""
    
    if not articles:
        return articles
    
    logger.info(
        "Classifying impact horizon for %d articles using FinBERT FLS (prediction window: %d days)",
        len(articles), prediction_window_days
    )
    
    for i, article in enumerate(articles):
        title = article.get("title", "")
        content = article.get("content") or ""
        
        # Combine title and content
        text = f"{title}. {content}".strip()
        
        if not text:
            article["impact_horizon"] = "SHORT_TERM"
            article["horizon_weight"] = 1.0
            article["fls_label"] = None
            article["fls_confidence"] = None
            continue
        
        result = classify_impact_horizon(text, prediction_window_days)
        
        article["fls_label"] = result["fls_label"]
        article["fls_confidence"] = result["fls_confidence"]
        article["impact_horizon"] = result["impact_horizon"]
        article["horizon_weight"] = result["horizon_weight"]
        
        # Combine with recency weight if exists
        recency_weight = article.get("recency_weight", 1.0)
        
        if combine_method == "multiplicative":
            article["final_weight"] = recency_weight * result["horizon_weight"]
        elif combine_method == "geometric":
            article["final_weight"] = (recency_weight * result["horizon_weight"]) ** 0.5
        else:  # weighted_avg
            article["final_weight"] = 0.6 * recency_weight + 0.4 * result["horizon_weight"]
        
        article["final_weight"] = round(article["final_weight"], 4)
        
        logger.debug(
            "[%d] %s -> FLS=%s (%.2f) -> %s (w=%.2f, final=%.2f)",
            i + 1, title[:50], result["fls_label"], result["fls_confidence"],
            result["impact_horizon"], result["horizon_weight"], article["final_weight"],
        )
    
    # Log weight distribution summary
    _log_weight_summary(articles, prediction_window_days)
    
    logger.info("Impact horizon classification complete")
    return articles


def _log_weight_summary(articles: list[dict], prediction_window_days: int) -> None:
    """Log summary of horizon distribution and weights."""
    horizon_counts = {"IMMEDIATE": 0, "SHORT_TERM": 0, "MEDIUM_TERM": 0}
    total_weight = 0.0
    
    for article in articles:
        horizon = article.get("impact_horizon", "SHORT_TERM")
        if horizon in horizon_counts:
            horizon_counts[horizon] += 1
        total_weight += article.get("final_weight", 1.0)
    
    logger.info(
        "Horizon distribution (prediction=%d days): IMMEDIATE=%d, SHORT=%d, MEDIUM=%d | Total weight=%.2f",
        prediction_window_days,
        horizon_counts["IMMEDIATE"],
        horizon_counts["SHORT_TERM"],
        horizon_counts["MEDIUM_TERM"],
        total_weight,
    )
