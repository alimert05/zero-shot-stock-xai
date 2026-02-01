from __future__ import annotations
 
import logging
import math
from typing import Literal
 
logger = logging.getLogger(__name__)
 
_classifier = None
 
 
def _get_classifier():
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline
 
            logger.info("Loading DeBERTa zero-shot classifier...")
            _classifier = pipeline(
                "zero-shot-classification",
                model="MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
                device=-1,  
            )
            logger.info("DeBERTa classifier loaded successfully")
        except Exception as exc:
            logger.error("Failed to load DeBERTa classifier: %s", exc)
            raise
    return _classifier
 
IMPACT_LABELS = [
    "immediate market reaction within 1-2 days",
    "short-term impact within 3-7 days",
    "medium-term impact within 1-2 weeks",
    "long-term impact over 2-4 weeks",
]

LABEL_TO_DAYS: dict[str, int] = {
    "immediate market reaction within 1-2 days": 2,
    "short-term impact within 3-7 days": 5,
    "medium-term impact within 1-2 weeks": 10,
    "long-term impact over 2-4 weeks": 21,
}

HORIZON_CATEGORY = {
    2: "IMMEDIATE",
    5: "SHORT_TERM",
    10: "MEDIUM_TERM",
    21: "LONG_TERM",
}
 
 
def classify_impact_horizon(
    title: str,
    content: str | None = None,
    max_content_chars: int = 500,
) -> dict:
    
    classifier = _get_classifier()

    if content:
        truncated_content = content[:max_content_chars]
        text = f"{title}. {truncated_content}"
    else:
        text = title
 
    try:
        result = classifier(
            text,
            candidate_labels=IMPACT_LABELS,
            hypothesis_template="This financial news will cause {}.",
        )
 
        top_label = result["labels"][0]
        confidence = result["scores"][0]
        horizon_days = LABEL_TO_DAYS[top_label]
        category = HORIZON_CATEGORY[horizon_days]
 
        return {
            "label": top_label,
            "horizon_days": horizon_days,
            "category": category,
            "confidence": confidence,
        }
 
    except Exception as exc:
        logger.warning("Impact horizon classification failed: %s", exc)

        return {
            "label": IMPACT_LABELS[2],
            "horizon_days": 10,
            "category": "MEDIUM_TERM",
            "confidence": 0.0,
        }
 
 
def calculate_impact_horizon_weight(
    days_ago: int,
    impact_horizon_days: int,
    prediction_window_days: int,
) -> float:

    days_until_impact = impact_horizon_days - days_ago

    if days_until_impact < 0:
        days_since_impact = abs(days_until_impact)
        decay_rate = 0.5
        return max(0.05, 0.3 * math.exp(-decay_rate * days_since_impact))
    
    if days_until_impact <= prediction_window_days:
        closeness_to_end = days_until_impact / prediction_window_days
        return 0.5 + 0.5 * closeness_to_end

    overshoot = days_until_impact - prediction_window_days
    decay_rate = 0.1
    return max(0.1, math.exp(-decay_rate * overshoot))
 
 
def calculate_combined_weight(
    recency_weight: float,
    impact_horizon_weight: float,
    method: Literal["weighted_avg", "multiplicative", "geometric"] = "weighted_avg",
    recency_importance: float = 0.4,
    horizon_importance: float = 0.6,
) -> float:

    if method == "weighted_avg":
        return (recency_importance * recency_weight +
                horizon_importance * impact_horizon_weight)
 
    elif method == "multiplicative":
        return recency_weight * impact_horizon_weight
 
    elif method == "geometric":
        return math.sqrt(recency_weight * impact_horizon_weight)
 
    else:
        raise ValueError(f"Unknown combination method: {method}")
 
 
def add_impact_horizon_data(
    articles: list[dict],
    prediction_window_days: int,
    combine_method: Literal["weighted_avg", "multiplicative", "geometric"] = "weighted_avg",
) -> None:

    logger.info(
        "Adding impact horizon data to %d articles (prediction window: %d days)",
        len(articles),
        prediction_window_days,
    )
 
    for i, article in enumerate(articles):
        title = article.get("title", "")
        content = article.get("content", "")
        days_ago = article.get("days_ago", 0)
        recency_weight = article.get("recency_weight", 1.0)

        if not title:
            article["impact_horizon"] = None
            article["impact_horizon_weight"] = 1.0
            article["final_weight"] = recency_weight
            continue

        horizon_result = classify_impact_horizon(title, content)

        horizon_weight = calculate_impact_horizon_weight(
            days_ago=days_ago if days_ago is not None else 0,
            impact_horizon_days=horizon_result["horizon_days"],
            prediction_window_days=prediction_window_days,
        )

        final_weight = calculate_combined_weight(
            recency_weight=recency_weight,
            impact_horizon_weight=horizon_weight,
            method=combine_method,
        )

        article["impact_horizon"] = {
            "category": horizon_result["category"],
            "horizon_days": horizon_result["horizon_days"],
            "confidence": round(horizon_result["confidence"], 4),
        }
        article["impact_horizon_weight"] = round(horizon_weight, 4)
        article["final_weight"] = round(final_weight, 4)
 
        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d articles", i + 1, len(articles))
 
    logger.info("Impact horizon classification complete")