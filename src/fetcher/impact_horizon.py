from __future__ import annotations
 
import logging
import math
 
logger = logging.getLogger(__name__)
 
_classifier = None
 
 
def _get_classifier():
    global _classifier
    if _classifier is None:
        try:
            from transformers import pipeline
            from config import IMPACT_HORIZON_DEVICE, IMPACT_HORIZON_MODEL
 
            logger.info("Loading DeBERTa zero-shot classifier...")
            _classifier = pipeline(
                "zero-shot-classification",
                model=IMPACT_HORIZON_MODEL,
                device=IMPACT_HORIZON_DEVICE,  
            )
            logger.info("DeBERTa classifier loaded successfully")
        except Exception as exc:
            logger.error("Failed to load DeBERTa classifier: %s", exc)
            raise
    return _classifier
 
IMPACT_LABELS = [
    "immediate market reaction within 1-5 days",
    "short-term impact within 5-10 days",
    "medium-term impact within 2-3 weeks",
    "long-term impact over 4 weeks",
]

LABEL_TO_DAYS: dict[str, int] = {
    "immediate market reaction within 1-5 days": 3,
    "short-term impact within 5-10 days": 7,
    "medium-term impact within 2-3 weeks": 14,
    "long-term impact over 4 weeks": 31,
}

HORIZON_CATEGORY = {
    3: "IMMEDIATE",
    7: "SHORT_TERM",
    14: "MEDIUM_TERM",
    31: "LONG_TERM",
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
    
    W = prediction_window_days
    impact_day = impact_horizon_days - days_ago
    mu = W / 2.0
    sigma = W / 2.0
    return math.exp(-((impact_day - mu) ** 2) / (2.0 * sigma ** 2))
 
 
def calculate_combined_weight(
    recency_weight: float,
    impact_horizon_weight: float,
) -> float:
    return math.sqrt(recency_weight * impact_horizon_weight)

 
 
def add_impact_horizon_data(
    articles: list[dict],
    prediction_window_days: int,
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