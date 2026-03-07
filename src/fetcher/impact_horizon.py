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

            logger.info("Loading zero-shot classifier...")
            _classifier = pipeline(
                "zero-shot-classification",
                model=IMPACT_HORIZON_MODEL,
                device=IMPACT_HORIZON_DEVICE,
            )
            logger.info("Zero-shot classifier loaded successfully")
        except Exception as exc:
            logger.error("Failed to load zero-shot classifier: %s", exc)
            raise
    return _classifier


EVENT_FAMILY_LABELS = [
    # 1 - Ball & Brown (1968): Day-0 spike; Bernard & Thomas (1989): PEAD 60+ days
    "earnings report, guidance, or financial results",
    # 2 - Kim et al. (1997): priced in 5-15 min; Lloyd Davies & Canes (1978): 2-day
    "analyst upgrade, downgrade, or price target revision",
    # 3 - Warren & Sorescu (2017): 5-day window standard
    "product launch, partnership, contract, or business development",
    # 4 - Vermaelen (1981): Day 0-2 reaction; post-announcement drift weeks
    "share buyback, dividend, stock offering, or debt issuance",
    # 5 - Holthausen & Leftwich (1986): CAR -7.5% at [-1,0]; investigation-filing ~9d
    "lawsuit, investigation, regulatory action, or compliance issue",
    # 6 - Target premium [-1,+1]; deal uncertainty, regulatory approval weeks-months
    "merger, acquisition, takeover, or corporate restructuring",
    # 7 - CARs significant in 3-5 day windows (Clayton, Hartzell & Rosenberg)
    "CEO change, executive departure, or board appointment",
    # 8 - Heston & Sinha (2016): commentary fades fast; Barberis et al. (2015): exp decay
    "market commentary, sector outlook, or opinion piece",
    # 9 - Modelling choice: ongoing conditions w/o clean event date; Altman (1968),
    #     Campbell et al. (2008) — informational relevance persists over longer windows
    "financial distress, credit downgrade, or going concern warning",
]

# Backward-compatible alias
EVENT_TYPE_LABELS = EVENT_FAMILY_LABELS

HORIZON_TO_DAYS: dict[str, int] = {
    "D1_IMMEDIATE": 1,
    "D2_5_SHORT": 5,
    "D6_10_DIFFUSION": 10,
    "D11_20_EXTENDED": 20,
    "D21_31_PERSISTENT": 31,
}

HORIZON_LABEL_TO_CATEGORY: dict[str, str] = {
    "D1_IMMEDIATE": "IMMEDIATE",
    "D2_5_SHORT": "SHORT_TERM",
    "D6_10_DIFFUSION": "SHORT_TERM",
    "D11_20_EXTENDED": "MEDIUM_TERM",
    "D21_31_PERSISTENT": "LONG_TERM",
}

# Optional backward-compatible mapping by representative day
HORIZON_CATEGORY = {
    1: "IMMEDIATE",
    5: "SHORT_TERM",
    10: "SHORT_TERM",
    20: "MEDIUM_TERM",
    31: "LONG_TERM",
}

EVENT_FAMILY_TO_PRIOR_HORIZON: dict[str, dict[str, str | None]] = {
    "earnings report, guidance, or financial results": {
        "primary": "D1_IMMEDIATE",
        "secondary": "D6_10_DIFFUSION", 
    },
    "analyst upgrade, downgrade, or price target revision": {
        "primary": "D1_IMMEDIATE",         
        "secondary": "D2_5_SHORT",       
    },
    "product launch, partnership, contract, or business development": {
        "primary": "D2_5_SHORT",            
        "secondary": "D6_10_DIFFUSION",
    },
    "share buyback, dividend, stock offering, or debt issuance": {
        "primary": "D2_5_SHORT",
        "secondary": "D11_20_EXTENDED",     
    },
    "lawsuit, investigation, regulatory action, or compliance issue": {
        "primary": "D2_5_SHORT",            
        "secondary": "D11_20_EXTENDED",     
    },
    "merger, acquisition, takeover, or corporate restructuring": {
        "primary": "D6_10_DIFFUSION",        
        "secondary": "D21_31_PERSISTENT",     
    },
    "CEO change, executive departure, or board appointment": {
        "primary": "D2_5_SHORT",            
        "secondary": "D6_10_DIFFUSION",
    },
    "market commentary, sector outlook, or opinion piece": {
        "primary": "D1_IMMEDIATE",      
        "secondary": "D2_5_SHORT",           
    },
    "financial distress, credit downgrade, or going concern warning": {
        "primary": "D21_31_PERSISTENT",       
        "secondary": "D11_20_EXTENDED",       
    },
}

# Backward-compatible alias
EVENT_TYPE_TO_HORIZON: dict[str, dict] = {
    event_family: {
        "days": HORIZON_TO_DAYS[prior["primary"]],
        "category": HORIZON_LABEL_TO_CATEGORY[prior["primary"]],
        "horizon_label": prior["primary"],
    }
    for event_family, prior in EVENT_FAMILY_TO_PRIOR_HORIZON.items()
}

FALLBACK_EVENT_FAMILY = "market commentary, sector outlook, or opinion piece"


def _build_classification_text(
    title: str,
    content: str | None = None,
    max_content_chars: int = 700,
) -> str:
    title = (title or "").strip()
    content = (content or "").strip()

    if not content:
        return title

    truncated_content = content[:max_content_chars].strip()
    if not truncated_content:
        return title

    return f"{title}. {truncated_content}"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def classify_impact_horizon(
    title: str,
    content: str | None = None,
    max_content_chars: int = 700,
) -> dict:
    """
    Classify article into an event family, then attach literature-informed
    primary and secondary horizon priors.

    Returns backward-compatible keys:
      - event_type
      - label
      - horizon_days
      - category
      - confidence

    Plus richer keys:
      - event_family
      - primary_horizon_label
      - primary_horizon_days
      - primary_category
      - secondary_horizon_label
      - secondary_horizon_days
      - secondary_category
      - alternative_event_family
      - alternative_confidence
    """

    classifier = _get_classifier()
    text = _build_classification_text(title, content, max_content_chars=max_content_chars)

    try:
        result = classifier(
            text,
            candidate_labels=EVENT_FAMILY_LABELS,
            hypothesis_template="The primary firm-specific event in this news article is {}.",
            multi_label=False,
        )

        labels = result.get("labels") or []
        scores = result.get("scores") or []

        if not labels:
            raise ValueError("Zero-shot classifier returned no labels")

        top_family = str(labels[0])
        confidence = float(scores[0]) if scores else 0.0

        alternative_event_family = str(labels[1]) if len(labels) > 1 else None
        alternative_confidence = float(scores[1]) if len(scores) > 1 else None

        prior = EVENT_FAMILY_TO_PRIOR_HORIZON.get(
            top_family,
            EVENT_FAMILY_TO_PRIOR_HORIZON[FALLBACK_EVENT_FAMILY],
        )

        primary_horizon_label = str(prior["primary"])
        secondary_horizon_label = prior.get("secondary")
        secondary_horizon_label = str(secondary_horizon_label) if secondary_horizon_label else None

        primary_horizon_days = HORIZON_TO_DAYS[primary_horizon_label]
        primary_category = HORIZON_LABEL_TO_CATEGORY[primary_horizon_label]

        secondary_horizon_days = (
            HORIZON_TO_DAYS[secondary_horizon_label]
            if secondary_horizon_label is not None
            else None
        )
        secondary_category = (
            HORIZON_LABEL_TO_CATEGORY[secondary_horizon_label]
            if secondary_horizon_label is not None
            else None
        )

        return {
            # New fields
            "event_family": top_family,
            "primary_horizon_label": primary_horizon_label,
            "primary_horizon_days": primary_horizon_days,
            "primary_category": primary_category,
            "secondary_horizon_label": secondary_horizon_label,
            "secondary_horizon_days": secondary_horizon_days,
            "secondary_category": secondary_category,
            "alternative_event_family": alternative_event_family,
            "alternative_confidence": alternative_confidence,
            # Backward-compatible fields
            "event_type": top_family,
            "label": top_family,
            "horizon_days": primary_horizon_days,
            "category": primary_category,
            "confidence": confidence,
        }

    except Exception as exc:
        logger.warning("Impact horizon classification failed: %s", exc)

        fallback_prior = EVENT_FAMILY_TO_PRIOR_HORIZON[FALLBACK_EVENT_FAMILY]
        primary_horizon_label = str(fallback_prior["primary"])
        secondary_horizon_label = fallback_prior.get("secondary")
        secondary_horizon_label = str(secondary_horizon_label) if secondary_horizon_label else None

        primary_horizon_days = HORIZON_TO_DAYS[primary_horizon_label]
        primary_category = HORIZON_LABEL_TO_CATEGORY[primary_horizon_label]

        secondary_horizon_days = (
            HORIZON_TO_DAYS[secondary_horizon_label]
            if secondary_horizon_label is not None
            else None
        )
        secondary_category = (
            HORIZON_LABEL_TO_CATEGORY[secondary_horizon_label]
            if secondary_horizon_label is not None
            else None
        )

        return {
            # New fields
            "event_family": FALLBACK_EVENT_FAMILY,
            "primary_horizon_label": primary_horizon_label,
            "primary_horizon_days": primary_horizon_days,
            "primary_category": primary_category,
            "secondary_horizon_label": secondary_horizon_label,
            "secondary_horizon_days": secondary_horizon_days,
            "secondary_category": secondary_category,
            "alternative_event_family": None,
            "alternative_confidence": None,
            # Backward-compatible fields
            "event_type": FALLBACK_EVENT_FAMILY,
            "label": FALLBACK_EVENT_FAMILY,
            "horizon_days": primary_horizon_days,
            "category": primary_category,
            "confidence": 0.0,
        }


def _single_horizon_weight(
    days_ago: int,
    impact_horizon_days: int,
    prediction_window_days: int,
    min_weight: float = 0.05,
) -> float:
    """
    Same core weighting shape you already used, but extracted so that
    primary and secondary horizons can be combined.
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
    """
    Backward compatible:
      - old usage can still pass impact_horizon_days

    New usage:
      - pass primary_horizon_days
      - optionally pass secondary_horizon_days
      - confidence controls how strongly we trust the primary prior
    """
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
    primary_mix = 0.60 + (0.25 * confidence)   # 0.60 .. 0.85
    secondary_mix = 1.0 - primary_mix

    combined = (primary_mix * primary_weight) + (secondary_mix * secondary_weight)
    return max(combined, min_weight)


def calculate_combined_weight(
    recency_weight: float,
    impact_horizon_weight: float,
) -> float:
    recency_weight = max(float(recency_weight), 0.0)
    impact_horizon_weight = max(float(impact_horizon_weight), 0.0)
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

        if not title:
            article["impact_horizon"] = None
            article["impact_horizon_weight"] = 1.0
            article["final_weight"] = recency_weight
            continue

        horizon_result = classify_impact_horizon(title, content)

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
            # Backward-compatible fields
            "event_type": horizon_result["event_type"],
            "label": horizon_result["label"],
            "category": horizon_result["category"],
            "horizon_days": horizon_result["horizon_days"],
            "confidence": round(horizon_result["confidence"], 4),
            # New richer fields
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

        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d articles", i + 1, len(articles))

    logger.info("Impact horizon classification complete")