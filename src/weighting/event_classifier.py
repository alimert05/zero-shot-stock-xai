"""Event family classification using zero-shot NLI."""

from __future__ import annotations

import logging

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
    "D6_10_DIFFUSION": "DIFFUSION",
    "D11_20_EXTENDED": "LONG_TERM",
    "D21_31_PERSISTENT": "LONG_TERM",
}

# Optional backward-compatible mapping by representative day
HORIZON_CATEGORY = {
    1: "IMMEDIATE",
    5: "SHORT_TERM",
    10: "DIFFUSION",
    20: "LONG_TERM",
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


def classify_impact_horizon(
    title: str,
    content: str | None = None,
    max_content_chars: int = 700,
) -> dict:
    """
    Classify article into an event family, then attach literature-informed
    primary and secondary horizon priors.

    Returns backward-compatible keys:
      - event_type, label, horizon_days, category, confidence

    Plus richer keys:
      - event_family, primary_horizon_label, primary_horizon_days,
        primary_category, secondary_*, alternative_*
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
        return _map_classifier_result(result)

    except Exception as exc:
        logger.warning("Impact horizon classification failed: %s", exc)
        return _fallback_horizon_result()


def _map_classifier_result(result: dict) -> dict:
    """Map a single zero-shot classifier result dict to horizon fields.

    Shared by both single-article (classify_impact_horizon) and batched paths.
    """
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
        "event_family": top_family,
        "primary_horizon_label": primary_horizon_label,
        "primary_horizon_days": primary_horizon_days,
        "primary_category": primary_category,
        "secondary_horizon_label": secondary_horizon_label,
        "secondary_horizon_days": secondary_horizon_days,
        "secondary_category": secondary_category,
        "alternative_event_family": alternative_event_family,
        "alternative_confidence": alternative_confidence,
        # Backward-compatible aliases
        "event_type": top_family,
        "label": top_family,
        "horizon_days": primary_horizon_days,
        "category": primary_category,
        "confidence": confidence,
    }


def _fallback_horizon_result() -> dict:
    """Return a fallback horizon result when classification fails."""
    fallback_prior = EVENT_FAMILY_TO_PRIOR_HORIZON[FALLBACK_EVENT_FAMILY]
    primary_label = str(fallback_prior["primary"])
    secondary_label = fallback_prior.get("secondary")
    secondary_label = str(secondary_label) if secondary_label else None

    return {
        "event_family": FALLBACK_EVENT_FAMILY,
        "primary_horizon_label": primary_label,
        "primary_horizon_days": HORIZON_TO_DAYS[primary_label],
        "primary_category": HORIZON_LABEL_TO_CATEGORY[primary_label],
        "secondary_horizon_label": secondary_label,
        "secondary_horizon_days": (
            HORIZON_TO_DAYS[secondary_label] if secondary_label else None
        ),
        "secondary_category": (
            HORIZON_LABEL_TO_CATEGORY[secondary_label] if secondary_label else None
        ),
        "alternative_event_family": None,
        "alternative_confidence": None,
        "event_type": FALLBACK_EVENT_FAMILY,
        "label": FALLBACK_EVENT_FAMILY,
        "horizon_days": HORIZON_TO_DAYS[primary_label],
        "category": HORIZON_LABEL_TO_CATEGORY[primary_label],
        "confidence": 0.0,
    }
