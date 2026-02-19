from __future__ import annotations

import logging
from typing import Any

from config import (
    XAI_THIN_EVIDENCE_THRESHOLD,
    XAI_CONCENTRATION_THRESHOLD,
    XAI_MARGIN_THRESHOLD,
    XAI_LOW_CONFIDENCE_THRESHOLD,
)

logger = logging.getLogger(__name__)


def _check_thin_evidence(articles_analyzed: int) -> dict[str, Any]:
    threshold = XAI_THIN_EVIDENCE_THRESHOLD
    flagged = articles_analyzed < threshold
    return {
        "flagged": flagged,
        "articles_analyzed": articles_analyzed,
        "threshold": threshold,
        "message": (
            f"Only {articles_analyzed} articles analyzed (threshold: {threshold})."
            if flagged
            else "Sufficient article count."
        ),
    }


def _check_weight_concentration(herfindahl: float) -> dict[str, Any]:
    threshold = XAI_CONCENTRATION_THRESHOLD
    flagged = herfindahl > threshold
    return {
        "flagged": flagged,
        "herfindahl_index": round(herfindahl, 4),
        "threshold": threshold,
        "message": (
            f"Weight heavily concentrated (Herfindahl={herfindahl:.3f} > {threshold})."
            if flagged
            else "Weight is well-distributed across articles."
        ),
    }


def _check_label_margin(normalized_scores: dict[str, float]) -> dict[str, Any]:
    threshold = XAI_MARGIN_THRESHOLD
    sorted_labels = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    top_label, top_score = sorted_labels[0]
    second_label, second_score = sorted_labels[1]
    margin = round(top_score - second_score, 4)
    flagged = margin < threshold
    return {
        "flagged": flagged,
        "top_label": top_label,
        "second_label": second_label,
        "margin": margin,
        "threshold": threshold,
        "message": (
            f"Narrow margin between {top_label} ({top_score:.3f}) and "
            f"{second_label} ({second_score:.3f}): margin={margin:.3f}."
            if flagged
            else f"Clear margin between top two labels ({margin:.3f})."
        ),
    }


def _check_low_confidence(final_confidence: float) -> dict[str, Any]:
    threshold = XAI_LOW_CONFIDENCE_THRESHOLD
    flagged = final_confidence < threshold
    return {
        "flagged": flagged,
        "final_confidence": round(final_confidence, 4),
        "threshold": threshold,
        "message": (
            f"Low prediction confidence ({final_confidence:.3f} < {threshold})."
            if flagged
            else "Confidence above threshold."
        ),
    }


def compute_reliability(
    prediction_result: dict[str, Any],
    herfindahl_index: float,
) -> dict[str, Any]:
    articles_analyzed = prediction_result.get("articles_analyzed", 0)
    normalized_scores = prediction_result.get("normalized_scores", {})
    final_confidence = prediction_result.get("final_confidence", 0.0)

    flags = {
        "thin_evidence":        _check_thin_evidence(articles_analyzed),
        "weight_concentration": _check_weight_concentration(herfindahl_index),
        "label_margin":         _check_label_margin(normalized_scores),
        "low_confidence":       _check_low_confidence(final_confidence),
    }

    flags_triggered = sum(1 for f in flags.values() if f["flagged"])

    if flags_triggered == 0:
        overall = "HIGH"
    elif flags_triggered == 1:
        overall = "MEDIUM"
    else:
        overall = "LOW"

    flagged_messages = [f["message"] for f in flags.values() if f["flagged"]]
    if flagged_messages:
        summary = f"Prediction has {overall} reliability: " + " ".join(flagged_messages)
    else:
        summary = (
            f"Prediction has HIGH reliability: {articles_analyzed} articles analyzed "
            f"with clear {prediction_result.get('final_label', '')} margin."
        )

    logger.info("Reliability: %s (%d flags)", overall, flags_triggered)

    return {
        "overall_reliability": overall,
        "flags_triggered": flags_triggered,
        "flags": flags,
        "summary_message": summary,
    }
