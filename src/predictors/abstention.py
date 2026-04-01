"""
Decision threshold abstention for sentiment predictors.

Applies per-class decision thresholds so that low-confidence directional
predictions default to neutral rather than committing to an unreliable signal.

References
----------
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff.
  IEEE Trans. Inform. Theory, 16(1), 41-46.
- Geifman, Y., & El-Yaniv, R. (2017). Selective prediction via rejection.
  NeurIPS.
"""

from __future__ import annotations

import logging

from config import (
    DECISION_THRESHOLD_ENABLED, DECISION_THRESHOLD_POS, DECISION_THRESHOLD_NEG,
)

logger = logging.getLogger(__name__)


def apply_decision_thresholds(
    normalized_scores: dict[str, float],
    tau_pos: float = DECISION_THRESHOLD_POS,
    tau_neg: float = DECISION_THRESHOLD_NEG,
) -> tuple[str, str | None]:
    """Apply per-class decision thresholds to aggregated scores.

    Logic:
        if positive >= tau_pos  -> positive
        elif negative >= tau_neg -> negative
        else                    -> neutral

    Returns
    -------
    (label, method)
        method is "decision_threshold" if thresholds changed the outcome
        from argmax, else None.
    """
    argmax_label = max(normalized_scores, key=normalized_scores.get)

    if not DECISION_THRESHOLD_ENABLED:
        return argmax_label, None

    pos = normalized_scores.get("positive", 0)
    neg = normalized_scores.get("negative", 0)

    if pos >= tau_pos and pos >= neg:
        label = "positive"
    elif neg >= tau_neg and neg >= pos:
        label = "negative"
    elif pos >= tau_pos:
        label = "positive"
    elif neg >= tau_neg:
        label = "negative"
    else:
        label = "neutral"

    method = "decision_threshold" if label != argmax_label else None
    return label, method


def apply_abstention(
    normalized_scores: dict[str, float],
    article_weights: list[float],
) -> dict:
    """
    Apply decision thresholds to aggregated scores.

    Positive/negative must exceed calibrated cutoffs, otherwise default
    to neutral.

    Returns a dict with:
      - final_label: the chosen label (or "neutral" if abstained)
      - abstention_test: diagnostic info
    """
    # ── Decision thresholds ──
    final_label, threshold_method = apply_decision_thresholds(normalized_scores)

    if threshold_method:
        logger.info(
            "Decision threshold: pos=%.4f (tau=%.2f), neg=%.4f (tau=%.2f) -> %s",
            normalized_scores.get("positive", 0), DECISION_THRESHOLD_POS,
            normalized_scores.get("negative", 0), DECISION_THRESHOLD_NEG,
            final_label,
        )

    # Compute margin between top two scores
    sorted_labels = sorted(normalized_scores, key=normalized_scores.get, reverse=True)
    top_label = sorted_labels[0]
    runner_up = sorted_labels[1]

    margin = round(
        normalized_scores[top_label] - normalized_scores[runner_up], 4
    )

    # Build threshold gap info for explainability
    pos = normalized_scores.get("positive", 0)
    neg = normalized_scores.get("negative", 0)
    threshold_gap = None
    if threshold_method == "decision_threshold" and final_label == "neutral":
        pos_gap = DECISION_THRESHOLD_POS - pos
        neg_gap = DECISION_THRESHOLD_NEG - neg
        if pos >= neg:
            threshold_gap = {
                "nearest_label": "positive",
                "score": round(pos, 4),
                "threshold": DECISION_THRESHOLD_POS,
                "shortfall": round(pos_gap, 4),
            }
        else:
            threshold_gap = {
                "nearest_label": "negative",
                "score": round(neg, 4),
                "threshold": DECISION_THRESHOLD_NEG,
                "shortfall": round(neg_gap, 4),
            }

    return {
        "final_label": final_label,
        "abstention_test": {
            "method": threshold_method if threshold_method else "none",
            "decision_thresholds": {
                "tau_pos": DECISION_THRESHOLD_POS,
                "tau_neg": DECISION_THRESHOLD_NEG,
            },
            "threshold_gap": threshold_gap,
            "margin": margin,
            "n_articles": len(article_weights),
        },
    }
