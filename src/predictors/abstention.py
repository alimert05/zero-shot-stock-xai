"""
Shared dynamic abstention margin for all sentiment predictors.

The margin is entropy-adaptive: models that produce confident, well-separated
aggregate scores (e.g. FinBERT) will see low entropy → small margin → the check
rarely triggers.  Models with noisier aggregation (e.g. zero-shot NLI) will see
higher entropy → larger margin → genuine uncertainty is caught.

Same code, same parameters, no per-model favouritism — the entropy mechanism
makes the threshold self-adjusting to each model's calibration quality.

References
----------
- Chow, C. K. (1970). On optimum recognition error and reject tradeoff.
  IEEE Trans. Inform. Theory, 16(1), 41-46.
- Geifman, Y., & El-Yaniv, R. (2017). Selective prediction via rejection.
  NeurIPS.
- Kish, L. (1965). Survey Sampling. Wiley.  (effective sample size)
"""

from __future__ import annotations

import math
import logging

from config import (
    DECISION_THRESHOLD_ENABLED, DECISION_THRESHOLD_POS, DECISION_THRESHOLD_NEG,
    DYNAMIC_MARGIN_ENABLED,
)

logger = logging.getLogger(__name__)


def normalized_entropy(scores: dict[str, float]) -> float:
    """
    Entropy of the score distribution, normalized to [0, 1].

    0 → one class dominates (very certain)
    1 → uniform distribution (maximally uncertain)
    """
    values = [max(float(v), 1e-12) for v in scores.values()]
    total = sum(values)
    if total <= 0:
        return 1.0

    probs = [v / total for v in values]
    entropy = -sum(p * math.log(p) for p in probs)
    max_entropy = math.log(len(probs))
    return entropy / max_entropy if max_entropy > 0 else 1.0


def effective_sample_size(weights: list[float]) -> float:
    """
    Kish-style effective sample size for weighted aggregation.

    If one article dominates the total weight, effective evidence stays low
    even when the raw article count is high.
    """
    if not weights:
        return 0.0

    weight_sum = sum(weights)
    weight_sq_sum = sum(w * w for w in weights)

    if weight_sq_sum <= 0:
        return 0.0

    return (weight_sum * weight_sum) / weight_sq_sum


def dynamic_abstention_margin(
    normalized_scores: dict[str, float],
    article_weights: list[float],
    base_margin: float = 0.02,
    entropy_strength: float = 0.03,
    evidence_strength: float = 0.04,
    min_margin: float = 0.02,
    max_margin: float = 0.12,
) -> float:
    """
    Dynamic abstention threshold.

    Increases when:
      - the class distribution is uncertain (high entropy)
      - the effective amount of evidence is small

    Decreases when:
      - the model distribution is sharp
      - many weighted articles support the aggregate decision

    Returns a float in [min_margin, max_margin].
    """
    entropy_term = normalized_entropy(normalized_scores)
    n_eff = effective_sample_size(article_weights)

    evidence_term = 1.0 / math.sqrt(max(n_eff, 1.0))

    threshold = (
        base_margin
        + (entropy_strength * entropy_term)
        + (evidence_strength * evidence_term)
    )

    return max(min(threshold, max_margin), min_margin)


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
    Apply decision thresholds and dynamic abstention margin.

    Two-stage process:
      1. Decision thresholds: positive/negative must exceed calibrated
         cutoffs, otherwise default to neutral.
      2. Dynamic margin: if the top-two scores are too close, override
         to neutral as a safety net.

    Returns a dict with:
      - final_label: the chosen label (or "neutral" if abstained)
      - abstention_test: diagnostic info about both checks
    """
    # ── Stage 1: Decision thresholds ──
    final_label, threshold_method = apply_decision_thresholds(normalized_scores)

    if threshold_method:
        logger.info(
            "Decision threshold: pos=%.4f (tau=%.2f), neg=%.4f (tau=%.2f) -> %s",
            normalized_scores.get("positive", 0), DECISION_THRESHOLD_POS,
            normalized_scores.get("negative", 0), DECISION_THRESHOLD_NEG,
            final_label,
        )

    # ── Stage 2: Dynamic margin (safety net) ──
    sorted_labels = sorted(normalized_scores, key=normalized_scores.get, reverse=True)
    top_label = sorted_labels[0]
    runner_up = sorted_labels[1]

    margin = round(
        normalized_scores[top_label] - normalized_scores[runner_up], 4
    )

    dyn_threshold = round(
        dynamic_abstention_margin(
            normalized_scores=normalized_scores,
            article_weights=article_weights,
        ),
        4,
    )

    entropy = round(normalized_entropy(normalized_scores), 4)
    n_eff = round(effective_sample_size(article_weights), 4)

    abstention_method = threshold_method

    if DYNAMIC_MARGIN_ENABLED and final_label != "neutral" and margin < dyn_threshold:
        final_label = "neutral"
        abstention_method = "dynamic_margin"
        logger.info(
            "Abstention: margin %.4f < dynamic threshold %.4f -> neutral "
            "(top=%s, runner_up=%s, entropy=%.4f, eff_n=%.4f)",
            margin, dyn_threshold, top_label, runner_up, entropy, n_eff,
        )
    elif DYNAMIC_MARGIN_ENABLED and not threshold_method:
        logger.info(
            "Dynamic margin check passed: %.4f >= %.4f -> keep %s "
            "(entropy=%.4f, eff_n=%.4f)",
            margin, dyn_threshold, top_label, entropy, n_eff,
        )

    # Build threshold gap info for explainability
    pos = normalized_scores.get("positive", 0)
    neg = normalized_scores.get("negative", 0)
    threshold_gap = None
    if abstention_method == "decision_threshold" and final_label == "neutral":
        # Identify the closest directional score and how far it fell short
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
            "method": abstention_method if abstention_method else "none",
            "decision_thresholds": {
                "tau_pos": DECISION_THRESHOLD_POS,
                "tau_neg": DECISION_THRESHOLD_NEG,
            },
            "threshold_gap": threshold_gap,
            "margin": margin,
            "threshold": dyn_threshold,
            "n_articles": len(article_weights),
            "effective_n": n_eff,
            "entropy": entropy,
        },
    }
