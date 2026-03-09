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


def apply_abstention(
    normalized_scores: dict[str, float],
    article_weights: list[float],
) -> dict:
    """
    Apply dynamic abstention margin to final aggregated scores.

    Returns a dict with:
      - final_label: the chosen label (or "neutral" if abstained)
      - abstention_test: diagnostic info about the margin check
    """
    final_label = max(normalized_scores, key=normalized_scores.get)

    sorted_labels = sorted(normalized_scores, key=normalized_scores.get, reverse=True)
    top_label = sorted_labels[0]
    runner_up = sorted_labels[1]

    margin = round(
        normalized_scores[top_label] - normalized_scores[runner_up], 4
    )

    threshold = round(
        dynamic_abstention_margin(
            normalized_scores=normalized_scores,
            article_weights=article_weights,
        ),
        4,
    )

    entropy = round(normalized_entropy(normalized_scores), 4)
    n_eff = round(effective_sample_size(article_weights), 4)

    abstention_method = None

    if margin < threshold:
        final_label = "neutral"
        abstention_method = "dynamic_margin"
        logger.info(
            "Abstention: margin %.4f < dynamic threshold %.4f -> neutral "
            "(top=%s, runner_up=%s, entropy=%.4f, eff_n=%.4f)",
            margin, threshold, top_label, runner_up, entropy, n_eff,
        )
    else:
        logger.info(
            "Dynamic margin check passed: %.4f >= %.4f -> keep %s "
            "(entropy=%.4f, eff_n=%.4f)",
            margin, threshold, top_label, entropy, n_eff,
        )

    return {
        "final_label": final_label,
        "abstention_test": {
            "method": abstention_method if abstention_method else "none",
            "margin": margin,
            "threshold": threshold,
            "n_articles": len(article_weights),
            "effective_n": n_eff,
            "entropy": entropy,
        },
    }
