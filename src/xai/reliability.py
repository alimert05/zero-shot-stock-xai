"""Evidence-quality flags for sentiment predictions.

Checks evidence volume, weight concentration, flip-set sensitivity, and
horizon coverage to produce an overall HIGH / MEDIUM / LOW evidence-quality
rating.
"""
from __future__ import annotations

import logging
from typing import Any

from config import (
    XAI_THIN_EVIDENCE_THRESHOLD,
    XAI_CONCENTRATION_THRESHOLD,
    XAI_MARGIN_THRESHOLD,
    XAI_FLIP_SENSITIVITY_THRESHOLD,
    XAI_SOURCE_CONCENTRATION_THRESHOLD
)

logger = logging.getLogger(__name__)


def _check_thin_evidence(articles_analyzed: int) -> dict[str, Any]:
    """Flag when the number of analyzed articles falls below the minimum threshold."""
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
    """Flag when article weight concentration exceeds the Herfindahl threshold."""
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


# Known news aggregators - these collect articles from many independent
# editorial sources, so a high share from an aggregator does NOT mean
# low editorial diversity.
_AGGREGATOR_DOMAINS = {
    "yahoo", "yahoo finance", "google", "google news", "finnhub",
    "msn", "msn money", "apple news", "smartnews", "flipboard",
    "newsbreak", "ground news",
}


def _check_source_diversity(
    merged_articles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Flag when editorial source diversity is too low, excluding known aggregators."""
    from collections import Counter

    domains: list[str] = []
    for art in merged_articles:
        domain = art.get("domain", "") or art.get("input_source", "unknown")
        domains.append(domain.lower().strip())

    unique = set(domains)
    n_unique = len(unique)
    total = len(domains) or 1

    counts = Counter(domains)
    top_domain, top_count = counts.most_common(1)[0] if counts else ("unknown", 0)
    top_share = round(top_count / total, 4)

    # Exclude aggregators from the concentration check
    non_agg_domains = [d for d in domains if d not in _AGGREGATOR_DOMAINS]
    n_non_agg = len(non_agg_domains) or 1
    non_agg_counts = Counter(non_agg_domains)
    if non_agg_counts:
        top_editorial, top_ed_count = non_agg_counts.most_common(1)[0]
        top_editorial_share = round(top_ed_count / n_non_agg, 4)
    else:
        top_editorial, top_editorial_share = "none", 0.0

    n_unique_editorial = len(set(non_agg_domains)) if non_agg_domains else 0

    too_few = n_unique_editorial < 2 and n_unique < 2
    too_concentrated = top_editorial_share > XAI_SOURCE_CONCENTRATION_THRESHOLD
    flagged = too_few or too_concentrated

    if flagged:
        parts = []
        if too_few:
            parts.append(f"only {n_unique_editorial} unique editorial source(s)")
        if too_concentrated:
            parts.append(
                f"top editorial source '{top_editorial}' has "
                f"{top_editorial_share * 100:.0f}% of non-aggregator articles"
            )
        msg = "Source diversity concern: " + "; ".join(parts) + "."
    else:
        msg = (
            f"{n_unique} sources ({n_unique_editorial} editorial), "
            f"top editorial share {top_editorial_share * 100:.0f}%."
        )

    return {
        "flagged": flagged,
        "unique_sources": n_unique,
        "top_domain": top_domain,
        "top_domain_share": top_share,
        "message": msg,
    }


def _check_label_margin(normalized_scores: dict[str, float]) -> dict[str, Any]:
    """Flag when the margin between the top two sentiment labels is too narrow."""
    threshold = XAI_MARGIN_THRESHOLD
    sorted_labels = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_labels) < 2:
        return {"flagged": False, "margin": 1.0, "message": "Fewer than 2 labels."}
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


def _check_flip_sensitivity(flip_set_data: dict[str, Any] | None) -> dict[str, Any]:
    """Flag when the prediction can be flipped by removing a small number of articles."""
    threshold = XAI_FLIP_SENSITIVITY_THRESHOLD

    if not flip_set_data:
        return {
            "flagged": False,
            "flip_set_size": None,
            "threshold": threshold,
            "message": "Flip-set data not available.",
        }

    flip_possible = flip_set_data.get("flip_possible", False)
    flip_set_size = flip_set_data.get("flip_set_size")
    articles_total = flip_set_data.get("articles_total", 0)

    if not flip_possible or flip_set_size is None:
        return {
            "flagged": False,
            "flip_set_size": articles_total,
            "threshold": threshold,
            "message": (
                f"Prediction is robust: no feasible combination of article "
                f"removals would change the label ({articles_total} articles)."
            ),
        }

    flagged = flip_set_size <= threshold
    return {
        "flagged": flagged,
        "flip_set_size": flip_set_size,
        "articles_total": articles_total,
        "threshold": threshold,
        "message": (
            f"Prediction is sensitive: removing {flip_set_size} of "
            f"{articles_total} articles would change the label."
            if flagged
            else f"Prediction requires removing {flip_set_size} of "
            f"{articles_total} articles to change the label."
        ),
    }


def _check_horizon_coverage(
    merged_articles: list[dict[str, Any]],
    prediction_window_days: int,
    max_backward_days: int | None = None,
) -> dict[str, Any]:
    """Flag when actual news lookback span is shorter than the intended lookback window."""
    ages = [a.get("days_ago", 0) for a in merged_articles]
    if not ages:
        return {
            "flagged": True,
            "lookback_days": 0,
            "intended_lookback_days": max_backward_days,
            "prediction_window_days": prediction_window_days,
            "message": "No article timing data available to assess horizon coverage.",
        }

    lookback_span = max(ages) - min(ages) + 1
    intended = max_backward_days if max_backward_days else prediction_window_days
    flagged = lookback_span < intended

    if flagged:
        msg = (
            f"News lookback is {lookback_span} days but the intended "
            f"backward window was {intended} days "
            f"(sqrt(W) scaling, W={prediction_window_days}), "
            f"signal may be incomplete."
        )
    else:
        msg = (
            f"News lookback ({lookback_span} days) covers the intended "
            f"{intended}-day backward window."
        )

    return {
        "flagged": flagged,
        "lookback_days": lookback_span,
        "intended_lookback_days": intended,
        "prediction_window_days": prediction_window_days,
        "message": msg,
    }


def compute_reliability(
    prediction_result: dict[str, Any],
    herfindahl_index: float,
    merged_articles: list[dict[str, Any]] | None = None,
    prediction_window_days: int = 7,
    max_backward_days: int | None = None,
    flip_set_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run all evidence-quality checks and return an overall rating with flag details."""
    articles_analyzed = prediction_result.get("articles_analyzed", 0)
    normalized_scores = prediction_result.get("normalized_scores", {})

    # Skip label_margin when decision thresholds overrode the raw argmax -
    # the margin between raw scores is irrelevant when the decision was
    # made by per-class threshold gates, not by score ranking.
    abst_test = prediction_result.get("abstention_test", {})
    argmax_label = max(normalized_scores, key=normalized_scores.get) if normalized_scores else "neutral"
    threshold_override = (
        abst_test.get("method") == "decision_threshold"
        and prediction_result.get("final_label") != argmax_label
    )
    if threshold_override:
        sorted_labels = sorted(normalized_scores.items(), key=lambda x: x[1], reverse=True)
        raw_margin = round(sorted_labels[0][1] - sorted_labels[1][1], 4) if len(sorted_labels) >= 2 else 0.0
        label_margin_result = {
            "flagged": False,
            "margin": raw_margin,
            "threshold": XAI_MARGIN_THRESHOLD,
            "message": "Label margin check skipped (decision thresholds overrode raw scores).",
        }
    else:
        label_margin_result = _check_label_margin(normalized_scores)

    flags = {
        "thin_evidence":        _check_thin_evidence(articles_analyzed),
        "weight_concentration": _check_weight_concentration(herfindahl_index),
        "label_margin":         label_margin_result,
        "flip_sensitivity":     _check_flip_sensitivity(flip_set_data),
        "source_diversity":     _check_source_diversity(merged_articles or []),
        "horizon_coverage":     _check_horizon_coverage(
            merged_articles or [], prediction_window_days, max_backward_days
        ),
    }

    flags_triggered = sum(1 for f in flags.values() if f["flagged"])

    if flags_triggered <= 1:
        overall = "HIGH"
    elif flags_triggered == 2:
        overall = "MEDIUM"
    else:
        overall = "LOW"

    flagged_messages = [f["message"] for f in flags.values() if f["flagged"]]
    if flagged_messages:
        summary = f"Prediction has {overall} evidence quality: " + " ".join(flagged_messages)
    else:
        summary = (
            f"Prediction has HIGH evidence quality: {articles_analyzed} articles analyzed, "
            f"prediction is robust and horizon coverage is sufficient."
        )

    logger.info("Evidence quality: %s (%d flags)", overall, flags_triggered)

    return {
        "overall_reliability": overall,
        "flags_triggered": flags_triggered,
        "flags": flags,
        "summary_message": summary,
    }
