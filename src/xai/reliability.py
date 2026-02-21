from __future__ import annotations

import logging
from typing import Any

from config import (
    XAI_THIN_EVIDENCE_THRESHOLD,
    XAI_CONCENTRATION_THRESHOLD,
    XAI_MARGIN_THRESHOLD,
    XAI_LOW_CONFIDENCE_THRESHOLD,
    XAI_SOURCE_CONCENTRATION_THRESHOLD,
    XAI_MIN_UNIQUE_SOURCES,
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
            f"Below reliability threshold ({final_confidence:.3f} < {threshold})."
            if flagged
            else f"Above reliability threshold ({final_confidence:.3f} ≥ {threshold})."
        ),
    }


def _check_source_diversity(
    merged_articles: list[dict[str, Any]],
) -> dict[str, Any]:
    domains: list[str] = []
    for art in merged_articles:
        domain = art.get("domain", "") or art.get("input_source", "unknown")
        domains.append(domain.lower().strip())

    unique = set(domains)
    n_unique = len(unique)
    total = len(domains) or 1

    # Top-domain share
    from collections import Counter
    counts = Counter(domains)
    top_domain, top_count = counts.most_common(1)[0] if counts else ("unknown", 0)
    top_share = round(top_count / total, 4)

    too_few = n_unique < XAI_MIN_UNIQUE_SOURCES
    too_concentrated = top_share > XAI_SOURCE_CONCENTRATION_THRESHOLD
    flagged = too_few or too_concentrated

    if flagged:
        parts = []
        if too_few:
            parts.append(f"only {n_unique} unique source(s)")
        if too_concentrated:
            parts.append(f"top domain '{top_domain}' has {top_share * 100:.0f}% of articles")
        msg = "Source diversity concern: " + "; ".join(parts) + "."
    else:
        msg = f"{n_unique} unique sources, top domain share {top_share * 100:.0f}%."

    return {
        "flagged": flagged,
        "unique_sources": n_unique,
        "top_domain": top_domain,
        "top_domain_share": top_share,
        "message": msg,
    }


def _check_timing_alignment(
    merged_articles: list[dict[str, Any]],
    prediction_window_days: int,
) -> dict[str, Any]:
    # Check whether any timezone / market-close alignment was applied.
    # Currently the pipeline uses UTC timestamps without market-close cutoff,
    # so we flag this as a known limitation and report article age spread.
    ages = [a.get("days_ago", 0) for a in merged_articles]
    if not ages:
        return {
            "flagged": True,
            "message": "No article timing data available.",
            "market_close_aligned": False,
            "oldest_days": 0,
            "newest_days": 0,
        }

    oldest = max(ages)
    newest = min(ages)

    # Market close alignment is not currently implemented — always flag as limitation
    flagged = True

    if oldest > prediction_window_days:
        msg = (
            f"Articles span {newest}–{oldest} days old; oldest exceeds the "
            f"{prediction_window_days}-day window and is down-weighted by recency."
        )
    else:
        msg = (
            f"Most articles are {newest}–{oldest} days old, "
            f"down-weighted by the recency function where applicable."
        )

    msg += " Market-close time alignment is not applied (UTC timestamps used)."

    return {
        "flagged": flagged,
        "market_close_aligned": False,
        "oldest_days": oldest,
        "newest_days": newest,
        "prediction_window_days": prediction_window_days,
        "message": msg,
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

    lookback_span = max(ages) - min(ages)
    # The intended lookback comes from the √W scaling algorithm in the fetcher.
    # Compare actual span against the intended window, not the forecast horizon.
    intended = max_backward_days if max_backward_days else prediction_window_days
    flagged = lookback_span < intended

    if flagged:
        msg = (
            f"News lookback is {lookback_span} days but the intended "
            f"backward window was {intended} days "
            f"(√W scaling, W={prediction_window_days}), "
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
) -> dict[str, Any]:
    articles_analyzed = prediction_result.get("articles_analyzed", 0)
    normalized_scores = prediction_result.get("normalized_scores", {})
    final_confidence = prediction_result.get("final_confidence", 0.0)

    flags = {
        "thin_evidence":        _check_thin_evidence(articles_analyzed),
        "weight_concentration": _check_weight_concentration(herfindahl_index),
        "label_margin":         _check_label_margin(normalized_scores),
        "low_confidence":       _check_low_confidence(final_confidence),
        "source_diversity":     _check_source_diversity(merged_articles or []),
        "timing_alignment":     _check_timing_alignment(
            merged_articles or [], prediction_window_days
        ),
        "horizon_coverage":     _check_horizon_coverage(
            merged_articles or [], prediction_window_days, max_backward_days
        ),
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
