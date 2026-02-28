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


# Known news aggregators — these collect articles from many independent
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
    from collections import Counter

    domains: list[str] = []
    for art in merged_articles:
        domain = art.get("domain", "") or art.get("input_source", "unknown")
        domains.append(domain.lower().strip())

    unique = set(domains)
    n_unique = len(unique)
    total = len(domains) or 1

    # Top-domain share
    counts = Counter(domains)
    top_domain, top_count = counts.most_common(1)[0] if counts else ("unknown", 0)
    top_share = round(top_count / total, 4)

    # Exclude aggregators from the concentration check — they host articles
    # from many independent editorial desks, so "68% from Yahoo" does NOT
    # indicate a single-viewpoint problem.
    non_agg_domains = [d for d in domains if d not in _AGGREGATOR_DOMAINS]
    n_non_agg = len(non_agg_domains) or 1
    non_agg_counts = Counter(non_agg_domains)
    if non_agg_counts:
        top_editorial, top_ed_count = non_agg_counts.most_common(1)[0]
        top_editorial_share = round(top_ed_count / n_non_agg, 4)
    else:
        top_editorial, top_editorial_share = "none", 0.0

    n_unique_editorial = len(set(non_agg_domains)) if non_agg_domains else 0

    # Flag only on editorial (non-aggregator) concentration
    too_few = n_unique_editorial < XAI_MIN_UNIQUE_SOURCES and n_unique < XAI_MIN_UNIQUE_SOURCES
    too_concentrated = top_editorial_share > XAI_SOURCE_CONCENTRATION_THRESHOLD
    flagged = too_few or too_concentrated

    # Count how many articles come through aggregators
    n_aggregator = sum(1 for d in domains if d in _AGGREGATOR_DOMAINS)

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
        agg_note = ""
        if n_aggregator > 0:
            agg_pct = round(n_aggregator / total * 100)
            agg_note = (
                f" ({agg_pct}% via aggregators like Yahoo/Finnhub"
                f" — these collect from many editorial sources)"
            )
        msg = (
            f"{n_unique} sources ({n_unique_editorial} editorial + "
            f"{len([d for d in unique if d in _AGGREGATOR_DOMAINS])} aggregators), "
            f"top editorial share {top_editorial_share * 100:.0f}%."
            f"{agg_note}"
        )

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
    """Check whether market-close timestamp alignment was applied."""
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

    # Check if articles have market_date (indicates ET alignment was applied)
    has_market_date = any(a.get("market_date") for a in merged_articles)

    if has_market_date:
        flagged = False
        if oldest > prediction_window_days:
            msg = (
                    f"Articles span {newest}-{oldest} days old (ET market-close aligned); "
                f"oldest exceeds the {prediction_window_days}-day window and is "
                f"down-weighted by recency."
            )
        else:
            msg = (
                f"Articles are {newest}-{oldest} days old (ET market-close aligned), "
                f"down-weighted by the recency function where applicable."
            )
    else:
        # Fallback: no market_date → still UTC, flag it
        flagged = True
        if oldest > prediction_window_days:
            msg = (
                f"Articles span {newest}-{oldest} days old; oldest exceeds the "
                f"{prediction_window_days}-day window and is down-weighted by recency."
            )
        else:
            msg = (
                f"Most articles are {newest}-{oldest} days old, "
                f"down-weighted by the recency function where applicable."
            )
        msg += " Market-close time alignment is not applied (UTC timestamps used)."

    return {
        "flagged": flagged,
        "market_close_aligned": has_market_date,
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

    lookback_span = max(ages) - min(ages) + 1   # +1 for inclusive day counting
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
