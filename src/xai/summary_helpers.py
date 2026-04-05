"""Shared data-extraction helpers for XAI summary rendering.

Both the terminal report (explainer.py) and the Streamlit UI (app.py) call
these functions so that prediction logic, threshold-override detection, and
evidence-quality summaries are computed in exactly one place.
"""

from __future__ import annotations

from typing import Any


def get_prediction_info(result: dict[str, Any]) -> dict[str, Any]:
    """Extract core prediction fields and detect threshold overrides."""
    pred = result.get("prediction_summary", {})
    meta = result.get("meta", {})
    scores = pred.get("normalized_scores", {})

    label = pred.get("final_label", "unknown")
    confidence = pred.get("final_confidence", 0.0)

    abst_test = pred.get("abstention_test") or {}
    abst_method = abst_test.get("method", "none")
    tau_pos = abst_test.get("decision_thresholds", {}).get("tau_pos", 0)
    tau_neg = abst_test.get("decision_thresholds", {}).get("tau_neg", 0)

    argmax_label = max(scores, key=scores.get) if scores else "neutral"

    threshold_neutral = (
        label == "neutral" and abst_method == "decision_threshold"
    )
    threshold_override = (
        abst_method == "decision_threshold" and label != argmax_label
    )

    return {
        "label": label,
        "confidence": confidence,
        "scores": scores,
        "n_articles": meta.get("articles_analyzed", 0),
        "window": meta.get("prediction_window_days", "?"),
        "lookback": meta.get("news_lookback", "?"),
        "company": meta.get("query", "the company"),
        "ticker": meta.get("ticker", ""),
        "total_weight": pred.get("total_weight", 0.0),
        "threshold_neutral": threshold_neutral,
        "threshold_override": threshold_override,
        "argmax_label": argmax_label,
        "tau_pos": tau_pos,
        "tau_neg": tau_neg,
        "abst_test": abst_test,
    }


def get_contrastive_info(result: dict[str, Any]) -> dict[str, Any]:
    """Extract contrastive analysis data with threshold-override detection."""
    layer2 = result.get("layer_2_article", {})
    contrastive = layer2.get("contrastive", {})

    if not contrastive:
        return {"has_data": False}

    winner = contrastive.get("winner", "")
    runner = contrastive.get("runner_up", "")
    gap = contrastive.get("score_gap", 0)
    is_threshold_override = gap < 0

    pred_info = get_prediction_info(result)
    tau_pos = pred_info["tau_pos"]
    tau_neg = pred_info["tau_neg"]

    return {
        "has_data": bool(winner and runner),
        "winner": winner,
        "runner_up": runner,
        "score_gap": gap,
        "winner_score": contrastive.get("winner_score", 0),
        "runner_up_score": contrastive.get("runner_up_score", 0),
        "n_favouring_winner": contrastive.get("n_favouring_winner", 0),
        "n_favouring_runner_up": contrastive.get("n_favouring_runner_up", 0),
        "total_push_toward_winner": contrastive.get("total_push_toward_winner", 0),
        "total_push_toward_runner_up": contrastive.get("total_push_toward_runner_up", 0),
        "top_gap_drivers": contrastive.get("top_gap_drivers", []),
        "third_place": contrastive.get("third_place"),
        "third_score": contrastive.get("third_score", 0),
        "neutral_effect": contrastive.get("neutral_effect", "none"),
        "is_threshold_override": is_threshold_override,
        "tau_pos": tau_pos,
        "tau_neg": tau_neg,
    }


def get_top_articles(result: dict[str, Any], n: int = 3) -> list[dict[str, Any]]:
    """Return the top N ranked articles with title, sentiment, and weight share."""
    layer2 = result.get("layer_2_article", {})
    ranked = layer2.get("ranked_articles", [])
    return [
        {
            "title": a.get("title", "?"),
            "sentiment": a.get("dominant_sentiment", "?"),
            "weight_share": a.get("weight_share", 0),
            "final_weight": a.get("final_weight", 0),
        }
        for a in ranked[:n]
    ]


def get_sentiment_counts(result: dict[str, Any]) -> dict[str, Any]:
    """Count articles by dominant sentiment from ranked articles."""
    layer2 = result.get("layer_2_article", {})
    ranked = layer2.get("ranked_articles", [])

    counts: dict[str, int] = {}
    for a in ranked:
        s = a.get("dominant_sentiment", "unknown")
        counts[s] = counts.get(s, 0) + 1

    return {
        "counts": counts,
        "total": len(ranked),
    }


def get_reliability_info(result: dict[str, Any]) -> dict[str, Any]:
    """Extract evidence-quality level, flagged concerns, and caution parts."""
    reliability = result.get("reliability", {})
    overall = reliability.get("overall_reliability", "UNKNOWN")
    flags = reliability.get("flags", {})
    flags_triggered = reliability.get("flags_triggered", 0)

    flagged_names = [
        name for name, info in flags.items()
        if info.get("flagged")
    ]

    # Build human-readable caution parts (shared between terminal and UI)
    caution_parts: list[str] = []
    for name, info in flags.items():
        if not info.get("flagged"):
            continue
        if name == "thin_evidence":
            caution_parts.append("thin evidence (few articles)")
        elif name == "weight_concentration":
            caution_parts.append("weight concentrated in one article")
        elif name == "label_margin":
            caution_parts.append(f"narrow decision margin ({info.get('margin', 0):.3f})")
        elif name == "flip_sensitivity":
            caution_parts.append(
                f"prediction is sensitive (removing {info.get('flip_set_size', '?')} "
                f"of {info.get('articles_total', '?')} articles would change the label)"
            )
        elif name == "source_diversity":
            caution_parts.append(
                f"source concentration ({info.get('top_domain', '?')} "
                f"has {info.get('top_domain_share', 0) * 100:.0f}% of articles)"
            )
        elif name == "horizon_coverage":
            caution_parts.append(
                f"news lookback ({info.get('lookback_days', '?')} days) "
                f"is shorter than the intended window "
                f"({info.get('intended_lookback_days', '?')} days)"
            )

    return {
        "overall": overall,
        "flags_triggered": flags_triggered,
        "flagged_names": flagged_names,
        "caution_parts": caution_parts,
        "flags": flags,
    }


def get_robustness_info(result: dict[str, Any]) -> dict[str, Any]:
    """Extract flip-set and label-flipping robustness data."""
    layer2 = result.get("layer_2_article", {})
    flip_set = layer2.get("minimum_flip_set", {})
    flip_articles = layer2.get("label_flipping_articles", [])

    has_single_flippers = bool(flip_articles)
    flip_possible = flip_set.get("flip_possible", False)

    return {
        "has_single_flippers": has_single_flippers,
        "single_flip_count": len(flip_articles),
        "single_flip_titles": flip_articles,
        "flip_possible": flip_possible,
        "flip_set_size": flip_set.get("flip_set_size", 0),
        "articles_total": flip_set.get("articles_total", 0),
        "new_label": flip_set.get("new_label", "?"),
        "flip_set_titles": flip_set.get("flip_set_titles", []),
        "is_robust": not has_single_flippers and not flip_possible,
    }


def get_storylines_info(result: dict[str, Any], n: int = 4) -> list[dict[str, Any]]:
    """Return the top N storyline clusters with label, sentiment, and count."""
    storylines_data = result.get("storylines", {})
    storylines = storylines_data.get("storylines", [])
    return [
        {
            "label": sl.get("label") or sl.get("keyword_label", "?"),
            "sentiment_group": sl.get("sentiment_group", "?"),
            "articles_count": sl.get("articles_count", 0),
            "contribution_score": sl.get("contribution_score", 0),
            "sentiment": sl.get("sentiment", {}),
            "top_titles": sl.get("top_titles", []),
        }
        for sl in storylines[:n]
    ]
