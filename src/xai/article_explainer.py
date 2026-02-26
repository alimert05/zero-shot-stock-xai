from __future__ import annotations

import logging
from typing import Any

from .utils import herfindahl_index, safe_round, get_dominant_label

logger = logging.getLogger(__name__)


def _compute_contribution_share(
    article: dict[str, Any],
    total_weight: float,
) -> dict[str, float]:
    weighted_scores = article.get("weighted_scores", {})
    if total_weight == 0:
        return {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    return {
        label: safe_round(score / total_weight)
        for label, score in weighted_scores.items()
    }


def _run_counterfactual(
    article: dict[str, Any],
    total_weighted: dict[str, float],
    total_weight: float,
    current_label: str,
) -> dict[str, Any]:
    article_weight = article.get("final_weight", 0.0)
    article_weighted_scores = article.get("weighted_scores", {})

    new_total_weight = total_weight - article_weight
    if new_total_weight <= 0:
        return {
            "new_label": "neutral",
            "new_confidence": 0.0,
            "new_normalized": {"positive": 0.0, "negative": 0.0, "neutral": 0.0},
            "label_would_change": current_label != "neutral",
        }

    new_normalized = {}
    for label in ["positive", "negative", "neutral"]:
        new_w = total_weighted.get(label, 0.0) - article_weighted_scores.get(label, 0.0)
        new_normalized[label] = safe_round(new_w / new_total_weight)

    new_label = max(new_normalized, key=new_normalized.get)
    new_confidence = safe_round(new_normalized[new_label])

    return {
        "new_label": new_label,
        "new_confidence": new_confidence,
        "new_normalized": new_normalized,
        "label_would_change": new_label != current_label,
    }


# ── Contrastive score-gap attribution ──────────────────────────────
#
# For each article compute a "net direction" score that measures how
# much it pushes the *winning* label ahead of the *runner-up*.
#
#   net_direction = (weighted_winner − weighted_runner_up) / total_weight
#
# Positive net_direction → article favours the winner.
# Negative net_direction → article favours the runner-up.
# Summing across all articles gives the aggregate margin.
#
# This answers the contrastive question "why label A *instead of*
# label B?" by attributing the score gap to individual articles.
#
# Ref: Miller (2019) "Explanation in Artificial Intelligence:
#      Insights from the Social Sciences", Art. Intell. 267, 1-38.

def _compute_contrastive(
    merged_articles: list[dict[str, Any]],
    prediction_result: dict[str, Any],
) -> dict[str, Any]:
    """Attribute the score gap between winner and runner-up to articles."""
    normalized = prediction_result.get("normalized_scores", {})
    total_weight = prediction_result.get("total_weight", 0.0)
    current_label = prediction_result.get("final_label", "neutral")

    sorted_labels = sorted(normalized, key=normalized.get, reverse=True)
    winner = sorted_labels[0]
    runner_up = sorted_labels[1]
    third_place = sorted_labels[2] if len(sorted_labels) > 2 else None

    winner_score = normalized.get(winner, 0.0)
    runner_up_score = normalized.get(runner_up, 0.0)
    third_score = normalized.get(third_place, 0.0) if third_place else 0.0
    score_gap = round(winner_score - runner_up_score, 4)

    # Per-article net direction (contribution to the gap)
    article_contributions: list[dict[str, Any]] = []
    for article in merged_articles:
        title = article.get("title", "")
        ws = article.get("weighted_scores", {})
        w_winner = ws.get(winner, 0.0)
        w_runner = ws.get(runner_up, 0.0)

        if total_weight > 0:
            net_direction = safe_round((w_winner - w_runner) / total_weight)
        else:
            net_direction = 0.0

        article_contributions.append({
            "title": title,
            "net_direction": net_direction,
            "weighted_winner": safe_round(w_winner),
            "weighted_runner_up": safe_round(w_runner),
            "favours": winner if net_direction >= 0 else runner_up,
        })

    # Sort by absolute net direction descending → biggest gap drivers first
    article_contributions.sort(key=lambda a: abs(a["net_direction"]), reverse=True)

    # Split into articles favouring winner vs runner-up
    favouring_winner = [a for a in article_contributions if a["net_direction"] >= 0]
    favouring_runner = [a for a in article_contributions if a["net_direction"] < 0]

    total_push_winner = safe_round(sum(a["net_direction"] for a in favouring_winner))
    total_push_runner = safe_round(sum(a["net_direction"] for a in favouring_runner))

    logger.info(
        "Contrastive: %s over %s by %.4f  "
        "(%d articles favour winner, %d favour runner-up)",
        winner, runner_up, score_gap,
        len(favouring_winner), len(favouring_runner),
    )

    # Analyse neutral's role: does it pull evenly from both sides or favour one?
    neutral_effect = "none"
    if third_place:
        # How much neutral "absorbed" from each side per article
        winner_vs_third = []
        runner_vs_third = []
        for article in merged_articles:
            ws = article.get("weighted_scores", {})
            if total_weight > 0:
                winner_vs_third.append(
                    (ws.get(winner, 0.0) - ws.get(third_place, 0.0)) / total_weight
                )
                runner_vs_third.append(
                    (ws.get(runner_up, 0.0) - ws.get(third_place, 0.0)) / total_weight
                )
        avg_winner_gap = sum(winner_vs_third) / len(winner_vs_third) if winner_vs_third else 0
        avg_runner_gap = sum(runner_vs_third) / len(runner_vs_third) if runner_vs_third else 0

        # If third class pulls more from runner-up, it helps the winner
        if abs(avg_winner_gap - avg_runner_gap) < 0.01:
            neutral_effect = "balanced"
        elif avg_runner_gap < avg_winner_gap:
            neutral_effect = f"helps_{winner}"
        else:
            neutral_effect = f"helps_{runner_up}"

    return {
        "winner": winner,
        "runner_up": runner_up,
        "third_place": third_place,
        "winner_score": winner_score,
        "runner_up_score": runner_up_score,
        "third_score": third_score,
        "score_gap": score_gap,
        "total_push_toward_winner": total_push_winner,
        "total_push_toward_runner_up": total_push_runner,
        "n_favouring_winner": len(favouring_winner),
        "n_favouring_runner_up": len(favouring_runner),
        "neutral_effect": neutral_effect,
        "top_gap_drivers": article_contributions[:5],
        "all_contributions": article_contributions,
    }


# ── Minimum flip set (greedy) ──────────────────────────────────────
#
# Find the smallest set of articles whose removal would flip the
# predicted label.  We use a greedy algorithm: iteratively remove
# the article with the highest net_direction toward the winner until
# the runner-up overtakes.
#
# This is the counterfactual "what would need to change?" answer.
#
# Ref: Wachter et al. (2017) "Counterfactual Explanations Without
#      Opening the Black Box", Harvard JL & Tech. 31(2).

def _compute_minimum_flip_set(
    merged_articles: list[dict[str, Any]],
    prediction_result: dict[str, Any],
) -> dict[str, Any]:
    """Greedy search for the fewest articles to remove to flip the label."""
    total_weighted = dict(prediction_result.get("weighted_scores", {}))
    total_weight = prediction_result.get("total_weight", 0.0)
    normalized = prediction_result.get("normalized_scores", {})
    current_label = prediction_result.get("final_label", "neutral")

    sorted_labels = sorted(normalized, key=normalized.get, reverse=True)
    winner = sorted_labels[0]
    runner_up = sorted_labels[1]

    # Sort articles by how much they favour the winner (most helpful first)
    scored = []
    for article in merged_articles:
        ws = article.get("weighted_scores", {})
        net = ws.get(winner, 0.0) - ws.get(runner_up, 0.0)
        scored.append((net, article))
    scored.sort(key=lambda x: x[0], reverse=True)

    # Greedily remove articles that most favour the winner
    remaining_weighted = {k: v for k, v in total_weighted.items()}
    remaining_weight = total_weight
    flip_set: list[str] = []

    for net, article in scored:
        if net <= 0:
            break  # remaining articles favour runner-up, removing them won't help

        aw = article.get("final_weight", 0.0)
        aws = article.get("weighted_scores", {})

        remaining_weight -= aw
        for label in remaining_weighted:
            remaining_weighted[label] -= aws.get(label, 0.0)

        flip_set.append(article.get("title", ""))

        if remaining_weight <= 0:
            break

        new_norm = {
            k: v / remaining_weight for k, v in remaining_weighted.items()
        }
        new_label = max(new_norm, key=new_norm.get)
        if new_label != current_label:
            logger.info(
                "Minimum flip set: %d articles → label changes from %s to %s",
                len(flip_set), current_label, new_label,
            )
            return {
                "flip_possible": True,
                "flip_set_size": len(flip_set),
                "flip_set_titles": flip_set,
                "new_label": new_label,
                "articles_total": len(merged_articles),
            }

    return {
        "flip_possible": False,
        "flip_set_size": None,
        "flip_set_titles": [],
        "new_label": None,
        "articles_total": len(merged_articles),
    }


def explain_articles(
    merged_articles: list[dict[str, Any]],
    prediction_result: dict[str, Any],
) -> dict[str, Any]:
    total_weight = prediction_result.get("total_weight", 0.0)
    total_weighted = prediction_result.get("weighted_scores", {})
    current_label = prediction_result.get("final_label", "neutral")

    # Sort by final_weight descending
    sorted_articles = sorted(
        merged_articles,
        key=lambda a: a.get("final_weight", 0.0),
        reverse=True,
    )

    ranked = []
    label_flipping = []
    positive_drivers: list[tuple[str, float]] = []
    negative_drivers: list[tuple[str, float]] = []

    all_weights = [a.get("final_weight", 0.0) for a in merged_articles]
    hhi = herfindahl_index(all_weights)

    for rank, article in enumerate(sorted_articles, start=1):
        title = article.get("title", "")
        final_weight = article.get("final_weight", 0.0)
        weight_share = safe_round(final_weight / total_weight) if total_weight > 0 else 0.0
        raw_scores = article.get("raw_scores", {})
        weighted_scores = article.get("weighted_scores", {})
        dominant = get_dominant_label(raw_scores) if raw_scores else current_label

        contribution_share = _compute_contribution_share(article, total_weight)
        counterfactual = _run_counterfactual(article, total_weighted, total_weight, current_label)

        if counterfactual["label_would_change"]:
            label_flipping.append(title)

        pos_contrib = contribution_share.get("positive", 0.0)
        neg_contrib = contribution_share.get("negative", 0.0)
        positive_drivers.append((title, pos_contrib))
        negative_drivers.append((title, neg_contrib))

        ranked.append({
            "rank": rank,
            "title": title,
            "final_weight": safe_round(final_weight),
            "weight_share": weight_share,
            "dominant_sentiment": dominant,
            "raw_scores": {k: safe_round(v) for k, v in raw_scores.items()},
            "weighted_scores": {k: safe_round(v) for k, v in weighted_scores.items()},
            "contribution_share": contribution_share,
            "counterfactual": counterfactual,
        })

    top_positive = [t for t, _ in sorted(positive_drivers, key=lambda x: x[1], reverse=True)[:3]]
    top_negative = [t for t, _ in sorted(negative_drivers, key=lambda x: x[1], reverse=True)[:3]]

    # Contrastive analysis: why winner instead of runner-up?
    contrastive = _compute_contrastive(merged_articles, prediction_result)

    # Minimum flip set: what would need to change?
    minimum_flip = _compute_minimum_flip_set(merged_articles, prediction_result)

    logger.info(
        "Article explanation complete: %d articles, %d label-flipping, HHI=%.4f, "
        "flip_set_size=%s",
        len(ranked), len(label_flipping), hhi,
        minimum_flip.get("flip_set_size", "N/A"),
    )

    return {
        "ranked_articles": ranked,
        "label_flipping_articles": label_flipping,
        "top_positive_drivers": top_positive,
        "top_negative_drivers": top_negative,
        "weight_concentration": round(hhi, 4),
        "contrastive": contrastive,
        "minimum_flip_set": minimum_flip,
    }
