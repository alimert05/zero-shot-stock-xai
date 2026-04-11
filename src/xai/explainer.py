"""
explainer.py -- Main XAI orchestrator for the sentiment analysis pipeline.

Coordinates all explanation layers (token-level LIME, article-level
counterfactuals, pipeline-level aggregation, narrative clustering) and
generates a human-readable summary report with supporting charts.
"""
# Imports and Constants

from __future__ import annotations

import json
import logging
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    JSON_PATH, XAI_OUTPUT_PATH, XAI_SUMMARY_PATH, XAI_LIME_TOP_N,
    XAI_CONCENTRATION_THRESHOLD,
)

from .article_explainer  import explain_articles
from .pipeline_explainer import explain_pipeline
from .reliability        import compute_reliability
from .token_explainer    import explain_tokens
from .narrative          import generate_narrative
from .charts              import generate_all_charts
from .narrative_cluster   import cluster_narratives
from .utils               import build_lime_noise_set, is_lime_noise_token

logger = logging.getLogger(__name__)


# Data Loading and Merging

def _load_articles(articles_json_path: str) -> dict[str, Any]:
    """Load the articles JSON file from disk and return its contents as a dict."""
    with open(articles_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _merge_article_data(
    prediction_result: dict[str, Any],
    articles_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Merge prediction article details with full article records by title."""
    # Build lookup: title -> full article record from articles.json
    article_lookup: dict[str, dict] = {}
    for art in articles_data.get("articles", []):
        title = art.get("title", "")
        if title:
            article_lookup[title] = art

    merged = []
    for detail in prediction_result.get("article_details", []):
        title = detail.get("title", "")
        base = article_lookup.get(title, {})

        merged.append({
            "title":                 title,
            "content":               base.get("content") or base.get("summary") or "",
            "days_ago":              base.get("days_ago", 0),
            "recency_weight":        base.get("recency_weight", 1.0),
            "impact_horizon":        base.get("impact_horizon", {}),
            "impact_horizon_weight": base.get("impact_horizon_weight", 1.0),
            "final_weight":          detail.get("final_weight", 1.0),
            "input_source":          detail.get("input_source", "unknown"),
            "domain":                base.get("domain", ""),
            "raw_scores":            detail.get("raw_scores", {}),
            "weighted_scores":       detail.get("weighted_scores", {}),
            "content_stats":         base.get("content_stats", {}),
            "market_date":           base.get("market_date", ""),
            "seendate_et":           base.get("seendate_et", ""),
        })

    return merged


# File I/O

def _save_result(result: dict[str, Any], output_path: str) -> None:
    """Write the full XAI result dict to a JSON file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    logger.info("XAI result saved to %s", output_path)


# ASCII Visualisation Helpers

def _ascii_bar(value: float, max_value: float, width: int = 30,
               fill: str = "\u2588", empty: str = "\u2591") -> str:
    """Fixed-width filled bar proportional to value/max_value."""
    if max_value <= 0:
        return empty * width
    filled = max(0, min(int(round(value / max_value * width)), width))
    return fill * filled + empty * (width - filled)


def _wrap(text: str, width: int = 56, indent: str = "  ") -> list[str]:
    """Word-wrap text and prefix each line with indent."""
    return [f"{indent}{line}" for line in textwrap.wrap(text, width=width)]


# Summary Text Builder

def _build_summary_text(result: dict[str, Any], chart_paths: dict | None = None) -> str:  # noqa: C901
    """Build the human-readable plain-text XAI summary report."""
    W = "=" * 60
    w = "-" * 60
    P = "\u2550" * 60          # part divider (double line)

    from xai.summary_helpers import (
        get_prediction_info, get_contrastive_info, get_reliability_info,
        get_robustness_info, get_sentiment_counts,
    )

    meta        = result["meta"]
    pred        = result["prediction_summary"]
    reliability = result["reliability"]
    layer1      = result["layer_1_token"]
    layer2      = result["layer_2_article"]
    layer3      = result["layer_3_pipeline"]
    narrative   = result["narrative"]
    storylines  = result.get("storylines", {})

    # Shared helpers - single source of truth for threshold/contrastive/reliability
    pred_info = get_prediction_info(result)
    cont_info = get_contrastive_info(result)
    rel_info  = get_reliability_info(result)
    rob_info  = get_robustness_info(result)
    sent_info = get_sentiment_counts(result)

    # Precompute shared variables so sections can appear in any order

    # Scores
    ns            = pred["normalized_scores"] or {}
    neutral_score = ns.get("neutral", 0.0)

    # Threshold-triggered neutral detection (from shared helper)
    abst_test = pred_info["abst_test"]
    threshold_neutral = pred_info["threshold_neutral"]
    pos_pct = ns.get("positive", 0.0) * 100
    neg_pct = ns.get("negative", 0.0) * 100
    neu_pct = ns.get("neutral", 0.0) * 100

    # Contrastive / flip-set
    contrastive = layer2.get("contrastive", {})
    flip_info   = layer2.get("minimum_flip_set", {})

    # Ranked articles + sentiment distribution (from shared helper)
    ranked_arts = layer2.get("ranked_articles", [])
    sent_counts = sent_info["counts"]
    total_arts = sent_info["total"] or 1

    # LIME articles + noise set
    lime_articles  = layer1.get("articles", [])
    all_art_titles = [a.get("title", "") for a in ranked_arts]
    noise_set = build_lime_noise_set(
        meta.get("query", ""), meta.get("ticker", ""),
        article_titles=all_art_titles,
    )

    # Evidence-quality flags (from shared helper)
    overall_rel = rel_info["overall"]
    flags       = rel_info["flags"]
    rel_icon    = {"HIGH": "\u2713", "MEDIUM": "!", "LOW": "\u2717"}.get(overall_rel, "?")

    thin_flag    = flags.get("thin_evidence", {}).get("flagged", False)
    conc_flag    = flags.get("weight_concentration", {}).get("flagged", False)
    margin_flag  = flags.get("label_margin", {}).get("flagged", False)
    flip_flag    = flags.get("flip_sensitivity", {}).get("flagged", False)
    source_flag  = flags.get("source_diversity", {}).get("flagged", False)
    horizon_flag = flags.get("horizon_coverage", {}).get("flagged", False)

    # Terminal-specific verbose caution parts (more detail than UI)
    caution_parts: list[str] = []
    if thin_flag:
        caution_parts.append("the prediction is based on very few articles")
    if conc_flag:
        caution_parts.append(
            "weight is concentrated in one article, "
            "making the result sensitive to that single source"
        )
    if margin_flag:
        caution_parts.append(
            "the positive and negative scores are very close - "
            "a small shift in news could change the verdict"
        )
    if flip_flag:
        fs = flags.get("flip_sensitivity", {})
        caution_parts.append(
            f"prediction is sensitive: removing {fs.get('flip_set_size', '?')} of "
            f"{fs.get('articles_total', '?')} articles would change the label"
        )
    if source_flag:
        caution_parts.append(
            "most articles come from the same editorial source, "
            "reducing editorial independence"
        )
    if horizon_flag:
        hc = flags.get("horizon_coverage", {})
        caution_parts.append(
            f"news lookback ({hc.get('lookback_days', '?')} days) is shorter than "
            f"the intended backward window ({hc.get('intended_lookback_days', '?')} days, "
            f"sqrt(W) scaling), signal may be incomplete"
        )

    # Chart label map (used in Charts + Advanced sections)
    chart_label_map = {
        "sentiment_scores":        "Sentiment score bar chart",
        "article_distribution":    "Article sentiment pie chart",
        "article_weights":         "Top-10 article weight chart",
        "horizon_breakdown":       "Timing horizon breakdown chart",
        "lime_tokens":             "Word-level attribution (LIME) chart",
        "reliability":             "Evidence quality dashboard",
        "storyline_contribution":  "Narrative storyline contribution chart",
        "contrastive_waterfall":   "Contrastive waterfall (why A not B?)",
        "article_timeline":        "Article timeline (recency vs influence)",
        "cumulative_score":        "Cumulative score build-up chart",
    }

    lines: list[str] = []


    #  HEADER

    ticker_str = f"  ({meta['ticker']})" if meta['ticker'] else ""
    lines += [
        W,
        "  SENTIMENT ANALYSIS EXPLANATION REPORT",
        W,
        f"  Company           : {meta['query']}{ticker_str}",
        f"  Forecast horizon  : {meta['prediction_window_days']} days",
        f"  News lookback     : {meta.get('news_lookback', 'N/A')}",
        f"  Articles analysed : {meta['articles_analyzed']}",
        f"  Generated at      : {meta['xai_timestamp']}",
        W,
    ]

    #  PART 1 - SUMMARY and KEY FINDINGS
    #  Plain-English explanation of the result and its context

    lines += [
        "",
        P,
        "  PART 1 - SUMMARY & KEY FINDINGS",
        "  Plain-English explanation of what the model decided and why",
        P,
        "",
    ]

    # Prediction result + score chart
    lines += [
        "  PREDICTION RESULT",
        "  Sentiment scores across all analysed articles",
        w,
        f"  Ground truth : close-to-close return over "
        f"{meta['prediction_window_days']}-day forecast horizon, "
        f"neutral band scaled by EWMA volatility (per-stock, per-period).",
        "  Disclaimer   : Model support measures how much weight the model",
        "                  assigns to the winning label relative to alternatives.",
        "                  It is NOT a calibrated probability of future return",
        "                  and should not be read as '41% chance the price drops'.",
        "",
        "  Score chart  (each \u2588 \u2248 2.5%):",
        "",
    ]
    for label in ["positive", "negative", "neutral"]:
        score  = ns.get(label, 0.0)
        bar    = _ascii_bar(score, 1.0, width=40)
        marker = "  < PREDICTED" if label == pred["final_label"] else ""
        lines.append(f"    {label:>8} : {score * 100:5.1f}%  {bar}{marker}")
    if threshold_neutral:
        verdict_label = "NEUTRAL - Mixed Sentiment"
        tau_pos_val = abst_test.get("decision_thresholds", {}).get("tau_pos", 0.0)
        tau_neg_val = abst_test.get("decision_thresholds", {}).get("tau_neg", 0.0)
        lines += [
            "",
            f"  Verdict     : {verdict_label}",
            f"  Model support : {pred['final_confidence'] * 100:.1f}%"
            "  (share of model weight, not a probability)",
            "",
            "  +- Why Neutral? -------------------------------------+",
            f"  | Articles show divided sentiment:                   |",
            f"  | Positive: {pos_pct:5.1f}%                          |",
            f"  | Negative: {neg_pct:5.1f}%                          |",
            f"  |                                                    |",
            f"  | Neither direction reaches the consensus threshold  |",
            f"  | required for a directional prediction:             |",
            f"  |   Positive needs >= {tau_pos_val * 100:.0f}%  (got {pos_pct:.1f}%)              |",
            f"  |   Negative needs >= {tau_neg_val * 100:.0f}%  (got {neg_pct:.1f}%)              |",
            f"  |                                                    |",
            f"  | When positive and negative signals are closely     |",
            f"  | balanced, a directional call would be unreliable.  |",
            f"  | The system classifies this as mixed/inconclusive.  |",
            "  +----------------------------------------------------+",
        ]
    else:
        lines += [
            "",
            f"  Verdict     : {str(pred['final_label']).upper()}",
            f"  Model support : {pred['final_confidence'] * 100:.1f}%"
            "  (share of model weight, not a probability)",
        ]
        if neutral_score == 0.0:
            lines.append(
                "  Note        : Neutral shows 0% because no articles carried"
                " sufficient neutral signal after weighting. The model supports"
                " 3-way classification (positive / negative / neutral)."
            )
    lines.append("")

    # Recommendation 
    if overall_rel == "HIGH":
        action = "This prediction has HIGH evidence quality and can be used with reasonable confidence."
    elif overall_rel == "MEDIUM":
        action = (
            "This prediction has MEDIUM evidence quality - treat it as indicative, not definitive. "
            + ("Specifically: " + "; ".join(caution_parts) + "." if caution_parts else "")
        )
    else:
        action = (
            "This prediction has LOW evidence quality and should NOT be used alone for decisions. "
            + ("Reasons: " + "; ".join(caution_parts) + "." if caution_parts else "")
        )

    lines += [
        "  RECOMMENDATION",
        "  What you should do with this result",
        w,
        f"  {action}",
        "",
    ]

    # Narrative themes 
    event_dist = layer3.get("event_type_distribution", {})
    event_sent = layer3.get("event_type_sentiment", {})
    event_short_labels = {
        "earnings report, guidance, or financial results":                   "Earnings / results ",
        "analyst upgrade, downgrade, or price target revision":              "Analyst action     ",
        "product launch, partnership, contract, or business development":    "Product / business ",
        "share buyback, dividend, stock offering, or debt issuance":         "Capital / payout   ",
        "lawsuit, investigation, regulatory action, or compliance issue":    "Regulatory / legal ",
        "merger, acquisition, takeover, or corporate restructuring":         "Strategy / M&A     ",
        "CEO change, executive departure, or board appointment":             "Leadership change  ",
        "market commentary, sector outlook, or opinion piece":               "General commentary ",
        "financial distress, credit downgrade, or going concern warning":    "Distress / credit  ",
    }
    if event_dist:
        lines += [
            "  NARRATIVE THEMES",
            "  What the news is about and the dominant tone per theme",
            w,
        ]
        total_e = sum(event_dist.values()) or 1
        for etype, ecount in sorted(event_dist.items(), key=lambda x: x[1], reverse=True):
            elabel = event_short_labels.get(etype, f"{etype[:20]:<20}")
            pct = ecount / total_e * 100
            # Per-theme sentiment
            s = event_sent.get(etype, {})
            s_pos = s.get("positive", 0)
            s_neg = s.get("negative", 0)
            s_neu = s.get("neutral", 0)
            dominant_s = max(s, key=s.get) if s else "neutral"
            tone_icon = {"positive": "+", "negative": "-", "neutral": "="}.get(dominant_s, "?")
            lines.append(
                f"    {elabel}: {ecount:>3} articles ({pct:.0f}%)  "
                f"{tone_icon} {dominant_s}  "
                f"(+{s_pos} / -{s_neg} / ={s_neu})"
            )
        lines.append("")

    # Storylines - auto-discovered from article titles via TF-IDF clustering
    # Grouped by sentiment: predicted label first, then opposing, then neutral
    sl_list = storylines.get("storylines", [])
    sl_other = storylines.get("other_count", 0)
    if sl_list:
        predicted_label = pred.get("final_label", "neutral").lower()

        # Group clusters by sentiment_group (article-level, not cluster-dominant)
        sl_by_sent: dict[str, list] = {"positive": [], "negative": [], "neutral": []}
        for sl in sl_list:
            grp = sl.get("sentiment_group", sl.get("sentiment", {}).get("dominant", "neutral"))
            sl_by_sent[grp].append(sl)

        # Within each group, sort by contribution_score (actual prediction impact)
        for group in sl_by_sent.values():
            group.sort(key=lambda s: s.get("contribution_score", 0), reverse=True)

        # Display order: predicted label first, then opposing, then neutral
        # All groups uncapped - full transparency
        if predicted_label == "positive":
            order = ["positive", "negative", "neutral"]
        elif predicted_label == "negative":
            order = ["negative", "positive", "neutral"]
        else:
            order = ["neutral", "positive", "negative"]

        section_headers = {
            predicted_label: f"Storylines supporting the {predicted_label.upper()} prediction",
        }
        for sent_key in ["positive", "negative", "neutral"]:
            if sent_key != predicted_label:
                section_headers[sent_key] = (
                    f"Opposing storylines ({sent_key})"
                    if sent_key != "neutral"
                    else "Neutral storylines"
                )

        lines += [
            "  Storylines (auto-discovered from article titles):",
            "",
        ]

        for sent_key in order:
            group = sl_by_sent[sent_key]
            if not group:
                continue

            lines.append(f"    {section_headers[sent_key]}:")
            for sl in group:
                n = sl["articles_count"]
                sent = sl.get("sentiment", {})
                dom = sent.get("dominant", "neutral")
                s_pos = sent.get("positive", 0)
                s_neg = sent.get("negative", 0)
                s_neu = sent.get("neutral", 0)
                tone_icon = {"positive": "+", "negative": "-", "neutral": "="}.get(dom, "?")
                lines.append(
                    f"      \"{sl['label']}\" ({n} articles)  "
                    f"{tone_icon} {dom}  (+{s_pos} / -{s_neg} / ={s_neu})"
                )
                for t in sl.get("top_titles", []):
                    lines.append(f"        \u2192 {t[:68]}")
            lines.append("")

        if sl_other > 0:
            lines.append(
                f"    Other topics ({sl_other} articles - titles too unique to cluster "
                f"with any other article; each covers a distinct story)"
            )
        lines.append("")

    # Final plain-English overview 
    n_pos = sent_counts.get("positive", 0)
    n_neg = sent_counts.get("negative", 0)
    n_neu = sent_counts.get("neutral", 0)

    top_art       = ranked_arts[0] if ranked_arts else {}
    top_art_title = top_art.get("title", "N/A")
    top_art_sent  = top_art.get("dominant_sentiment", "unknown")
    top_art_share = round(top_art.get("weight_share", 0.0) * 100, 1)

    lime_sentence = ""
    if lime_articles:
        # Try to match the top contrastive gap driver first, fall back to
        # most-influential article so the word-level evidence connects to
        # the "why X instead of Y?" reasoning.
        lime_by_title = {la["title"]: la for la in lime_articles}
        top_gap_title = None
        if contrastive:
            for gd in contrastive.get("top_gap_drivers", []):
                if gd["title"] in lime_by_title:
                    top_gap_title = gd["title"]
                    break

        if top_gap_title and top_gap_title in lime_by_title:
            matching_lime = lime_by_title[top_gap_title]
            source_desc = "the top gap-driving article"
        else:
            matching_lime = lime_by_title.get(top_art_title, lime_articles[0])
            source_desc = "the most influential article"

        top_words = matching_lime.get("top_tokens_supporting", [])
        if top_words:
            lime_sentence = (
                f" At the word level, the strongest signals in {source_desc}"
                f" included: {', '.join(top_words[:5])}."
            )

    horizon_dist = layer3.get("horizon_distribution", {})
    dominant_horizon = max(horizon_dist, key=horizon_dist.get).replace("_", " ").lower() if horizon_dist else "immediate"

    weight_sentence = (
        f"Articles were weighted by recency and prediction-horizon relevance -"
        f" the average article was {layer3['avg_days_ago']} days old,"
        f" with most classified as {dominant_horizon} horizon."
    )

    # Contrastive sentence - why winner instead of runner-up (from shared helper)
    contrastive_sentence = ""
    if cont_info["has_data"]:
        c_winner_ov = cont_info["winner"].upper()
        c_runner_ov = cont_info["runner_up"].upper()
        if cont_info["is_threshold_override"]:
            contrastive_sentence = (
                f" Decision thresholds selected {c_winner_ov} over {c_runner_ov}:"
                f" {c_runner_ov.lower()} scored higher but did not reach its threshold"
                f" ({cont_info['tau_pos'] * 100:.0f}% for positive,"
                f" {cont_info['tau_neg'] * 100:.0f}% for negative),"
                f" while {c_winner_ov.lower()} exceeded its lower threshold."
            )
        else:
            contrastive_sentence = (
                f" The model chose {c_winner_ov} over {c_runner_ov} by a score gap of"
                f" {cont_info['score_gap']:.4f}:"
                f" {cont_info['n_favouring_winner']} articles"
                f" pushed toward {c_winner_ov} while"
                f" {cont_info['n_favouring_runner_up']} pushed toward {c_runner_ov}."
            )

    # Flip set sentence
    flip_sentence = ""
    if flip_info.get("flip_possible"):
        flip_n_ov    = flip_info["flip_set_size"]
        flip_total_ov = flip_info["articles_total"]
        flip_sentence = (
            f" The prediction is sensitive: removing as few as"
            f" {flip_n_ov} of {flip_total_ov} articles would change the verdict."
        )
    elif flip_info:
        flip_sentence = (
            " No subset of articles can flip the verdict - the"
            " prediction is robust across the article pool."
        )

    reliability_sentence = {
        "HIGH":   "The prediction carries HIGH evidence quality and can be used with confidence.",
        "MEDIUM": (
            "The prediction carries MEDIUM evidence quality. "
            + ("; ".join(caution_parts).capitalize() + "." if caution_parts else "")
        ),
        "LOW": (
            "The prediction carries LOW evidence quality and should not be used in isolation. "
            + ("; ".join(caution_parts).capitalize() + "." if caution_parts else "")
        ),
    }.get(overall_rel, "")

    if threshold_neutral:
        tau_pos_ov = abst_test.get("decision_thresholds", {}).get("tau_pos", 0.0)
        final_paragraph = (
            f"Out of {meta['articles_analyzed']} news articles analysed for"
            f" {meta['query']} over a {meta['prediction_window_days']}-day forecast"
            f" horizon, articles showed divided sentiment: {pos_pct:.1f}% of"
            f" aggregated weight leaned positive while {neg_pct:.1f}% leaned negative."
            f" Because neither direction reached the {tau_pos_ov * 100:.0f}% consensus"
            f" threshold required for a directional prediction, the overall outlook"
            f" is classified as NEUTRAL (mixed/inconclusive)."
            f"{contrastive_sentence}"
            f" The single most influential article was \"{top_art_title}\""
            f" ({top_art_sent} sentiment, contributing {top_art_share}% of total weight)."
            f" {weight_sentence}"
            f"{lime_sentence}"
            f"{flip_sentence}"
            f" {reliability_sentence}"
        )
    else:
        final_paragraph = (
            f"Out of {meta['articles_analyzed']} news articles analysed for"
            f" {meta['query']} over a {meta['prediction_window_days']}-day forecast"
            f" horizon, {n_pos} carried positive sentiment, {n_neg} carried negative"
            f" sentiment, and {n_neu} were neutral."
            f" The model produced a weighted aggregation of these scores and predicted"
            f" a {str(pred['final_label']).upper()} label with"
            f" {pred['final_confidence'] * 100:.1f}% model support."
            f"{contrastive_sentence}"
            f" The single most influential article was \"{top_art_title}\""
            f" ({top_art_sent} sentiment, contributing {top_art_share}% of total weight)."
            f" {weight_sentence}"
            f"{lime_sentence}"
            f"{flip_sentence}"
            f" {reliability_sentence}"
        )

    lines += [
        W,
        "  FINAL PLAIN-ENGLISH OVERVIEW",
        "  A complete, readable summary of the entire analysis for a general reader",
        W,
        "",
    ]
    lines += _wrap(final_paragraph, width=56)
    lines += ["", ""]

    # Disclaimer
    lines += [
        "  DISCLAIMER",
        w,
        "  This report presents news-based sentiment analysis only.",
        "  It is NOT financial advice and does NOT recommend any",
        "  trading action (buy, sell, or hold). The authors are not",
        "  responsible for any investment decision made using this output.",
        "",
    ]

    
    #  PART 2 - DETAILED TECHNICAL ANALYSIS
    #  Scores, article breakdowns, and model-internal attributions

    lines += [
        P,
        "  PART 2 - DETAILED TECHNICAL ANALYSIS",
        "  Scores, article breakdowns, and model-internal attributions",
        P,
        "",
    ]

    # Contrastive explanation 
    # "Why label A INSTEAD OF label B?" -- the core XAI question.

    if contrastive:
        c_winner   = cont_info["winner"].upper()
        c_runner   = cont_info["runner_up"].upper()
        c_gap      = cont_info["score_gap"]
        c_n_w      = cont_info["n_favouring_winner"]
        c_n_r      = cont_info["n_favouring_runner_up"]
        c_push_w   = cont_info["total_push_toward_winner"]
        c_push_r   = cont_info["total_push_toward_runner_up"]
        top_drivers = cont_info["top_gap_drivers"]

        if cont_info["is_threshold_override"]:
            lines += [
                f"  WHY {c_winner} INSTEAD OF {c_runner}?",
                "  Contrastive explanation - decision threshold override",
                w,
                f"  Raw scores : {c_runner} {cont_info['runner_up_score']:.4f}"
                f"  vs  {c_winner} {cont_info['winner_score']:.4f}"
                f"  ({c_runner.lower()} scored higher)",
                "",
                f"  Threshold override:",
                f"    {c_runner} scored {cont_info['runner_up_score']:.4f}"
                f" but needs >= {cont_info['tau_pos'] * 100:.0f}% (positive)"
                f" / {cont_info['tau_neg'] * 100:.0f}% (negative)",
                f"    {c_winner} scored {cont_info['winner_score']:.4f}"
                f" and exceeded its lower threshold",
                f"    This reflects calibrated correction for the model's positive bias",
                "",
                f"  Article split:",
                f"    {c_n_w:>3} articles push toward {c_winner}"
                f"  (total push: +{c_push_w:.4f})",
                f"    {c_n_r:>3} articles push toward {c_runner}"
                f"  (total push: {c_push_r:.4f})",
                "",
            ]
        else:
            lines += [
                f"  WHY {c_winner} INSTEAD OF {c_runner}?",
                "  Contrastive explanation - what tipped the balance",
                w,
                f"  Score gap : {c_winner} {cont_info['winner_score']:.4f}"
                f"  vs  {c_runner} {cont_info['runner_up_score']:.4f}"
                f"  (gap = {c_gap:.4f})",
                "",
                f"  Article split:",
                f"    {c_n_w:>3} articles push toward {c_winner}"
                f"  (total push: +{c_push_w:.4f})",
                f"    {c_n_r:>3} articles push toward {c_runner}"
                f"  (total push: {c_push_r:.4f})",
                "",
            ]

        # Third-place class role
        c_third = contrastive.get("third_place")
        c_third_score = contrastive.get("third_score", 0)
        c_neutral_effect = contrastive.get("neutral_effect", "none")
        if c_third and c_third_score > 0:
            third_upper = c_third.upper()
            if c_neutral_effect == "balanced":
                effect_text = (
                    f"absorbing probability mass roughly equally from both sides"
                )
            elif c_neutral_effect.startswith("helps_"):
                helped = c_neutral_effect.split("_", 1)[1].upper()
                hurt = c_runner if helped == c_winner else c_winner
                effect_text = (
                    f"drawing more mass from {hurt} than {helped},"
                    f" indirectly helping {helped}"
                )
            else:
                effect_text = "with minimal impact on the gap"
            lines += [
                f"  Third class: {third_upper} at {c_third_score * 100:.1f}%"
                f" - {effect_text}.",
                "",
            ]

        # Top gap drivers table
        if top_drivers:
            lines += [
                "  Top 5 articles driving the gap:",
                f"    {'#':<4} {'Direction':>10}  {'Net':>8}  Title",
                f"    {'-' * 70}",
            ]
            for i, drv in enumerate(top_drivers, 1):
                direction = f"\u2192 {drv['favours'][:3].upper()}"
                net_pct = drv["net_direction"] * 100
                lines.append(
                    f"    {i:<4} {direction:>10}  {net_pct:>+7.2f}%  {drv['title'][:48]}"
                )
            lines.append("")

        # LIME <-> contrastive bridge
        # For each top gap driver that also has LIME data, show
        # which words pushed it toward the winner or runner-up.
        # This answers "WHY did this article favour one side?"
        if top_drivers and lime_articles:
            lime_by_title_bridge: dict[str, dict] = {
                la["title"]: la for la in lime_articles
            }
            bridge_entries: list[tuple[dict, dict]] = []
            for drv in top_drivers:
                la = lime_by_title_bridge.get(drv["title"])
                if la:
                    bridge_entries.append((drv, la))

            if bridge_entries:
                lines += [
                    "  Word-level evidence for top gap drivers:",
                    "  (connecting article-level influence to token-level LIME attribution)",
                    "",
                ]
                for drv, la in bridge_entries:
                    favours     = drv["favours"]
                    net_pct     = drv["net_direction"] * 100
                    supporting  = la.get("top_tokens_supporting", [])[:4]
                    opposing    = la.get("top_tokens_opposing", [])[:4]

                    lines.append(
                        f"    {drv['title'][:60]}"
                    )
                    lines.append(
                        f"      Favours {favours.upper()} ({net_pct:+.2f}% net push)"
                    )
                    if supporting:
                        lines.append(
                            f"      Words reinforcing {str(pred['final_label']).upper()}: "
                            + ", ".join(supporting)
                        )
                    if opposing:
                        lines.append(
                            f"      Words pulling against: "
                            + ", ".join(opposing)
                        )
                    if not supporting and not opposing:
                        lines.append(
                            "      (no significant token-level signals found)"
                        )
                    lines.append("")

        # Minimum flip set
        if flip_info.get("flip_possible"):
            flip_n = flip_info["flip_set_size"]
            flip_total = flip_info["articles_total"]
            flip_new = flip_info["new_label"].upper()
            lines += [
                "  Counterfactual (minimum flip set):",
                f"    Removing just {flip_n} of {flip_total} articles would flip"
                f" the prediction from {c_winner} to {flip_new}.",
                "    These articles are:",
            ]
            for t in flip_info["flip_set_titles"][:5]:
                lines.append(f"      [!] {t[:70]}")
            if flip_n > 5:
                lines.append(f"      ... and {flip_n - 5} more")
            lines.append("")
        else:
            lines += [
                "  Counterfactual (minimum flip set):",
                f"    No subset of articles can flip the prediction - {c_winner}"
                " is robust across the article pool.",
                "",
            ]

    # Evidence quality
    lines += [
        "  PREDICTION RELIABILITY",
        "  How much to trust this prediction",
        w,
        f"  Overall : [{rel_icon}] {overall_rel}  ({reliability['flags_triggered']} concern(s) found)",
        "  Rating rule : HIGH if 0-1 flags, MEDIUM if 2, LOW if 3+.",
    ]
    lines.append("")
    for flag_name, flag_data in flags.items():
        is_skipped = flag_name == "label_margin" and "skipped" in flag_data.get("message", "").lower()
        if is_skipped:
            icon = "-"
        elif flag_data["flagged"]:
            icon = "\u26a0"
        else:
            icon = "\u2713"
        label = flag_name.replace("_", " ")
        lines.append(f"    [{icon}] {label:<22} {flag_data['message']}")
    lines += [
        "",
    ]

    # Calibration context
    calibration = result.get("calibration")
    if calibration:
        lines += [
            "  HISTORICAL CALIBRATION",
            "  How this model has performed on past predictions",
            w,
        ]
        for level in ["HIGH", "MEDIUM", "LOW"]:
            cal = calibration.get(level, {})
            if cal:
                n = cal.get("n", 0)
                acc = cal.get("accuracy", 0.0) * 100
                lines.append(
                    f"    {level:<8} evidence-quality predictions: "
                    f"{acc:.0f}% correct direction  (n={n})"
                )
        lines += [
            "",
            f"  Backtest period : {calibration.get('period', 'N/A')}",
            "",
        ]
    else:
        lines += [
            "  HISTORICAL CALIBRATION",
            "  How this model has performed on past predictions",
            w,
            "  Not yet available. Run the backtest evaluation module to",
            "  populate this section with accuracy, precision, and recall",
            "  broken down by evidence-quality level (HIGH / MEDIUM / LOW).",
            "",
        ]

    # Layer 2 - Article influence
    lines += [
        "  WHICH ARTICLES DROVE THE PREDICTION",
        "  The articles that had the most impact on the final verdict",
        w,
    ]

    # Article sentiment distribution chart
    lines += ["  Article sentiment distribution:", ""]
    for s_label, s_icon in [("positive", "+"), ("negative", "-"), ("neutral", "=")]:
        cnt = sent_counts.get(s_label, 0)
        bar = _ascii_bar(cnt, total_arts, width=30)
        pct = cnt / total_arts * 100
        lines.append(f"    {s_icon} {s_label:>8} : {bar}  {cnt:>3} articles ({pct:.0f}%)")
    if threshold_neutral:
        lines.append("")
        lines.append(
            "  Note: Although most articles lean positive or negative,"
        )
        lines.append(
            "  the sentiment is closely divided. The system requires a clear"
        )
        lines.append(
            "  directional majority to issue a positive or negative prediction."
        )
        lines.append(
            "  Because the split is too close, the verdict is NEUTRAL (mixed)."
        )
    lines.append("")

    # Drivers
    if layer2["top_positive_drivers"]:
        lines.append("  Articles pushing toward POSITIVE:")
        for t in layer2["top_positive_drivers"]:
            lines.append(f"    [+]  {t[:76]}")
    if layer2["top_negative_drivers"]:
        lines.append("  Articles pushing toward NEGATIVE:")
        for t in layer2["top_negative_drivers"]:
            lines.append(f"    [-]  {t[:76]}")
    lines.append("")

    # Label-flipping articles
    flippers = layer2["label_flipping_articles"]
    if flippers:
        lines.append("  Critical articles (removing any one would flip the verdict):")
        for t in flippers:
            lines.append(f"    [!]  {t[:76]}")
        lines.append("")

    # Top-10 table with weight bar chart
    # "Weight" is the effective article weight used in the sentiment aggregation
    # (geometric mean of recency and horizon weights, with coverage boost and headline-only discount applied).
    # "Share" is the article's percentage of total weight across all articles.
    lines += [
        "  Top 10 most influential articles:",
        "  (Weight = recency \u00d7 horizon factor, 0-1 scale;  Share = % of total pool weight)",
        f"    {'#':<4} {'Sent':>4}  {'Weight':>6}  {'Share':>6}  {'Weight chart':<32}  Title",
        f"    {w[:78]}",
    ]
    max_w = max((a["final_weight"] for a in ranked_arts[:10]), default=1.0)
    for art in ranked_arts[:10]:
        sentiment = art["dominant_sentiment"][:3].upper()
        share_pct = art.get("weight_share", 0.0) * 100
        bar       = _ascii_bar(art["final_weight"], max_w, width=30)
        lines.append(
            f"    #{art['rank']:<3} {sentiment:>4}  {art['final_weight']:>6.4f}"
            f"  {share_pct:>5.1f}%  {bar}  {art['title'][:32]}"
        )
    hhi = layer2["weight_concentration"]
    lines += [
        "",
        f"  Weight concentration (HHI): {hhi:.4f}  "
        f"({'well-distributed' if hhi <= XAI_CONCENTRATION_THRESHOLD else 'concentrated'}) - "
        f"0 = perfectly uniform across all articles, 1 = one article dominates",
        "",
    ]

    # Neutral-won justification for top article
    if ranked_arts:
        top = ranked_arts[0]
        top_sent = top.get("dominant_sentiment", "unknown")
        if top_sent == "neutral":
            top_raw = top.get("raw_scores", {})
            pos_s = top_raw.get("positive", 0.0)
            neg_s = top_raw.get("negative", 0.0)
            neu_s = top_raw.get("neutral", 0.0)
            lines += [
                f"  Note on #{top['rank']} ({top['title'][:50]}\u2026):",
                f"    This article was classified as NEUTRAL despite its headline.",
                f"    Raw scores: positive={pos_s:.3f}, negative={neg_s:.3f}, "
                f"neutral={neu_s:.3f}.",
            ]
            if neu_s > pos_s and neu_s > neg_s:
                lines.append(
                    "    The neutral hypothesis captured the highest probability - "
                    "the article's language is informational or factual rather than "
                    "directionally bullish or bearish."
                )
            elif abs(pos_s - neg_s) < 0.10:
                lines.append(
                    "    Positive and negative scores nearly cancel out, "
                    "resulting in neutral as the dominant label by default."
                )
            else:
                lines.append(
                    "    The model judged the content as not strongly directional "
                    "despite the headline phrasing."
                )
            lines.append("")

    # Layer 3 - Pipeline weighting
    lines += [
        "  HOW ARTICLES WERE WEIGHTED",
        "  More recent and near-horizon articles receive higher weight",
        w,
        "  De-duplication  : Articles are de-duplicated across all sources",
        "                    by headline. When the same headline appears on",
        "                    multiple domains (e.g. Yahoo and Benzinga), only",
        "                    the oldest copy is kept - the first to break the",
        "                    news is what the market reacted to. If two copies",
        "                    share the same timestamp, the one with more content",
        "                    is preferred.",
        "",
        f"  Average article age      : {layer3['avg_days_ago']} days old",
        f"  Average recency weight   : {layer3['avg_recency_weight']}"
        "  (1.0 = today, lower = older)",
        f"  Average horizon weight   : {layer3['avg_horizon_weight']}"
        "  (1.0 = ideal timing, lower = off-horizon)",
        "",
        "  Article timing breakdown:",
        "",
    ]
    horizon_labels = {
        "IMMEDIATE":   "Immediate (0-1 days) ",
        "SHORT_TERM":  "Short-term (2-5 days)",
        "DIFFUSION":   "Diffusion (6-10 days)",
        "LONG_TERM":   "Long-term (11-31 days)",
    }
    horizon_dist = layer3["horizon_distribution"]
    max_h        = max(horizon_dist.values(), default=1)
    total_h      = sum(horizon_dist.values()) or 1
    for cat, count in horizon_dist.items():
        label = horizon_labels.get(cat, f"{cat:<20}")
        bar   = _ascii_bar(count, max_h, width=30)
        pct   = count / total_h * 100
        lines.append(f"    {label} : {bar}  {count:>3} ({pct:.0f}%)")
    lines.append("")

    # Layer 1 - Token attribution (LIME) - summary only 
    lines += [
        "  WHICH WORDS DROVE THE PREDICTION",
        "  Words inside each article that pushed the model toward or away from the verdict",
        w,
        "  IMPORTANT: these tokens explain why the *sentiment model* chose its",
        "  label - they do NOT indicate why a stock price would move. LIME",
        "  perturbs the input text and measures which",
        "  words most change the model's output distribution. The results are",
        "  model-internal attributions, not causal drivers of market returns.",
        "",
        "  Company name, ticker, and common stopwords are filtered so only",
        "  sentiment-bearing words remain. Full token weights are in the",
        "  ADVANCED / DIAGNOSTICS section at the end of this report.",
        "",
    ]
    if not lime_articles:
        lines.append("  (Word-level analysis did not run or returned no results)")
    else:
        for art in lime_articles:
            supporting_tokens = art["top_tokens_supporting"]
            opposing_tokens   = art["top_tokens_opposing"]
            artefacts         = [t for t in opposing_tokens if is_lime_noise_token(t, noise_set)]

            lines += [
                f"  [{art['rank']}] {art['title'][:72]}",
                f"      Words pushing TOWARD  " + str(pred["final_label"]).upper() + " : "
                + (", ".join(supporting_tokens) if supporting_tokens else "(none)"),
                "      Words pushing AGAINST " + str(pred["final_label"]).upper() + " : "
                + (", ".join(opposing_tokens) if opposing_tokens else "(none)"),
            ]
            if artefacts:
                lines.append(
                    f"      Note: {', '.join(artefacts)} "
                    f"{'are' if len(artefacts) > 1 else 'is a'} common filler word(s) - "
                    "likely a text-format artefact, not meaningful signal."
                )
            lines.append("")

    # LIME interpretability warning
    if lime_articles:
        total_top_tokens = 0
        stopword_top_tokens = 0
        for art in lime_articles:
            for t in art.get("top_tokens_supporting", [])[:3]:
                total_top_tokens += 1
                if is_lime_noise_token(t, noise_set):
                    stopword_top_tokens += 1
            for t in art.get("top_tokens_opposing", [])[:3]:
                total_top_tokens += 1
                if is_lime_noise_token(t, noise_set):
                    stopword_top_tokens += 1
        if total_top_tokens > 0:
            stopword_ratio = stopword_top_tokens / total_top_tokens
            if stopword_ratio > 0.40:
                lines += [
                    "  \u26a0 LOW INTERPRETABILITY WARNING",
                    f"    {stopword_top_tokens}/{total_top_tokens} "
                    f"({stopword_ratio * 100:.0f}%) of top-ranked tokens are common"
                    " stopwords or filler words.",
                    "    Token-level signals are likely dominated by text-format"
                    " artefacts rather than meaningful sentiment cues.",
                    "    Consider sentence-level attributions or manual review"
                    " for this prediction.",
                    "",
                ]
    lines.append("")

    # Charts section (names only, paths in Advanced)
    if chart_paths:
        lines += [
            "  CHARTS",
            "  Visual summaries generated for this analysis (see paths in Advanced section)",
            w,
        ]
        for key in chart_paths:
            label = chart_label_map.get(key, key.replace("_", " ").title())
            lines.append(f"    [{key}]  {label}")
        lines.append("")

    # Advanced / Diagnostics
    # Technical details for developers and researchers - moved
    # here so the main report stays end-user friendly.
    lines += [
        "",
        "  ADVANCED / DIAGNOSTICS",
        "  Technical details for developers and reproducibility",
        w,
    ]

    # Narrative model info
    lines += [
        f"  Narrative model      : {narrative['model']}"
        f"  ({'live Ollama' if narrative['ollama_available'] else 'template fallback'})",
        f"  XAI version          : {meta.get('xai_version', '1.0.0')}",
        f"  LIME method          : {layer1.get('method', 'LIME')}"
        f"  (top {layer1.get('lime_top_n', 'N/A')} articles)",
        "",
    ]

    # Chart file paths
    if chart_paths:
        lines.append("  Chart file paths:")
        for key, path in chart_paths.items():
            label = chart_label_map.get(key, key)
            lines.append(f"    {label}: {path}")
        lines.append("")

    # Full LIME token influence charts + raw weights (moved from main body)
    if lime_articles:
        lines += [
            "  LIME TOKEN DETAIL",
            "  Full token influence charts and raw numerical weights",
            f"  {'-' * 50}",
            "  (+) = supports predicted label     (-) = opposes predicted label",
            "",
        ]
        for art in lime_articles:
            lines += [
                f"  [{art['rank']}] {art['title'][:72]}",
            ]
            # Show article content so the reader can see where non-headline tokens come from
            art_content = art.get("content", "")
            if art_content:
                lines.append(f"      Content: {art_content[:120]}")
                if len(art_content) > 120:
                    lines.append(f"               {art_content[120:240]}")
            else:
                lines.append("      Content: (none - headline only)")
            lines.append(
                f"      Article weight : {art['final_weight']:.4f}  |  "
                f"Relevance to prediction : {art['influence_score']:.4f}"
            )

            # Token influence bar chart (noise tokens excluded)
            tw_sorted = sorted(
                [tw for tw in art["token_weights"]
                 if not is_lime_noise_token(tw["token"], noise_set)],
                key=lambda x: abs(x["weight"]), reverse=True,
            )[:8]
            max_tw = max((abs(t["weight"]) for t in tw_sorted), default=1.0)
            lines.append("      Token influence chart:")
            for tw in tw_sorted:
                direction = "+" if tw["direction"] == "supports" else "-"
                bar       = _ascii_bar(abs(tw["weight"]), max_tw, width=20)
                lines.append(
                    f"        {direction} {tw['token']:<16} {bar}  ({tw['weight']:+.4f})"
                )
            lines.append("")

            # Raw weight table (noise tokens excluded)
            lines += [
                f"      {'Token':<22} {'Dir':<5} {'Weight':>10}",
                f"      {'-' * 40}",
            ]
            for tw in sorted(
                [tw for tw in art["token_weights"]
                 if not is_lime_noise_token(tw["token"], noise_set)],
                key=lambda x: abs(x["weight"]), reverse=True,
            )[:10]:
                direction = "(+)" if tw["direction"] == "supports" else "(-)"
                lines.append(
                    f"      {tw['token']:<22} {direction:<5} {tw['weight']:>+10.6f}"
                )
            lines.append("")

    lines += [w]

    return "\n".join(lines)


# File I/O (Summary)

def _save_summary(
    result: dict[str, Any],
    summary_path: str,
    chart_paths: dict | None = None,
) -> None:
    """Render the summary text and write it to the given file path."""
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = _build_summary_text(result, chart_paths=chart_paths)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    logger.info("XAI summary saved to %s", summary_path)


# Main Orchestrator

def run_xai(
    prediction_result: dict[str, Any],
    articles_json_path: str | None = None,
    output_path: str | None = None,
    summary_path: str | None = None,
    charts_dir: str | None = None,
) -> dict[str, Any]:
    """Run all XAI layers and produce the full explainability report."""
    if articles_json_path is None:
        articles_json_path = str(JSON_PATH)
    if output_path is None:
        output_path = str(XAI_OUTPUT_PATH)
    if summary_path is None:
        summary_path = str(XAI_SUMMARY_PATH)
    charts_dir_path = Path(charts_dir) if charts_dir else Path(summary_path).parent / "charts"

    company_name = (
        prediction_result.get("company_name")
        or prediction_result.get("query")
        or "unknown"
    )
    ticker = prediction_result.get("ticker") or ""

    logger.info("Starting XAI for company='%s' ticker='%s'", company_name, ticker)

    articles_data = _load_articles(articles_json_path)
    prediction_window_days = articles_data.get("prediction_window_days", 7)
    max_backward_days = articles_data.get("max_backward_days")
    merged_articles = _merge_article_data(prediction_result, articles_data)

    logger.info("Merged %d articles for XAI.", len(merged_articles))

    # Compute news lookback date range from article ages
    ages = [a.get("days_ago", 0) for a in merged_articles if a.get("days_ago") is not None]
    if ages:
        from datetime import timedelta
        now = datetime.now()
        oldest_date = (now - timedelta(days=max(ages))).strftime("%Y-%m-%d")
        newest_date = (now - timedelta(days=min(ages))).strftime("%Y-%m-%d")
        # Check if articles have market_date (ET alignment applied)
        has_market_date = any(a.get("market_date") for a in merged_articles)
        if has_market_date:
            news_lookback = f"{oldest_date} to {newest_date} (ET market-close aligned)"
        else:
            news_lookback = f"{oldest_date} to {newest_date} (UTC seendate)"
    else:
        news_lookback = "N/A"

    # Layer 2 + 3 first
    article_explanation  = explain_articles(merged_articles, prediction_result)
    pipeline_explanation = explain_pipeline(merged_articles, prediction_window_days)

    # Evidence quality (needs HHI from article_explanation + merged_articles)
    hhi = article_explanation.get("weight_concentration", 0.0)
    flip_set_data = article_explanation.get("minimum_flip_set")
    reliability = compute_reliability(
        prediction_result, hhi,
        merged_articles=merged_articles,
        prediction_window_days=prediction_window_days,
        max_backward_days=max_backward_days,
        flip_set_data=flip_set_data,
    )

    # Layer 1 
    logger.info("Running LIME token attribution (top %d articles)...", XAI_LIME_TOP_N)
    token_explanation = explain_tokens(
        merged_articles,
        company_name=company_name,
        predicted_label=prediction_result.get("final_label", "positive"),
        ticker=ticker,
    )

    # Narrative storyline clustering (TF-IDF + Ollama labels)
    logger.info("Running narrative storyline clustering...")
    storyline_data = cluster_narratives(merged_articles)

    # Narrative - Ollama
    narrative = generate_narrative(
        prediction_result=prediction_result,
        article_explanation=article_explanation,
        pipeline_explanation=pipeline_explanation,
        reliability=reliability,
        token_explanation=token_explanation,
        company_name=company_name,
    )

    result = {
        "meta": {
            "query": prediction_result.get("query", ""),
            "ticker": ticker,
            "xai_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "articles_analyzed": prediction_result.get("articles_analyzed", 0),
            "prediction_window_days": prediction_window_days,
            "news_lookback": news_lookback,
            "xai_version": "1.0.0",
        },
        "prediction_summary": {
            "final_label":       prediction_result.get("final_label"),
            "final_confidence":  prediction_result.get("final_confidence"),
            "normalized_scores": prediction_result.get("normalized_scores"),
            "total_weight":      prediction_result.get("total_weight"),
            "abstention_test":   prediction_result.get("abstention_test"),
        },
        "reliability":      reliability,
        "layer_1_token":    {
            "method":           "LIME",
            "lime_top_n":       XAI_LIME_TOP_N,
            "articles":         token_explanation,
        },
        "layer_2_article":  article_explanation,
        "layer_3_pipeline": pipeline_explanation,
        "narrative":        narrative,
        "storylines":       storyline_data,
    }

    _save_result(result, output_path)

    # Generate matplotlib PNG charts
    logger.info("Generating PNG charts into %s ...", charts_dir_path)
    chart_paths = generate_all_charts(result, charts_dir_path)

    summary_text = _build_summary_text(result, chart_paths=chart_paths)

    # Save to file
    path = Path(summary_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(summary_text)
    logger.info("XAI summary saved to %s", summary_path)

    # Console preview
    print(summary_text)

    return result
