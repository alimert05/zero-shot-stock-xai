"""
Streamlit web interface for the Zero-Shot Stock Sentiment & XAI pipeline.

Run with:
    cd src
    streamlit run app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st
import json
import pandas as pd
from datetime import date, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from config import (
    JSON_PATH,
    ZEROSHOT_PREDS,
    XAI_OUTPUT_PATH,
    XAI_EXPLANATIONS_PATH,
    SENTIMENT_MODEL,
    MODEL_NAME,
)

st.set_page_config(
    page_title="Stock Sentiment Analyser",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

_LABEL_COLOURS = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
}

_RELIABILITY_COLOURS = {
    "HIGH":   "#2ecc71",
    "MEDIUM": "#f39c12",
    "LOW":    "#e74c3c",
}

_HORIZON_COLOURS = {
    "IMMEDIATE":   "#e74c3c",
    "SHORT_TERM":  "#f39c12",
    "MEDIUM_TERM": "#3498db",
    "LONG_TERM":   "#2ecc71",
}

_EVENT_TYPE_INFO = {
    "earnings report or financial results": {
        "description": "Quarterly/annual earnings releases, revenue figures, profit margins, and financial guidance.",
        "horizon": "IMMEDIATE (3 days)",
        "why": "Markets react within hours to days as traders price in financial performance.",
    },
    "analyst rating, upgrade, or downgrade": {
        "description": "Buy/sell/hold ratings, price target changes, and research notes from investment banks and analysts.",
        "horizon": "SHORT-TERM (7 days)",
        "why": "Analyst opinions influence institutional trading decisions over the following week.",
    },
    "product launch, innovation, or technology": {
        "description": "New product announcements, R&D breakthroughs, patent filings, and technology partnerships.",
        "horizon": "SHORT-TERM (7 days)",
        "why": "Product news generates initial buzz that settles within a week as the market assesses real impact.",
    },
    "regulatory action, legal case, or investigation": {
        "description": "Government investigations, lawsuits, compliance rulings, fines, and regulatory approvals.",
        "horizon": "MEDIUM-TERM (14 days)",
        "why": "Legal and regulatory outcomes unfold over weeks as details emerge and implications become clear.",
    },
    "strategic restructuring, merger, or acquisition": {
        "description": "M&A deals, spin-offs, major reorganisations, leadership changes, and strategic pivots.",
        "horizon": "LONG-TERM (31 days)",
        "why": "Structural changes take weeks to months for the market to fully evaluate and price in.",
    },
    "general market commentary or opinion": {
        "description": "Broad market analysis, opinion columns, sector outlooks, and economic commentary mentioning the company.",
        "horizon": "SHORT-TERM (7 days)",
        "why": "Commentary sentiment fades quickly as new information replaces older opinions.",
    },
}


def _badge(text: str, colour: str, size: str = "0.85rem") -> str:
    return (
        f'<span style="background:{colour};color:#fff;padding:3px 10px;'
        f'border-radius:4px;font-size:{size};font-weight:600;'
        f'margin-right:4px;display:inline-block;">{text}</span>'
    )


def _label_badge(label: str, confidence: float) -> str:
    colour = _LABEL_COLOURS.get(label, "#3498db")
    return (
        f'<span style="background:{colour};color:#fff;padding:8px 22px;'
        f'border-radius:6px;font-size:1.5rem;font-weight:700;">'
        f'{label.upper()}&nbsp;&nbsp;{confidence:.1%}</span>'
    )


def _reliability_badge(level: str) -> str:
    colour = _RELIABILITY_COLOURS.get(level, "#3498db")
    return (
        f'<span style="background:{colour};color:#fff;padding:4px 14px;'
        f'border-radius:4px;font-weight:600;">{level}</span>'
    )


def _hex_to_rgba(hex_color: str, alpha: float = 0.3) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _load_article_content() -> dict[str, str]:
    """Load articles.json and return a title -> content lookup."""
    try:
        with open(str(JSON_PATH), "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            a.get("title", ""): (a.get("content") or "")
            for a in data.get("articles", [])
        }
    except Exception:
        return {}


def _build_comprehensive_summary(result: dict) -> str:
    """Build a large plain-English summary covering everything from the XAI result."""
    pred = result.get("prediction_summary", {})
    meta = result.get("meta", {})
    narrative = result.get("narrative", {})
    reliability = result.get("reliability", {})
    layer2 = result.get("layer_2_article", {})
    contrastive = layer2.get("contrastive", {})
    storylines_data = result.get("storylines", {})
    flip_set = layer2.get("minimum_flip_set", {})
    flip_articles = layer2.get("label_flipping_articles", [])

    company = meta.get("query", "the company")
    ticker = meta.get("ticker", "")
    label = pred.get("final_label", "unknown")
    confidence = pred.get("final_confidence", 0)
    n_articles = meta.get("articles_analyzed", "?")
    window = meta.get("prediction_window_days", "?")
    lookback = meta.get("news_lookback", "?")
    scores = pred.get("normalized_scores", {})

    parts = []

    parts.append(
        f"The model analysed **{n_articles} news articles** about **{company}**"
        f"{f' ({ticker})' if ticker else ''} "
        f"over a **{window}-day prediction window** "
        f"(news lookback: {lookback}) "
        f"and predicted a **{label.upper()}** label with **{confidence:.1%} confidence**."
    )

    if scores:
        score_parts = []
        for lbl in ["positive", "negative", "neutral"]:
            val = scores.get(lbl, 0)
            score_parts.append(f"{lbl} {val:.1%}")
        parts.append(
            f"The weighted sentiment distribution across all articles is: {', '.join(score_parts)}."
        )

    narrative_text = narrative.get("summary") or narrative.get("fallback_summary") or ""
    if narrative_text:
        parts.append(narrative_text)

    winner = contrastive.get("winner", "")
    runner = contrastive.get("runner_up", "")
    gap = contrastive.get("score_gap", 0)
    n_fav_w = contrastive.get("n_favouring_winner", 0)
    n_fav_r = contrastive.get("n_favouring_runner_up", 0)
    if winner and runner:
        parts.append(
            f"The prediction chose **{winner.upper()}** over **{runner.upper()}** "
            f"by a score gap of **{gap:.4f}**. "
            f"{n_fav_w} article{'s' if n_fav_w != 1 else ''} favoured {winner} "
            f"while {n_fav_r} favoured {runner}."
        )

    ranked = layer2.get("ranked_articles", [])
    if ranked:
        top3 = ranked[:3]
        driver_parts = []
        for a in top3:
            t = a.get("title", "?")
            s = a.get("dominant_sentiment", "?")
            w = a.get("weight_share", 0)
            driver_parts.append(f'"{t}" ({s}, {w:.1%} weight)')
        parts.append(
            f"The most influential articles were: {'; '.join(driver_parts)}."
        )

    storylines = storylines_data.get("storylines", [])
    if storylines:
        theme_parts = []
        for sl in storylines[:4]:
            name = sl.get("label") or sl.get("keyword_label", "?")
            sg = sl.get("sentiment_group", "?")
            nc = sl.get("articles_count", 0)
            theme_parts.append(f"{name} ({sg}, {nc} articles)")
        parts.append(
            f"The main narrative themes identified are: {'; '.join(theme_parts)}."
        )

    rel_level = reliability.get("overall_reliability", "?")
    flags_triggered = reliability.get("flags_triggered", 0)
    flagged_names = [
        name for name, info in reliability.get("flags", {}).items()
        if info.get("flagged")
    ]
    rel_text = (
        f"Prediction reliability is **{rel_level}** "
        f"({flags_triggered} of 7 checks flagged)."
    )
    if flagged_names:
        rel_text += f" Concerns: {', '.join(flagged_names)}."
    parts.append(rel_text)

    if flip_articles:
        parts.append(
            f"The prediction is sensitive: removing any single one of "
            f"**{len(flip_articles)} article{'s' if len(flip_articles) != 1 else ''}** "
            f"would flip the label."
        )
    elif flip_set and flip_set.get("flip_possible"):
        n_flip = flip_set.get("flip_set_size", "?")
        new_lbl = flip_set.get("new_label", "?")
        parts.append(
            f"Removing as few as **{n_flip} articles** would flip the prediction to "
            f"**{new_lbl.upper()}**."
        )
    else:
        parts.append(
            "The prediction is robust: no feasible combination of article removals "
            "would change the label."
        )

    return "\n\n".join(parts)


def _run_full_pipeline(
    company_name: str,
    start_date_str: str,
    end_date_str: str,
    num_articles: int,
) -> dict | None:
    """Execute fetcher -> predictor -> XAI and return the XAI result dict."""
    from fetcher.fetcher import Fetcher
    from predictor.zero_shot import run_sentiment_prediction
    from xai import run_xai

    progress = st.status("Running analysis pipeline...", expanded=True)

    progress.write("Fetching news articles from Finnhub...")
    fetcher = Fetcher()
    try:
        has_articles = fetcher.run_pipeline(
            company_name=company_name,
            start_date=start_date_str,
            end_date=end_date_str,
            num_articles=num_articles,
        )
    except RuntimeError as exc:
        progress.update(label="Pipeline failed", state="error")
        st.error(str(exc))
        return None
    except Exception as exc:
        progress.update(label="Pipeline failed", state="error")
        st.error(f"Fetcher error: {exc}")
        return None

    if not has_articles:
        progress.update(label="No articles found", state="error")
        st.warning(
            "No articles were found for this company / date range. "
            "Try a wider date window or check the company name."
        )
        return None

    try:
        with open(str(JSON_PATH), "r", encoding="utf-8") as f:
            articles_data = json.load(f)
        n_articles = articles_data.get("article_count", len(articles_data.get("articles", [])))
    except Exception:
        n_articles = "?"
    progress.write(f"Fetched **{n_articles}** articles.")

    progress.write("Running zero-shot sentiment prediction...")
    try:
        prediction_result = run_sentiment_prediction(
            articles_json_path=str(JSON_PATH),
            output_path=str(ZEROSHOT_PREDS),
        )
    except Exception as exc:
        progress.update(label="Prediction failed", state="error")
        st.error(f"Prediction error: {exc}")
        return None

    label = prediction_result.get("final_label", "?").upper()
    conf = prediction_result.get("final_confidence", 0)
    progress.write(f"Prediction: **{label}** (confidence {conf:.2%})")

    progress.write("Running XAI explainability analysis (this may take a few minutes)...")
    try:
        xai_result = run_xai(
            prediction_result=prediction_result,
            articles_json_path=str(JSON_PATH),
            output_path=str(XAI_OUTPUT_PATH),
        )
    except Exception as exc:
        progress.update(label="XAI failed", state="error")
        st.error(f"XAI error: {exc}")
        return None

    progress.write("Analysis complete!")
    progress.update(label="Pipeline complete", state="complete")
    return xai_result


def _render_overview(result: dict) -> None:
    pred = result.get("prediction_summary", {})
    narrative = result.get("narrative", {})
    meta = result.get("meta", {})

    label = pred.get("final_label", "unknown")
    confidence = pred.get("final_confidence", 0)
    total_weight = pred.get("total_weight", 0)
    n_articles = meta.get("articles_analyzed", "?")
    window = meta.get("prediction_window_days", "?")

    st.markdown("### Prediction Result")
    st.markdown(_label_badge(label, confidence), unsafe_allow_html=True)
    st.write("")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Confidence", f"{confidence:.2%}")
    m2.metric("Articles Analysed", n_articles)
    m3.metric("Prediction Window", f"{window} days")
    m4.metric("Total Weight", f"{total_weight:.2f}" if isinstance(total_weight, (int, float)) else "?")

    st.divider()

    scores = pred.get("normalized_scores", {})
    if scores:
        st.markdown("### Sentiment Score Distribution")
        cols = st.columns(3)
        for i, lbl in enumerate(["positive", "negative", "neutral"]):
            val = scores.get(lbl, 0)
            colour = _LABEL_COLOURS.get(lbl, "#3498db")
            with cols[i]:
                st.markdown(
                    f'<div style="text-align:center;font-weight:600;'
                    f'color:{colour};">{lbl.capitalize()}</div>',
                    unsafe_allow_html=True,
                )
                st.progress(val)
                st.markdown(
                    f'<div style="text-align:center;">{val:.2%}</div>',
                    unsafe_allow_html=True,
                )

    st.divider()

    st.markdown("### Final Summary")
    summary = _build_comprehensive_summary(result)
    st.markdown(
        f'<div style="background:#f0f2f6;padding:20px 24px;border-radius:8px;'
        f'font-size:1.05rem;line-height:1.7;">{summary}</div>',
        unsafe_allow_html=True,
    )

    with st.expander("Narrative generation details"):
        st.write(f"**Model:** {narrative.get('model', '?')}")
        st.write(f"**Ollama available:** {narrative.get('ollama_available', '?')}")
        st.write(f"**Fallback used:** {narrative.get('fallback_used', '?')}")
        violations = narrative.get("hallucination_violations", [])
        if violations:
            st.warning("Hallucination violations detected:")
            for v in violations:
                st.write(f"- {v}")
        else:
            st.write("No hallucination violations.")

    st.divider()

    st.markdown("### Analysis Metadata")
    mc1, mc2 = st.columns(2)
    with mc1:
        st.write(f"**Company:** {meta.get('query', '?')}")
        st.write(f"**Ticker:** {meta.get('ticker', '?')}")
        st.write(f"**News lookback:** {meta.get('news_lookback', '?')}")
    with mc2:
        st.write(f"**XAI version:** {meta.get('xai_version', '?')}")
        st.write(f"**Timestamp:** {meta.get('xai_timestamp', '?')}")
        st.write(f"**Model:** `{MODEL_NAME}`")


def _render_reliability(result: dict) -> None:
    reliability = result.get("reliability", {})

    rel_level = reliability.get("overall_reliability", "?")
    flags_triggered = reliability.get("flags_triggered", 0)
    summary_msg = reliability.get("summary_message", "")

    st.markdown("### Prediction Reliability")
    st.markdown(
        f"Overall: {_reliability_badge(rel_level)}"
        f"&nbsp;&nbsp;({flags_triggered} of 7 flags triggered)",
        unsafe_allow_html=True,
    )
    st.write("")

    if summary_msg:
        st.info(summary_msg)

    st.divider()

    st.markdown("### Reliability Flags")
    flags = reliability.get("flags", {})

    _FLAG_DESCRIPTIONS = {
        "thin_evidence":      "Are there enough articles to make a reliable prediction?",
        "weight_concentration": "Is the prediction driven by too few articles?",
        "label_margin":       "Is the gap between the top two labels large enough?",
        "low_confidence":     "Is the overall confidence above the minimum threshold?",
        "source_diversity":   "Do the articles come from diverse editorial sources?",
        "timing_alignment":   "Are article timestamps aligned to market-close sessions?",
        "horizon_coverage":   "Does the news span cover the intended backward window?",
    }

    for name, info in flags.items():
        flagged = info.get("flagged", False)
        msg = info.get("message", "")
        description = _FLAG_DESCRIPTIONS.get(name, "")

        if flagged:
            st.markdown(
                f"**:orange[{name}]** &nbsp; :orange[FLAGGED]",
            )
        else:
            st.markdown(
                f"**:green[{name}]** &nbsp; :green[PASS]",
            )

        st.caption(description)
        st.write(msg)

        details = {k: v for k, v in info.items() if k not in ("flagged", "message")}
        if details:
            with st.expander(f"Details for {name}"):
                for k, v in details.items():
                    st.write(f"**{k}:** {v}")

        st.write("")


def _render_storylines(result: dict) -> None:
    storylines_data = result.get("storylines", {})
    layer3 = result.get("layer_3_pipeline", {})

    storylines = storylines_data.get("storylines", [])
    other_count = storylines_data.get("other_count", 0)
    method = storylines_data.get("method", "?")

    if not storylines:
        st.warning("No storyline clusters available.")
        return

    st.markdown("### Narrative Storylines")
    st.caption(f"Clustering method: `{method}`")
    st.write("")

    for sentiment_group in ["positive", "negative", "neutral"]:
        group_items = [s for s in storylines if s.get("sentiment_group") == sentiment_group]
        if not group_items:
            continue

        colour = _LABEL_COLOURS.get(sentiment_group, "#95a5a6")
        st.markdown(
            f'<h4 style="color:{colour};">{sentiment_group.upper()} Themes</h4>',
            unsafe_allow_html=True,
        )

        for sl in group_items:
            label_text = sl.get("label") or sl.get("keyword_label", "Unknown")
            n_arts = sl.get("articles_count", 0)
            weighted_score = sl.get("weighted_score", 0)
            contribution = sl.get("contribution_score", 0)

            st.markdown(f"**{label_text}**")
            c1, c2, c3 = st.columns(3)
            c1.metric("Articles", n_arts)
            c2.metric("Weighted Score", f"{weighted_score:.4f}" if isinstance(weighted_score, (int, float)) else "?")
            c3.metric("Contribution", f"{contribution:.4f}" if isinstance(contribution, (int, float)) else "?")

            sent = sl.get("sentiment", {})
            if sent:
                st.caption(
                    f"Positive: {sent.get('positive', 0)} | "
                    f"Negative: {sent.get('negative', 0)} | "
                    f"Neutral: {sent.get('neutral', 0)}"
                )

            titles = sl.get("top_titles", [])
            if titles:
                with st.expander(f"Articles in this theme ({len(titles)})"):
                    for t in titles:
                        st.write(f"- {t}")
            st.write("")

    if other_count:
        st.caption(f"_{other_count} additional article(s) did not cluster into any theme._")

    st.divider()

    event_dist = layer3.get("event_type_distribution", {})
    event_sent = layer3.get("event_type_sentiment", {})

    if event_dist:
        st.markdown("### Event Type Distribution")
        st.write(
            "Each article is classified into an event type using zero-shot NLI. "
            "The event type determines the article's **impact horizon** — "
            "how many days the market typically takes to fully price in that type of news. "
            "Articles whose horizon aligns with the prediction window receive higher weight."
        )
        st.write("")

        evt_rows = []
        for evt, count in sorted(event_dist.items(), key=lambda x: x[1], reverse=True):
            sent = event_sent.get(evt, {})
            info = _EVENT_TYPE_INFO.get(evt, {})
            evt_rows.append({
                "Event Type": evt.replace("_", " ").title(),
                "Count": count,
                "Impact Horizon": info.get("horizon", "?"),
                "Positive": sent.get("positive", 0),
                "Negative": sent.get("negative", 0),
                "Neutral": sent.get("neutral", 0),
            })

        df = pd.DataFrame(evt_rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        with st.expander("What do these event types mean?"):
            for evt_name, info in _EVENT_TYPE_INFO.items():
                st.markdown(f"**{evt_name.title()}**")
                st.write(info["description"])
                st.caption(f"Impact horizon: {info['horizon']} — {info['why']}")
                st.write("")


def _build_merged_table(result: dict) -> pd.DataFrame:
    """Merge article rankings, contrastive contributions, and weight breakdown into one table."""
    layer2 = result.get("layer_2_article", {})
    layer3 = result.get("layer_3_pipeline", {})
    ranked = layer2.get("ranked_articles", [])
    contrastive = layer2.get("contrastive", {})

    contribs_lookup = {}
    winner = contrastive.get("winner", "?")
    runner = contrastive.get("runner_up", "?")
    for c in contrastive.get("all_contributions", []):
        contribs_lookup[c.get("title", "")] = c.get("net_direction", 0)

    weight_lookup = {}
    for art in layer3.get("articles", []):
        rec = art.get("recency_explanation", {})
        hor = art.get("horizon_explanation", {})
        combo = art.get("combination_explanation", {})
        weight_lookup[art.get("title", "")] = {
            "recency_wt": rec.get("recency_weight", 0),
            "event_type": hor.get("event_type", "?"),
            "horizon_cat": hor.get("horizon_category", "?"),
            "horizon_wt": hor.get("impact_horizon_weight", 0),
        }

    rows = []
    for art in ranked:
        title = art.get("title", "?")
        winfo = weight_lookup.get(title, {})
        net_dir = contribs_lookup.get(title, 0)

        rows.append({
            "Rank": art.get("rank", ""),
            "Title": title,
            "Sentiment": art.get("dominant_sentiment", "?").capitalize(),
            "Weight": round(art.get("final_weight", 0), 4),
            "Share": f"{art.get('weight_share', 0):.1%}",
            "Days Ago": art.get("days_ago", "?"),
            "Recency Wt": round(winfo.get("recency_wt", 0), 4),
            "Event Type": winfo.get("event_type", "?").replace("_", " ").title(),
            "Horizon": winfo.get("horizon_cat", "?"),
            "Horizon Wt": round(winfo.get("horizon_wt", 0), 4),
            f"Net ({winner[:3].upper()} vs {runner[:3].upper()})": f"{net_dir:+.4f}",
        })

    return pd.DataFrame(rows)


def _render_article_rankings(result: dict) -> None:
    layer2 = result.get("layer_2_article", {})
    ranked = layer2.get("ranked_articles", [])

    if not ranked:
        st.warning("No article analysis data available.")
        return

    hhi = layer2.get("weight_concentration", 0)
    top_pos = layer2.get("top_positive_drivers", [])
    top_neg = layer2.get("top_negative_drivers", [])

    st.markdown("### Article Analysis")
    st.write(
        f"**Weight concentration (Herfindahl index):** {hhi:.4f}"
        f" {'(concentrated)' if hhi > 0.4 else '(well-distributed)'}"
    )
    st.write("")

    df = _build_merged_table(result)
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Title": st.column_config.TextColumn(width="large"),
            "Recency Wt": st.column_config.NumberColumn(format="%.4f"),
            "Horizon Wt": st.column_config.NumberColumn(format="%.4f"),
        },
    )

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Top Positive Drivers")
        if top_pos:
            for t in top_pos:
                st.write(f"- {t}")
        else:
            st.write("_none_")
    with c2:
        st.markdown("#### Top Negative Drivers")
        if top_neg:
            for t in top_neg:
                st.write(f"- {t}")
        else:
            st.write("_none_")

    st.divider()

    content_lookup = _load_article_content()

    st.markdown("### Per-Article Details")
    for art in ranked:
        title = art.get("title", "?")
        rank = art.get("rank", "?")
        with st.expander(f"[{rank}] {title}"):
            content = content_lookup.get(title, "")
            if content:
                preview = content[:500] + ("..." if len(content) > 500 else "")
                st.markdown(
                    f'<div style="background:#f8f9fa;padding:10px 14px;border-radius:6px;'
                    f'font-size:0.9rem;margin-bottom:12px;border-left:3px solid #3498db;">'
                    f'<strong>Content:</strong> {preview}</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.caption("_No article content available (headline only)._")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.write(f"**Sentiment:** {art.get('dominant_sentiment', '?').capitalize()}")
                st.write(f"**Final weight:** {art.get('final_weight', 0):.4f}")
                st.write(f"**Weight share:** {art.get('weight_share', 0):.1%}")
                st.write(f"**Days ago:** {art.get('days_ago', '?')}")
            with col2:
                raw = art.get("raw_scores", {})
                st.write("**Raw scores:**")
                for lbl in ["positive", "negative", "neutral"]:
                    st.write(f"&nbsp;&nbsp;{lbl}: {raw.get(lbl, 0):.4f}")
            with col3:
                weighted = art.get("weighted_scores", {})
                st.write("**Weighted scores:**")
                for lbl in ["positive", "negative", "neutral"]:
                    st.write(f"&nbsp;&nbsp;{lbl}: {weighted.get(lbl, 0):.4f}")

            contrib = art.get("contribution_share", {})
            if contrib:
                st.write("**Contribution per label:**")
                for lbl in ["positive", "negative", "neutral"]:
                    val = contrib.get(lbl, 0)
                    st.write(f"&nbsp;&nbsp;{lbl}: {val:.2%}")

            cf = art.get("counterfactual", {})
            if cf:
                would_flip = cf.get("label_would_change", False)
                new_lbl = cf.get("new_label", "?")
                new_conf = cf.get("new_confidence", 0)
                if would_flip:
                    st.warning(
                        f"Removing this article would flip the prediction to "
                        f"**{new_lbl.upper()}** (confidence {new_conf:.2%})"
                    )
                else:
                    st.caption(
                        f"Removing this article: label stays **{new_lbl}** "
                        f"(confidence {new_conf:.2%})"
                    )


def _render_robustness(result: dict) -> None:
    layer2 = result.get("layer_2_article", {})
    flip_set = layer2.get("minimum_flip_set", {})
    flip_articles = layer2.get("label_flipping_articles", [])
    pred = result.get("prediction_summary", {})
    label = pred.get("final_label", "?").upper()

    st.markdown("### Prediction Robustness")
    st.write(
        "This section tests how stable the prediction is. "
        "If removing a small number of articles would change the predicted label, "
        "the prediction is sensitive to its evidence base."
    )

    st.divider()

    st.markdown("### Label-Flipping Articles")
    st.caption(
        "These are individual articles whose removal alone would change the prediction. "
        "A high number suggests the prediction depends heavily on specific evidence."
    )
    if flip_articles:
        st.warning(
            f"**{len(flip_articles)} article{'s' if len(flip_articles) != 1 else ''}** "
            f"can individually flip the **{label}** prediction:"
        )
        for title in flip_articles:
            st.write(f"- {title}")
    else:
        st.success("No single article removal would flip the predicted label.")

    st.divider()

    st.markdown("### Minimum Flip Set")
    st.caption(
        "The smallest group of articles that, if removed together, would change the prediction."
    )
    if flip_set and flip_set.get("flip_possible"):
        n_flip = flip_set.get("flip_set_size", "?")
        new_label = flip_set.get("new_label", "?")
        st.warning(
            f"Removing just **{n_flip}** article{'s' if n_flip != 1 else ''} "
            f"would flip the prediction from **{label}** to **{new_label.upper()}**:"
        )
        for title in flip_set.get("flip_set_titles", []):
            st.write(f"- {title}")
    else:
        st.success(
            "No feasible combination of article removals would flip the predicted label. "
            "The prediction is robust."
        )


def _render_weighting(result: dict) -> None:
    layer3 = result.get("layer_3_pipeline", {})

    if not layer3:
        st.warning("No pipeline weighting data available.")
        return

    window = layer3.get("prediction_window_days", "?")
    formula = layer3.get("weight_formula", "")
    avg_days = layer3.get("avg_days_ago", "?")
    avg_recency = layer3.get("avg_recency_weight", "?")
    avg_horizon = layer3.get("avg_horizon_weight", "?")

    st.markdown("### Weight Formula")
    st.code(formula, language=None)
    st.caption(f"Prediction window: **{window}** days")

    st.divider()

    st.markdown("### Average Weights")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Days Ago", avg_days)
    c2.metric("Avg Recency Weight", f"{avg_recency:.4f}" if isinstance(avg_recency, (int, float)) else "?")
    c3.metric("Avg Horizon Weight", f"{avg_horizon:.4f}" if isinstance(avg_horizon, (int, float)) else "?")

    st.divider()

    horizon_dist = layer3.get("horizon_distribution", {})
    if horizon_dist:
        st.markdown("### Horizon Category Distribution")
        h_cols = st.columns(len(horizon_dist))
        for i, (cat, count) in enumerate(horizon_dist.items()):
            colour = _HORIZON_COLOURS.get(cat, "#3498db")
            with h_cols[i]:
                st.markdown(
                    _badge(cat.replace("_", " "), colour),
                    unsafe_allow_html=True,
                )
                st.metric("Articles", count, label_visibility="collapsed")

    st.divider()

    articles = layer3.get("articles", [])
    if articles:
        st.markdown("### Detailed Weight Interpretations")
        for art in articles[:20]:
            title = art.get("title", "?")
            with st.expander(title):
                rec = art.get("recency_explanation", {})
                hor = art.get("horizon_explanation", {})
                combo = art.get("combination_explanation", {})
                st.write(f"**Recency:** {rec.get('interpretation', '')}")
                st.write(f"**Horizon:** {hor.get('interpretation', '')}")
                st.write(f"**Combined:** {combo.get('interpretation', '')}")


def _render_lime(result: dict) -> None:
    lime_data = result.get("layer_1_token", {})
    articles = lime_data.get("articles", [])
    lime_method = lime_data.get("method", "LIME")
    lime_top_n = lime_data.get("lime_top_n", "?")

    if not articles:
        st.warning("No LIME token attribution data available.")
        return

    st.markdown("### LIME Token Attribution")
    st.caption(
        f"Method: **{lime_method}** | Top articles analysed: **{lime_top_n}**"
    )
    st.caption(
        "LIME perturbs the input text and measures how each word affects the model's prediction. "
        "**Supporting** tokens push toward the predicted label; **opposing** tokens push against it."
    )
    st.write("")

    for art in articles:
        rank = art.get("rank", "?")
        title = art.get("title", "?")
        influence = art.get("influence_score", 0)
        weight = art.get("final_weight", 0)
        explained_label = art.get("lime_label_explained", "?")

        with st.expander(f"[{rank}] {title}"):

            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Explaining Label", explained_label.upper())
            mc2.metric("Influence Score", f"{influence:.4f}")
            mc3.metric("Article Weight", f"{weight:.4f}")

            content = art.get("content", "")
            if content:
                preview = content[:400] + ("..." if len(content) > 400 else "")
                st.caption(f"**Content:** {preview}")
            else:
                st.caption("**Content:** _(none -- headline only)_")

            st.write("")

            supporting = art.get("top_tokens_supporting", [])
            opposing = art.get("top_tokens_opposing", [])

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Supporting tokens:**")
                if supporting:
                    badges = " ".join(
                        _badge(t, "#2ecc71") for t in supporting
                    )
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.write("_none_")

            with col2:
                st.markdown("**Opposing tokens:**")
                if opposing:
                    badges = " ".join(
                        _badge(t, "#e74c3c") for t in opposing
                    )
                    st.markdown(badges, unsafe_allow_html=True)
                else:
                    st.write("_none_")

            st.write("")

            token_weights = art.get("token_weights", [])
            if token_weights:
                tw_df = pd.DataFrame(token_weights)
                tw_df = tw_df.sort_values("weight", ascending=False, key=abs).reset_index(drop=True)
                st.dataframe(
                    tw_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "weight": st.column_config.NumberColumn(format="%.6f"),
                    },
                )


def _chart_sentiment_scores(result: dict) -> go.Figure:
    pred = result.get("prediction_summary", {})
    ns = pred.get("normalized_scores", {})
    final = pred.get("final_label", "").lower()
    conf = pred.get("final_confidence", 0)

    labels = ["Neutral", "Negative", "Positive"]
    scores = [ns.get("neutral", 0), ns.get("negative", 0), ns.get("positive", 0)]
    colors = [_LABEL_COLOURS["neutral"], _LABEL_COLOURS["negative"], _LABEL_COLOURS["positive"]]

    border_widths = []
    for lbl_key in ["neutral", "negative", "positive"]:
        border_widths.append(3 if lbl_key == final else 0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels, x=scores, orientation="h",
        marker_color=colors,
        marker_line_width=border_widths,
        marker_line_color="#2c3e50",
        text=[f"{s:.1%}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Sentiment Scores — Verdict: {final.upper()} ({conf:.1%} confidence)",
        xaxis_title="Score", xaxis=dict(range=[0, 1.15]),
        template="plotly_white", height=300, margin=dict(l=80),
    )
    return fig


def _chart_article_distribution(result: dict) -> go.Figure:
    layer2 = result.get("layer_2_article", {})
    ranked = layer2.get("ranked_articles", [])
    counts: dict[str, int] = {}
    for a in ranked:
        s = a.get("dominant_sentiment", "unknown")
        counts[s] = counts.get(s, 0) + 1

    if not counts:
        return go.Figure().update_layout(title="No article data available")

    fig = go.Figure(data=[go.Pie(
        labels=[k.capitalize() for k in counts],
        values=list(counts.values()),
        marker_colors=[_LABEL_COLOURS.get(k, "#95a5a6") for k in counts],
        hole=0.3,
        textinfo="label+percent+value",
    )])
    fig.update_layout(
        title=f"Article Sentiment Distribution ({sum(counts.values())} articles)",
        template="plotly_white", height=450,
    )
    return fig


def _chart_article_weights(result: dict) -> go.Figure:
    layer2 = result.get("layer_2_article", {})
    ranked = layer2.get("ranked_articles", [])[:10]

    if not ranked:
        return go.Figure().update_layout(title="No article data available")

    titles = []
    weights = []
    colors = []
    for a in ranked:
        t = a.get("title", "?")
        titles.append(f"#{a['rank']} {t[:45]}..." if len(t) > 45 else f"#{a['rank']} {t}")
        weights.append(a.get("final_weight", 0))
        colors.append(_LABEL_COLOURS.get(a.get("dominant_sentiment", "neutral"), "#95a5a6"))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=titles[::-1], x=weights[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{w:.4f}" for w in weights[::-1]],
        textposition="outside",
    ))
    fig.update_layout(
        title="Top 10 Most Influential Articles",
        xaxis_title="Final Weight",
        template="plotly_white", height=max(350, len(ranked) * 55),
        margin=dict(l=300),
    )
    return fig


def _chart_horizon_breakdown(result: dict) -> go.Figure:
    layer3 = result.get("layer_3_pipeline", {})
    horizon_dist = layer3.get("horizon_distribution", {})
    horizon_labels = {
        "IMMEDIATE": "Breaking / same-day",
        "SHORT_TERM": "Short-term (days)",
        "MEDIUM_TERM": "Medium-term (weeks)",
        "LONG_TERM": "Long-term (months)",
    }

    cats = list(horizon_dist.keys())
    counts = list(horizon_dist.values())
    xlabels = [horizon_labels.get(c, c) for c in cats]
    colors = [_HORIZON_COLOURS.get(c, "#3498db") for c in cats]

    if not counts:
        return go.Figure().update_layout(title="No horizon data available")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=xlabels, y=counts, marker_color=colors,
        text=counts, textposition="outside",
    ))
    fig.update_layout(
        title="Article Timing Horizon Breakdown",
        yaxis_title="Number of Articles",
        template="plotly_white", height=400,
    )
    return fig


def _chart_lime_tokens(result: dict) -> go.Figure:
    lime_data = result.get("layer_1_token", {})
    articles = lime_data.get("articles", [])
    pred = result.get("prediction_summary", {})
    predicted_label = pred.get("final_label", "positive")

    if not articles:
        return go.Figure().update_layout(title="No LIME data available")

    n = len(articles)
    subtitles = [f"[{a.get('rank', '?')}] {a.get('title', '?')[:60]}" for a in articles]
    fig = make_subplots(
        rows=n, cols=1, subplot_titles=subtitles,
        vertical_spacing=max(0.02, 0.15 / n),
    )

    for idx, art in enumerate(articles):
        tw = sorted(art.get("token_weights", []), key=lambda x: x["weight"], reverse=True)[:12]
        if not tw:
            continue
        tokens = [t["token"] for t in tw]
        weights = [t["weight"] for t in tw]
        colors = [_LABEL_COLOURS["positive"] if w >= 0 else _LABEL_COLOURS["negative"] for w in weights]

        fig.add_trace(go.Bar(
            y=tokens[::-1], x=weights[::-1], orientation="h",
            marker_color=colors[::-1], showlegend=False,
            text=[f"{w:.4f}" for w in weights[::-1]], textposition="outside",
        ), row=idx + 1, col=1)

    fig.update_layout(
        title=f"Word-Level Attribution (LIME) — {predicted_label.upper()} label",
        template="plotly_white", height=max(400, n * 300),
    )
    return fig


def _chart_reliability(result: dict) -> go.Figure:
    reliability = result.get("reliability", {})
    flags = reliability.get("flags", {})
    overall = reliability.get("overall_reliability", "UNKNOWN")

    names = []
    colors = []
    status_labels = []
    hover_msgs = []
    for key, data in flags.items():
        flagged = data.get("flagged", False)
        names.append(key.replace("_", " ").title())
        colors.append(_LABEL_COLOURS["negative"] if flagged else _LABEL_COLOURS["positive"])
        status_labels.append("FLAGGED" if flagged else "PASS")
        hover_msgs.append(data.get("message", ""))

    if not names:
        return go.Figure().update_layout(title="No reliability data available")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=names[::-1], x=[1] * len(names), orientation="h",
        marker_color=colors[::-1],
        text=status_labels[::-1], textposition="inside",
        textfont=dict(color="white", size=13),
        hovertext=hover_msgs[::-1], hoverinfo="text",
    ))
    fig.update_layout(
        title=f"Reliability Dashboard — Overall: {overall}",
        xaxis=dict(visible=False),
        template="plotly_white", height=max(280, len(names) * 45),
        margin=dict(l=160),
    )
    return fig


def _chart_storyline_contribution(result: dict) -> go.Figure:
    storylines_data = result.get("storylines", {})
    pred = result.get("prediction_summary", {})
    predicted_label = pred.get("final_label", "positive")
    storylines = storylines_data.get("storylines", [])

    if not storylines:
        return go.Figure().update_layout(title="No storyline data available")

    sorted_sl = sorted(storylines, key=lambda s: s.get("contribution_score", 0), reverse=True)
    labels = []
    scores = []
    colors = []
    for sl in sorted_sl:
        lbl = sl.get("label") or sl.get("keyword_label", "Unknown")
        cnt = sl.get("articles_count", 0)
        grp = sl.get("sentiment_group", "neutral")
        labels.append(f"{lbl} ({cnt} art.)")
        scores.append(sl.get("contribution_score", 0))
        colors.append(_LABEL_COLOURS.get(grp, "#95a5a6"))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels[::-1], x=scores[::-1], orientation="h",
        marker_color=colors[::-1],
        text=[f"{s:.3f}" for s in scores[::-1]], textposition="outside",
    ))
    fig.update_layout(
        title=f"Narrative Storylines — {predicted_label.upper()} prediction",
        xaxis_title="Contribution Score",
        template="plotly_white", height=max(350, len(labels) * 40),
        margin=dict(l=250),
    )
    return fig


def _chart_contrastive_waterfall(result: dict) -> go.Figure:
    layer2 = result.get("layer_2_article", {})
    contrastive = layer2.get("contrastive", {})
    winner = contrastive.get("winner", "?")
    runner_up = contrastive.get("runner_up", "?")
    score_gap = contrastive.get("score_gap", 0)
    all_contribs = contrastive.get("all_contributions", [])

    if not all_contribs:
        return go.Figure().update_layout(title="No contrastive data available")

    top = sorted(all_contribs, key=lambda a: abs(a["net_direction"]), reverse=True)[:15]
    labels = [a["title"][:40] + ("..." if len(a["title"]) > 40 else "") for a in top]
    values = [a["net_direction"] for a in top]

    fig = go.Figure(go.Waterfall(
        y=labels, x=values, orientation="h",
        measure=["relative"] * len(values),
        connector=dict(line=dict(color="#7f8c8d", width=0.8, dash="dot")),
        increasing=dict(marker=dict(color=_LABEL_COLOURS["positive"])),
        decreasing=dict(marker=dict(color=_LABEL_COLOURS["negative"])),
        text=[f"{v:+.4f}" for v in values], textposition="outside",
    ))
    fig.update_layout(
        title=(
            f"Contrastive Waterfall — {winner.upper()} vs {runner_up.upper()} "
            f"(gap = {score_gap:.4f})"
        ),
        xaxis_title=f"Net contribution (+ favours {winner.upper()})",
        template="plotly_white", height=max(400, len(labels) * 40),
        margin=dict(l=280),
    )
    return fig


def _chart_article_timeline(result: dict) -> go.Figure:
    layer2 = result.get("layer_2_article", {})
    ranked = layer2.get("ranked_articles", [])

    if not ranked:
        return go.Figure().update_layout(title="No article data available")

    rows = []
    for a in ranked:
        rows.append({
            "days_ago": a.get("days_ago", 0),
            "weight": a.get("final_weight", 0.01),
            "sentiment": a.get("dominant_sentiment", "neutral").capitalize(),
            "title": a.get("title", "?")[:60],
        })

    df = pd.DataFrame(rows)
    color_map = {
        "Positive": _LABEL_COLOURS["positive"],
        "Negative": _LABEL_COLOURS["negative"],
        "Neutral": _LABEL_COLOURS["neutral"],
    }

    fig = px.scatter(
        df, x="days_ago", y="weight", color="sentiment",
        size="weight", hover_data=["title"],
        color_discrete_map=color_map,
    )
    fig.update_layout(
        title="Article Timeline — Recency vs Influence",
        xaxis_title="Days Ago (0 = today)", yaxis_title="Article Weight",
        xaxis=dict(autorange="reversed"),
        template="plotly_white", height=450,
    )
    return fig


def _chart_cumulative_score(result: dict) -> go.Figure:
    layer2 = result.get("layer_2_article", {})
    pred = result.get("prediction_summary", {})
    ranked = layer2.get("ranked_articles", [])
    predicted_label = pred.get("final_label", "positive")

    if not ranked:
        return go.Figure().update_layout(title="No article data available")

    running = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    cum = {"positive": [], "negative": [], "neutral": []}
    x_labels = []

    for i, a in enumerate(ranked):
        ws = a.get("weighted_scores", {})
        for lbl in running:
            running[lbl] += ws.get(lbl, 0)
            cum[lbl].append(running[lbl])
        x_labels.append(i + 1)

    norm = {"positive": [], "negative": [], "neutral": []}
    for i in range(len(x_labels)):
        total = cum["positive"][i] + cum["negative"][i] + cum["neutral"][i]
        for lbl in norm:
            norm[lbl].append(cum[lbl][i] / total if total > 0 else 0)

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=[
            "Cumulative Weighted Scores",
            "Normalised Sentiment Proportion",
        ],
        vertical_spacing=0.12,
    )

    for name, colour in [
        ("positive", _LABEL_COLOURS["positive"]),
        ("negative", _LABEL_COLOURS["negative"]),
        ("neutral", _LABEL_COLOURS["neutral"]),
    ]:
        fig.add_trace(go.Scatter(
            x=x_labels, y=cum[name], mode="lines", name=name.capitalize(),
            line=dict(color=colour, width=2),
            fill="tozeroy", fillcolor=_hex_to_rgba(colour, 0.15),
        ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x_labels, y=norm["positive"], mode="lines", name="Positive (norm)",
        line=dict(color=_LABEL_COLOURS["positive"], width=0.5),
        stackgroup="one",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x_labels, y=norm["negative"], mode="lines", name="Negative (norm)",
        line=dict(color=_LABEL_COLOURS["negative"], width=0.5),
        stackgroup="one",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=x_labels, y=norm["neutral"], mode="lines", name="Neutral (norm)",
        line=dict(color=_LABEL_COLOURS["neutral"], width=0.5),
        stackgroup="one",
    ), row=2, col=1)

    fig.update_layout(
        title=f"Prediction Build-Up — {predicted_label.upper()}",
        template="plotly_white", height=700,
    )
    fig.update_xaxes(title_text="Articles Added (sorted by weight)", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative Score", row=1, col=1)
    fig.update_yaxes(title_text="Proportion", row=2, col=1, range=[0, 1])
    return fig


_CHART_OPTIONS = {
    "Sentiment Scores": _chart_sentiment_scores,
    "Article Distribution": _chart_article_distribution,
    "Top 10 Article Weights": _chart_article_weights,
    "Horizon Breakdown": _chart_horizon_breakdown,
    "LIME Token Attribution": _chart_lime_tokens,
    "Reliability Dashboard": _chart_reliability,
    "Storyline Contribution": _chart_storyline_contribution,
    "Contrastive Waterfall": _chart_contrastive_waterfall,
    "Article Timeline": _chart_article_timeline,
    "Cumulative Score Build-Up": _chart_cumulative_score,
}


def _render_charts(result: dict) -> None:
    st.markdown("### Interactive Charts")

    selected = st.selectbox("Select a chart to view", list(_CHART_OPTIONS.keys()))

    builder = _CHART_OPTIONS[selected]
    fig = builder(result)
    st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.title("Stock Sentiment Analyser")
    st.caption("Zero-shot NLI sentiment prediction with explainable AI")

    with st.sidebar:
        st.header("Analysis Parameters")

        company_name = st.text_input(
            "Company Name",
            value="Apple Inc.",
            help="Enter the company name or ticker symbol",
        )

        today = date.today()
        default_start = today - timedelta(days=7)

        col_d1, col_d2 = st.columns(2)
        with col_d1:
            start_date = st.date_input("Start Date", value=default_start)
        with col_d2:
            end_date = st.date_input("End Date", value=today)

        num_articles = st.slider(
            "Max Articles",
            min_value=10,
            max_value=500,
            value=250,
            step=10,
            help="Maximum number of articles to include after filtering",
        )

        st.divider()
        run_btn = st.button("Run Analysis", type="primary", use_container_width=True)

        st.divider()
        st.caption(f"Model: `{MODEL_NAME}`")
        st.caption(f"Pipeline: `{SENTIMENT_MODEL}`")

    if run_btn:
        if not company_name.strip():
            st.error("Please enter a company name.")
            return

        if end_date < start_date:
            st.error("End date cannot be before start date.")
            return

        start_str = start_date.strftime("%d-%m-%Y")
        end_str = end_date.strftime("%d-%m-%Y")

        xai_result = _run_full_pipeline(company_name, start_str, end_str, num_articles)

        if xai_result is not None:
            st.session_state["xai_result"] = xai_result

    result = st.session_state.get("xai_result")

    if result is None:
        st.info(
            "Configure the analysis parameters in the sidebar and click "
            "**Run Analysis** to start."
        )
        return

    (
        tab_overview,
        tab_reliability,
        tab_storylines,
        tab_rankings,
        tab_robustness,
        tab_weighting,
        tab_lime,
        tab_charts,
    ) = st.tabs([
        "Overview",
        "Reliability",
        "Storylines",
        "Article Analysis",
        "Robustness",
        "Weighting",
        "LIME Tokens",
        "Charts",
    ])

    with tab_overview:
        _render_overview(result)

    with tab_reliability:
        _render_reliability(result)

    with tab_storylines:
        _render_storylines(result)

    with tab_rankings:
        _render_article_rankings(result)

    with tab_robustness:
        _render_robustness(result)

    with tab_weighting:
        _render_weighting(result)

    with tab_lime:
        _render_lime(result)

    with tab_charts:
        _render_charts(result)


if __name__ == "__main__":
    main()
