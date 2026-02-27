"""
charts.py — Matplotlib PNG chart generators for the XAI report.

Each function accepts the relevant slice of the XAI result dict and a
``out_dir`` (Path).  All functions return the Path of the saved PNG so
the caller can embed the filename in the text summary.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Colour palette — consistent across all charts
_COL = {
    "positive": "#2ecc71",   # green
    "negative": "#e74c3c",   # red
    "neutral":  "#95a5a6",   # grey
    "bar":      "#3498db",   # blue (generic bars)
    "bar_alt":  "#9b59b6",   # purple (secondary bars)
    "warn":     "#e67e22",   # orange (flagged items)
    "ok":       "#27ae60",   # dark green (clean items)
    "bg":       "#fafafa",
    "grid":     "#dde1e7",
}

_FONT = "DejaVu Sans"


def _savefig(fig, path: Path) -> Path:
    """Save figure to path and close it."""
    import matplotlib
    matplotlib.use("Agg")   # non-interactive backend — safe on all OSes
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.clf()
    import matplotlib.pyplot as plt
    plt.close(fig)
    logger.info("Chart saved: %s", path)
    return path


# ── 1. Sentiment score bar chart ─────────────────────────────────────────────

def plot_sentiment_scores(
    pred: dict[str, Any],
    out_dir: Path,
    filename: str = "01_sentiment_scores.png",
) -> Path:
    """Horizontal bar chart of positive / negative / neutral scores."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ns     = pred.get("normalized_scores") or {}
    labels = ["Positive", "Negative", "Neutral"]
    scores = [ns.get("positive", 0.0), ns.get("negative", 0.0), ns.get("neutral", 0.0)]
    colors = [_COL["positive"], _COL["negative"], _COL["neutral"]]
    final  = str(pred.get("final_label", "")).lower()

    fig, ax = plt.subplots(figsize=(7, 2.8), facecolor=_COL["bg"])
    ax.set_facecolor(_COL["bg"])

    bars = ax.barh(labels, scores, color=colors, height=0.5, edgecolor="white", linewidth=0.8)

    # Highlight predicted bar with a border
    for bar, lbl in zip(bars, ["positive", "negative", "neutral"]):
        if lbl == final:
            bar.set_edgecolor("#2c3e50")
            bar.set_linewidth(2.2)

    # Percentage labels
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{score * 100:.1f}%",
            va="center", ha="left", fontsize=10, fontname=_FONT,
        )

    # "PREDICTED" annotation
    for bar, lbl in zip(bars, ["positive", "negative", "neutral"]):
        if lbl == final:
            ax.annotate(
                "◄ PREDICTED",
                xy=(bar.get_width(), bar.get_y() + bar.get_height() / 2),
                xytext=(bar.get_width() + 0.12, bar.get_y() + bar.get_height() / 2),
                fontsize=8, color="#2c3e50", va="center", fontname=_FONT,
            )

    ax.set_xlim(0, 1.35)
    ax.set_xlabel("Score", fontsize=10, fontname=_FONT)
    ax.set_title(
        f"Sentiment Scores  —  Verdict: {final.upper()}  "
        f"({pred.get('final_confidence', 0) * 100:.1f}% confidence)",
        fontsize=11, fontname=_FONT, pad=10,
    )
    ax.xaxis.grid(True, color=_COL["grid"], linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _savefig(fig, out_dir / filename)


# ── 2. Article sentiment distribution pie ────────────────────────────────────

def plot_article_distribution(
    layer2: dict[str, Any],
    out_dir: Path,
    filename: str = "02_article_distribution.png",
) -> Path:
    """Pie chart showing how many articles were positive / negative / neutral."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = layer2.get("ranked_articles", [])
    counts: dict[str, int] = {}
    for a in ranked:
        s = a.get("dominant_sentiment", "unknown")
        counts[s] = counts.get(s, 0) + 1

    labels  = [k.capitalize() for k in counts]
    sizes   = list(counts.values())
    colors  = [_COL.get(k, _COL["neutral"]) for k in counts]
    explode = [0.04] * len(labels)

    if not sizes:
        return out_dir / filename   # nothing to plot

    fig, ax = plt.subplots(figsize=(5, 4), facecolor=_COL["bg"])
    ax.set_facecolor(_COL["bg"])

    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct="%1.0f%%", startangle=90,
        textprops={"fontsize": 10, "fontname": _FONT},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontsize(9)

    ax.set_title(
        f"Article Sentiment Distribution  ({sum(sizes)} articles)",
        fontsize=11, fontname=_FONT, pad=12,
    )
    fig.tight_layout()
    return _savefig(fig, out_dir / filename)


# ── 3. Top-10 article weight bar chart ───────────────────────────────────────

def plot_article_weights(
    layer2: dict[str, Any],
    out_dir: Path,
    filename: str = "03_article_weights.png",
) -> Path:
    """Horizontal bar chart of the top-10 most influential articles."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ranked = layer2.get("ranked_articles", [])[:10]
    if not ranked:
        return out_dir / filename

    titles  = [f"#{a['rank']} {a['title'][:45]}…" if len(a['title']) > 45
               else f"#{a['rank']} {a['title']}" for a in ranked]
    weights = [a["final_weight"] for a in ranked]
    colors  = [_COL.get(a["dominant_sentiment"], _COL["neutral"]) for a in ranked]

    fig, ax = plt.subplots(figsize=(9, max(3.5, len(ranked) * 0.55)), facecolor=_COL["bg"])
    ax.set_facecolor(_COL["bg"])

    bars = ax.barh(titles[::-1], weights[::-1], color=colors[::-1],
                   height=0.6, edgecolor="white", linewidth=0.8)

    for bar, w in zip(bars, weights[::-1]):
        ax.text(
            bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{w:.4f}",
            va="center", ha="left", fontsize=8, fontname=_FONT,
        )

    ax.set_xlabel("Final Weight", fontsize=10, fontname=_FONT)
    ax.set_title("Top 10 Most Influential Articles", fontsize=11, fontname=_FONT, pad=10)
    ax.xaxis.grid(True, color=_COL["grid"], linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Colour legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=_COL["positive"], label="Positive"),
        Patch(facecolor=_COL["negative"], label="Negative"),
        Patch(facecolor=_COL["neutral"],  label="Neutral"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              prop={"family": _FONT, "size": 8})

    fig.tight_layout()
    return _savefig(fig, out_dir / filename)


# ── 4. Horizon timing breakdown bar chart ────────────────────────────────────

def plot_horizon_breakdown(
    layer3: dict[str, Any],
    out_dir: Path,
    filename: str = "04_horizon_breakdown.png",
) -> Path:
    """Bar chart of how articles are distributed across timing horizons."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    horizon_dist   = layer3.get("horizon_distribution", {})
    horizon_labels = {
        "IMMEDIATE":   "Breaking / same-day",
        "SHORT_TERM":  "Short-term (days)",
        "MEDIUM_TERM": "Medium-term (weeks)",
        "LONG_TERM":   "Long-term (months)",
    }
    cats   = list(horizon_dist.keys())
    counts = list(horizon_dist.values())
    xlabels = [horizon_labels.get(c, c) for c in cats]

    if not counts:
        return out_dir / filename

    colors = [_COL["bar"]] * len(cats)

    fig, ax = plt.subplots(figsize=(6.5, 3.5), facecolor=_COL["bg"])
    ax.set_facecolor(_COL["bg"])

    bars = ax.bar(xlabels, counts, color=colors, width=0.5,
                  edgecolor="white", linewidth=0.8)

    for bar, cnt in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
            str(cnt),
            ha="center", va="bottom", fontsize=10, fontname=_FONT,
        )

    ax.set_ylabel("Number of Articles", fontsize=10, fontname=_FONT)
    ax.set_title("Article Timing Horizon Breakdown", fontsize=11, fontname=_FONT, pad=10)
    ax.yaxis.grid(True, color=_COL["grid"], linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    plt.xticks(fontsize=9, fontname=_FONT)

    fig.tight_layout()
    return _savefig(fig, out_dir / filename)


# ── 5. LIME token attribution chart (per article) ────────────────────────────

def plot_lime_tokens(
    lime_articles: list[dict[str, Any]],
    predicted_label: str,
    out_dir: Path,
    filename: str = "05_lime_tokens.png",
) -> Path:
    """
    Horizontal diverging bar chart: green bars = supports predicted label,
    red bars = opposes predicted label.  One subplot per article.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not lime_articles:
        return out_dir / filename

    n_arts  = len(lime_articles)
    fig_h   = max(3.5, n_arts * 3.2)
    fig, axes = plt.subplots(
        n_arts, 1, figsize=(9, fig_h), facecolor=_COL["bg"],
        squeeze=False,
    )

    for idx, (art, ax) in enumerate(zip(lime_articles, axes[:, 0])):
        ax.set_facecolor(_COL["bg"])

        tw_sorted = sorted(
            art.get("token_weights", []),
            key=lambda x: x["weight"], reverse=True,
        )[:12]

        if not tw_sorted:
            ax.set_visible(False)
            continue

        tokens  = [t["token"] for t in tw_sorted]
        weights = [t["weight"] for t in tw_sorted]
        colors  = [_COL["positive"] if w >= 0 else _COL["negative"] for w in weights]

        bars = ax.barh(tokens[::-1], weights[::-1], color=colors[::-1],
                       height=0.6, edgecolor="white", linewidth=0.5)

        ax.axvline(0, color="#2c3e50", linewidth=0.9, linestyle="-")
        ax.xaxis.grid(True, color=_COL["grid"], linewidth=0.5, linestyle="--")
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)

        short_title = art["title"][:60] + ("…" if len(art["title"]) > 60 else "")
        ax.set_title(
            f"[{art['rank']}] {short_title}\n"
            f"weight={art['final_weight']:.4f}  influence={art['influence_score']:.4f}",
            fontsize=8.5, fontname=_FONT, loc="left", pad=4,
        )
        ax.set_xlabel(f"LIME weight  (+ = supports {predicted_label.upper()})", fontsize=8)
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        f"Word-Level Attribution (LIME) — {predicted_label.upper()} label",
        fontsize=12, fontname=_FONT, y=1.01,
    )
    fig.tight_layout()
    return _savefig(fig, out_dir / filename)


# ── 6. Reliability flag dashboard ────────────────────────────────────────────

def plot_reliability(
    reliability: dict[str, Any],
    out_dir: Path,
    filename: str = "06_reliability.png",
) -> Path:
    """
    Colour-coded table / dashboard showing each reliability flag status
    and the overall reliability level.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    flags = reliability.get("flags", {})
    overall = reliability.get("overall_reliability", "UNKNOWN")

    flag_labels = {
        "thin_evidence":        "Evidence Volume",
        "weight_concentration": "Evidence Diversity",
        "label_margin":         "Decision Confidence",
        "low_confidence":       "Score Confidence",
    }

    rows = []
    for key, data in flags.items():
        rows.append({
            "label":   flag_labels.get(key, key.replace("_", " ").title()),
            "flagged": data.get("flagged", False),
            "message": data.get("message", ""),
        })

    n = len(rows) + 1  # +1 for overall row
    fig, ax = plt.subplots(figsize=(8, 0.7 * n + 1.2), facecolor=_COL["bg"])
    ax.set_facecolor(_COL["bg"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, n)
    ax.axis("off")

    overall_color = {"HIGH": _COL["ok"], "MEDIUM": _COL["warn"], "LOW": _COL["negative"]}.get(overall, _COL["neutral"])

    # Overall row at top
    y = n - 0.85
    rect = mpatches.FancyBboxPatch(
        (0.1, y - 0.3), 9.8, 0.7,
        boxstyle="round,pad=0.05", facecolor=overall_color, alpha=0.25, linewidth=1.2,
        edgecolor=overall_color,
    )
    ax.add_patch(rect)
    ax.text(0.4, y + 0.05, "OVERALL RELIABILITY", fontsize=10,
            fontname=_FONT, va="center", fontweight="bold")
    ax.text(9.8, y + 0.05, overall,
            fontsize=11, fontname=_FONT, va="center", ha="right",
            fontweight="bold", color=overall_color)

    for i, row in enumerate(rows):
        y = n - 1.85 - i
        color = _COL["negative"] if row["flagged"] else _COL["ok"]
        icon  = "⚠  " if row["flagged"] else "✓  "

        rect = mpatches.FancyBboxPatch(
            (0.1, y - 0.28), 9.8, 0.65,
            boxstyle="round,pad=0.05", facecolor=color, alpha=0.10,
            linewidth=0.8, edgecolor=color,
        )
        ax.add_patch(rect)

        ax.text(0.35, y + 0.05, icon + row["label"],
                fontsize=9.5, fontname=_FONT, va="center",
                color="#2c3e50", fontweight="bold" if row["flagged"] else "normal")
        ax.text(9.75, y + 0.05, row["message"],
                fontsize=8.5, fontname=_FONT, va="center", ha="right",
                color="#555")

    ax.set_title("Prediction Reliability Dashboard", fontsize=12,
                 fontname=_FONT, pad=10)
    fig.tight_layout()
    return _savefig(fig, out_dir / filename)


# ── 7. Storyline contribution chart ──────────────────────────────────────────

def plot_storyline_contribution(
    storyline_data: dict[str, Any],
    predicted_label: str,
    out_dir: Path,
    filename: str = "07_storyline_contribution.png",
) -> Path:
    """
    Grouped horizontal bar chart showing narrative clusters by sentiment
    group, sized by contribution_score.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    storylines = storyline_data.get("storylines", [])
    if not storylines:
        return out_dir / filename

    # Group storylines by sentiment_group
    groups: dict[str, list] = {"positive": [], "negative": [], "neutral": []}
    for sl in storylines:
        grp = sl.get("sentiment_group", "neutral")
        groups.setdefault(grp, []).append(sl)

    # Order: predicted label first, then opposing, then neutral
    if predicted_label == "positive":
        order = ["positive", "negative", "neutral"]
    elif predicted_label == "negative":
        order = ["negative", "positive", "neutral"]
    else:
        order = ["neutral", "positive", "negative"]

    # Flatten into display rows
    labels = []
    scores = []
    colors = []
    section_breaks = []  # y-positions where sections change

    for grp in order:
        grp_sl = groups.get(grp, [])
        if not grp_sl:
            continue
        grp_sl.sort(key=lambda s: s.get("contribution_score", 0), reverse=True)
        section_breaks.append(len(labels))
        for sl in grp_sl:
            lbl_text = sl.get("label") or sl.get("keyword_label", "Unknown")
            count = sl.get("articles_count", 0)
            labels.append(f"{lbl_text} ({count} art.)")
            scores.append(sl.get("contribution_score", 0.0))
            colors.append(_COL.get(grp, _COL["neutral"]))

    if not labels:
        return out_dir / filename

    n = len(labels)
    fig_h = max(3.5, n * 0.55 + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor=_COL["bg"])
    ax.set_facecolor(_COL["bg"])

    y_pos = list(range(n))
    bars = ax.barh(
        y_pos[::-1], scores[::-1], color=colors[::-1],
        height=0.6, edgecolor="white", linewidth=0.8,
    )

    # Score labels
    for bar, s in zip(bars, scores[::-1]):
        xpos = bar.get_width()
        ax.text(
            xpos + max(abs(xpos) * 0.02, 0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{s:.3f}", va="center", ha="left", fontsize=8, fontname=_FONT,
        )

    ax.set_yticks(y_pos[::-1])
    ax.set_yticklabels(labels[::-1], fontsize=8, fontname=_FONT)
    ax.set_xlabel("Contribution Score", fontsize=10, fontname=_FONT)
    ax.set_title(
        f"Narrative Storylines by Sentiment — {predicted_label.upper()} prediction",
        fontsize=11, fontname=_FONT, pad=10,
    )
    ax.xaxis.grid(True, color=_COL["grid"], linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    legend_elements = [
        Patch(facecolor=_COL["positive"], label="Positive"),
        Patch(facecolor=_COL["negative"], label="Negative"),
        Patch(facecolor=_COL["neutral"],  label="Neutral"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8,
              prop={"family": _FONT, "size": 8})

    fig.tight_layout()
    return _savefig(fig, out_dir / filename)


# ── 8. Contrastive waterfall chart ──────────────────────────────────────────

def plot_contrastive_waterfall(
    contrastive: dict[str, Any],
    out_dir: Path,
    filename: str = "08_contrastive_waterfall.png",
) -> Path:
    """
    Waterfall chart showing how each top article contributes to the gap
    between the winning and runner-up labels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    winner     = contrastive.get("winner", "?")
    runner_up  = contrastive.get("runner_up", "?")
    score_gap  = contrastive.get("score_gap", 0)
    all_contribs = contrastive.get("all_contributions", [])

    if not all_contribs:
        return out_dir / filename

    # Take top 15 by absolute contribution
    top = sorted(all_contribs, key=lambda a: abs(a["net_direction"]), reverse=True)[:15]

    labels   = [a["title"][:40] + ("…" if len(a["title"]) > 40 else "") for a in top]
    values   = [a["net_direction"] for a in top]
    colors   = [_COL["positive"] if v >= 0 else _COL["negative"] for v in values]

    # Waterfall: cumulative running total
    cumulative = []
    running = 0.0
    starts = []
    for v in values:
        starts.append(running)
        running += v
        cumulative.append(running)

    n = len(labels)
    fig_h = max(4, n * 0.5 + 2)
    fig, ax = plt.subplots(figsize=(10, fig_h), facecolor=_COL["bg"])
    ax.set_facecolor(_COL["bg"])

    y_pos = list(range(n))

    # Draw bars floating from start to end
    for i in range(n):
        ax.barh(
            n - 1 - i, values[i], left=starts[i],
            color=colors[i], height=0.6,
            edgecolor="white", linewidth=0.8,
        )
        # Value label
        end = starts[i] + values[i]
        ax.text(
            end + 0.001 * (1 if values[i] >= 0 else -1),
            n - 1 - i,
            f"{values[i]:+.4f}", va="center",
            ha="left" if values[i] >= 0 else "right",
            fontsize=7.5, fontname=_FONT,
        )

    # Connector lines between bars
    for i in range(n - 1):
        y_from = n - 1 - i - 0.3
        y_to   = n - 1 - i - 0.7
        x      = cumulative[i]
        ax.plot([x, x], [y_from, y_to], color="#7f8c8d", linewidth=0.6, linestyle=":")

    ax.axvline(0, color="#2c3e50", linewidth=0.9, linestyle="-")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels[::-1], fontsize=8, fontname=_FONT)
    ax.set_xlabel(
        f"Net contribution to gap  (+ favours {winner.upper()}, - favours {runner_up.upper()})",
        fontsize=9, fontname=_FONT,
    )
    ax.set_title(
        f"Contrastive Waterfall — Why {winner.upper()} instead of {runner_up.upper()}?  "
        f"(gap = {score_gap:.4f})",
        fontsize=11, fontname=_FONT, pad=10,
    )
    ax.xaxis.grid(True, color=_COL["grid"], linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _savefig(fig, out_dir / filename)


# ── 9. Article timeline scatter ─────────────────────────────────────────────

def plot_article_timeline(
    ranked_articles: list[dict[str, Any]],
    out_dir: Path,
    filename: str = "09_article_timeline.png",
) -> Path:
    """
    Scatter plot of articles over time (days_ago), sized by final_weight,
    coloured by dominant sentiment.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    if not ranked_articles:
        return out_dir / filename

    days   = []
    weights = []
    colors = []
    for a in ranked_articles:
        d = a.get("days_ago")
        if d is None:
            # Try to get from the ranked article data
            d = 0
        days.append(d)
        weights.append(a.get("final_weight", 0.01))
        colors.append(_COL.get(a.get("dominant_sentiment", "neutral"), _COL["neutral"]))

    # Size: normalise weights to point sizes (min 30, max 400)
    w_arr = np.array(weights)
    if w_arr.max() > w_arr.min():
        sizes = 30 + 370 * (w_arr - w_arr.min()) / (w_arr.max() - w_arr.min())
    else:
        sizes = np.full_like(w_arr, 120)

    fig, ax = plt.subplots(figsize=(10, 4.5), facecolor=_COL["bg"])
    ax.set_facecolor(_COL["bg"])

    ax.scatter(days, weights, s=sizes, c=colors, alpha=0.7,
               edgecolors="white", linewidths=0.6)

    ax.set_xlabel("Days Ago (0 = today)", fontsize=10, fontname=_FONT)
    ax.set_ylabel("Article Weight", fontsize=10, fontname=_FONT)
    ax.set_title("Article Timeline — Recency vs Influence",
                 fontsize=11, fontname=_FONT, pad=10)

    ax.invert_xaxis()  # most recent on the right
    ax.xaxis.grid(True, color=_COL["grid"], linewidth=0.5, linestyle="--")
    ax.yaxis.grid(True, color=_COL["grid"], linewidth=0.5, linestyle="--")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    legend_elements = [
        Patch(facecolor=_COL["positive"], label="Positive"),
        Patch(facecolor=_COL["negative"], label="Negative"),
        Patch(facecolor=_COL["neutral"],  label="Neutral"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8,
              prop={"family": _FONT, "size": 8})

    fig.tight_layout()
    return _savefig(fig, out_dir / filename)


# ── 10. Cumulative score build-up chart ─────────────────────────────────────

def plot_cumulative_score(
    ranked_articles: list[dict[str, Any]],
    predicted_label: str,
    out_dir: Path,
    filename: str = "10_cumulative_score.png",
) -> Path:
    """
    Area chart showing how the weighted sentiment score accumulates as each
    article is added (sorted by weight descending).  Visualises how the
    prediction 'forms' as evidence builds up.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not ranked_articles:
        return out_dir / filename

    # Articles already sorted by weight (descending)
    cum_pos = []
    cum_neg = []
    cum_neu = []
    running = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    x_labels = []

    for i, a in enumerate(ranked_articles):
        ws = a.get("weighted_scores", {})
        for lbl in running:
            running[lbl] += ws.get(lbl, 0.0)
        cum_pos.append(running["positive"])
        cum_neg.append(running["negative"])
        cum_neu.append(running["neutral"])
        x_labels.append(i + 1)

    # Normalise each step to show running proportions
    norm_pos, norm_neg, norm_neu = [], [], []
    for p, ng, nu in zip(cum_pos, cum_neg, cum_neu):
        total = p + ng + nu
        if total > 0:
            norm_pos.append(p / total)
            norm_neg.append(ng / total)
            norm_neu.append(nu / total)
        else:
            norm_pos.append(0)
            norm_neg.append(0)
            norm_neu.append(0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), facecolor=_COL["bg"],
                                    gridspec_kw={"height_ratios": [1, 1]})

    # --- Top panel: cumulative raw weighted scores ---
    ax1.set_facecolor(_COL["bg"])
    ax1.fill_between(x_labels, cum_pos, alpha=0.3, color=_COL["positive"])
    ax1.fill_between(x_labels, cum_neg, alpha=0.3, color=_COL["negative"])
    ax1.fill_between(x_labels, cum_neu, alpha=0.2, color=_COL["neutral"])
    ax1.plot(x_labels, cum_pos, color=_COL["positive"], linewidth=1.8, label="Positive")
    ax1.plot(x_labels, cum_neg, color=_COL["negative"], linewidth=1.8, label="Negative")
    ax1.plot(x_labels, cum_neu, color=_COL["neutral"],  linewidth=1.2, label="Neutral", linestyle="--")

    ax1.set_ylabel("Cumulative Weighted Score", fontsize=10, fontname=_FONT)
    ax1.set_title(
        f"Prediction Build-Up — How Evidence Accumulates ({predicted_label.upper()})",
        fontsize=11, fontname=_FONT, pad=10,
    )
    ax1.legend(fontsize=8, loc="upper left", prop={"family": _FONT, "size": 8})
    ax1.xaxis.grid(True, color=_COL["grid"], linewidth=0.5, linestyle="--")
    ax1.yaxis.grid(True, color=_COL["grid"], linewidth=0.5, linestyle="--")
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)

    # --- Bottom panel: normalised running proportions (stacked area) ---
    ax2.set_facecolor(_COL["bg"])
    ax2.stackplot(
        x_labels, norm_pos, norm_neg, norm_neu,
        colors=[_COL["positive"], _COL["negative"], _COL["neutral"]],
        alpha=0.7,
        labels=["Positive", "Negative", "Neutral"],
    )

    ax2.set_xlabel("Articles Added (sorted by weight)", fontsize=10, fontname=_FONT)
    ax2.set_ylabel("Running Proportion", fontsize=10, fontname=_FONT)
    ax2.set_title(
        "Normalised Sentiment Proportion as Articles Accumulate",
        fontsize=10, fontname=_FONT, pad=8,
    )
    ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8, loc="upper right", prop={"family": _FONT, "size": 8})
    ax2.xaxis.grid(True, color=_COL["grid"], linewidth=0.5, linestyle="--")
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    return _savefig(fig, out_dir / filename)


# ── Master function ───────────────────────────────────────────────────────────

def generate_all_charts(result: dict[str, Any], charts_dir: Path) -> dict[str, Path]:
    """
    Generate all 10 charts and return a dict mapping chart name → Path.
    If matplotlib is not installed, logs a warning and returns empty dict.
    """
    try:
        import matplotlib  # noqa: F401  (probe import)
    except ImportError:
        logger.warning(
            "matplotlib not installed — PNG charts skipped. "
            "Install with: pip install matplotlib"
        )
        return {}

    pred        = result["prediction_summary"]
    reliability = result["reliability"]
    layer1      = result["layer_1_token"]
    layer2      = result["layer_2_article"]
    layer3      = result["layer_3_pipeline"]
    storylines  = result.get("storylines", {})
    contrastive = layer2.get("contrastive", {})

    lime_articles   = layer1.get("articles", [])
    ranked_articles = layer2.get("ranked_articles", [])
    predicted_label = pred.get("final_label", "positive")

    paths: dict[str, Path] = {}
    try:
        paths["sentiment_scores"]     = plot_sentiment_scores(pred, charts_dir)
        paths["article_distribution"] = plot_article_distribution(layer2, charts_dir)
        paths["article_weights"]      = plot_article_weights(layer2, charts_dir)
        paths["horizon_breakdown"]    = plot_horizon_breakdown(layer3, charts_dir)
        if lime_articles:
            paths["lime_tokens"]      = plot_lime_tokens(lime_articles, predicted_label, charts_dir)
        paths["reliability"]          = plot_reliability(reliability, charts_dir)

        # ── New charts (7-10) ─────────────────────────────────
        if storylines.get("storylines"):
            paths["storyline_contribution"] = plot_storyline_contribution(
                storylines, predicted_label, charts_dir,
            )
        if contrastive.get("all_contributions"):
            paths["contrastive_waterfall"] = plot_contrastive_waterfall(
                contrastive, charts_dir,
            )
        if ranked_articles:
            paths["article_timeline"] = plot_article_timeline(
                ranked_articles, charts_dir,
            )
            paths["cumulative_score"] = plot_cumulative_score(
                ranked_articles, predicted_label, charts_dir,
            )
    except Exception as exc:
        logger.error("Chart generation failed: %s", exc, exc_info=True)

    logger.info("Generated %d chart(s) in %s", len(paths), charts_dir)
    return paths
