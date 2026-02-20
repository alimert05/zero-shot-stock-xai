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


# ── Master function ───────────────────────────────────────────────────────────

def generate_all_charts(result: dict[str, Any], charts_dir: Path) -> dict[str, Path]:
    """
    Generate all 6 charts and return a dict mapping chart name → Path.
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

    pred       = result["prediction_summary"]
    reliability = result["reliability"]
    layer1     = result["layer_1_token"]
    layer2     = result["layer_2_article"]
    layer3     = result["layer_3_pipeline"]

    lime_articles   = layer1.get("articles", [])
    predicted_label = pred.get("final_label", "positive")

    paths: dict[str, Path] = {}
    try:
        paths["sentiment_scores"]    = plot_sentiment_scores(pred, charts_dir)
        paths["article_distribution"] = plot_article_distribution(layer2, charts_dir)
        paths["article_weights"]     = plot_article_weights(layer2, charts_dir)
        paths["horizon_breakdown"]   = plot_horizon_breakdown(layer3, charts_dir)
        if lime_articles:
            paths["lime_tokens"]     = plot_lime_tokens(lime_articles, predicted_label, charts_dir)
        paths["reliability"]         = plot_reliability(reliability, charts_dir)
    except Exception as exc:
        logger.error("Chart generation failed: %s", exc, exc_info=True)

    logger.info("Generated %d chart(s) in %s", len(paths), charts_dir)
    return paths
