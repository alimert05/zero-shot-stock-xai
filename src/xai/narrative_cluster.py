"""Narrative storyline clustering for XAI reports.

Groups articles into storylines by TF-IDF similarity on titles,
then optionally labels each cluster with a one-phrase Ollama summary.

Academic context:
  - TF-IDF weighting: Salton & Buckley (1988)
  - Agglomerative clustering: Ward (1963)
  - Cosine similarity for text: Manning et al. (2008)
  - Narrative explanations in XAI: Biran & Cotton (2017)
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config import XAI_LLAMA_MODEL, XAI_LLAMA_ENABLED

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────
MAX_CLUSTERS = 7            # cap for readability
MIN_CLUSTER_SIZE = 2        # singletons go to "Other topics"
DISTANCE_THRESHOLD = 0.85   # cosine distance cut (0 = identical, ~1 = unrelated)
MIN_CLUSTERED_RATIO = 0.25  # adaptive: relax threshold if < 25 % of articles cluster
THRESHOLD_RELAX_STEP = 0.10 # how much to relax per retry
MAX_THRESHOLD_RETRIES = 2   # max adaptive relaxation attempts


def _cluster_titles(
    titles: list[str],
) -> list[int]:
    """Agglomerative clustering on TF-IDF vectors of article titles.

    Uses cosine distance with average linkage — a natural metric for
    short-text similarity (Manning et al., 2008).  An adaptive loop
    relaxes the distance threshold when too few articles cluster (< 25 %).

    Returns a list of cluster labels (0-indexed). Singletons are
    relabelled to -1 so they can be grouped into "Other topics".
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import AgglomerativeClustering
    from collections import Counter

    if len(titles) < 3:
        return [0] * len(titles)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=(1, 2),   # unigrams + bigrams for richer similarity
        min_df=1,
        sublinear_tf=True,
    )
    tfidf_matrix = vectorizer.fit_transform(titles)
    dense = tfidf_matrix.toarray()

    # ── Adaptive threshold loop ──────────────────────────────────
    threshold = DISTANCE_THRESHOLD
    best_labels: np.ndarray | None = None

    for attempt in range(1 + MAX_THRESHOLD_RETRIES):
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=threshold,
            metric="cosine",
            linkage="average",
        )
        labels = clustering.fit_predict(dense)

        # Cap at MAX_CLUSTERS: merge smallest into "other"
        counts = Counter(labels)
        if len(counts) > MAX_CLUSTERS:
            top_clusters = {c for c, _ in counts.most_common(MAX_CLUSTERS)}
            labels = np.array([
                lbl if lbl in top_clusters else -1 for lbl in labels
            ])

        # Relabel singletons to -1
        counts = Counter(labels)
        labels = np.array([
            lbl if counts[lbl] >= MIN_CLUSTER_SIZE else -1
            for lbl in labels
        ])

        clustered = len(labels) - int(np.sum(labels == -1))
        clustered_ratio = clustered / len(labels)
        best_labels = labels

        if clustered_ratio >= MIN_CLUSTERED_RATIO:
            logger.debug(
                "Clustering attempt %d (threshold=%.2f): %.1f%% clustered — accepted.",
                attempt + 1, threshold, clustered_ratio * 100,
            )
            break

        logger.debug(
            "Clustering attempt %d (threshold=%.2f): %.1f%% clustered — relaxing.",
            attempt + 1, threshold, clustered_ratio * 100,
        )
        threshold += THRESHOLD_RELAX_STEP

    return best_labels.tolist()


def _extract_cluster_keywords(
    titles: list[str],
    labels: list[int],
) -> dict[int, list[str]]:
    """Top 3 TF-IDF terms (unigrams or bigrams) per cluster for fallback labelling."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    cluster_ids = sorted(set(labels) - {-1})
    keywords: dict[int, list[str]] = {}

    for cid in cluster_ids:
        cluster_titles = [t for t, l in zip(titles, labels) if l == cid]
        if not cluster_titles:
            continue

        vec = TfidfVectorizer(
            stop_words="english",
            max_features=200,
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
        )
        tfidf = vec.fit_transform(cluster_titles)
        feature_names = vec.get_feature_names_out()

        # Mean TF-IDF across cluster articles → top keywords
        mean_scores = np.asarray(tfidf.mean(axis=0)).flatten()
        top_indices = mean_scores.argsort()[::-1][:3]
        keywords[cid] = [feature_names[i] for i in top_indices]

    return keywords


def _ollama_label_clusters(
    cluster_titles: dict[int | str, list[str]],
) -> dict[int | str, str]:
    """Ask Ollama to produce a one-phrase label for each cluster."""
    if not XAI_LLAMA_ENABLED:
        return {}

    try:
        import ollama
    except ImportError:
        logger.debug("ollama not installed; skipping cluster labelling.")
        return {}

    labels: dict[int, str] = {}

    for cid, titles in cluster_titles.items():
        # Feed up to 8 titles to keep prompt short
        title_block = "\n".join(f"- {t}" for t in titles[:8])
        prompt = (
            "Below are headlines from a set of related news articles about "
            "the same company. Write ONE short phrase (max 8 words) that "
            "summarises the common storyline. Do NOT mention the company "
            "name. Respond with ONLY the phrase, nothing else.\n\n"
            f"{title_block}"
        )

        try:
            response = ollama.chat(
                model=XAI_LLAMA_MODEL,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": 0.1,
                    "num_predict": 30,
                },
            )
            label = response["message"]["content"].strip().strip('"').strip("'")
            # Truncate if LLM ignores the length constraint
            if len(label.split()) > 12:
                label = " ".join(label.split()[:8]) + "…"
            labels[cid] = label
            logger.debug("Cluster %d label: %s", cid, label)
        except Exception as exc:
            logger.warning("Ollama cluster label failed for cluster %d: %s", cid, exc)
            continue

    return labels


def _build_group_storylines(
    articles: list[dict[str, Any]],
    sent_group: str,
) -> tuple[list[dict[str, Any]], int, dict[str, list[str]]]:
    """Cluster one sentiment group and return (storylines, other_count, cluster_titles).

    The returned cluster_titles dict is keyed by "{sent_group}_{cid}"
    so multiple groups can be merged for a single Ollama labelling pass.
    """
    from .utils import get_dominant_label

    titles = [a.get("title", "") for a in articles]

    if len(titles) < 3:
        return [], len(titles), {}

    labels = _cluster_titles(titles)
    cluster_ids = sorted(set(labels) - {-1})
    group_other = labels.count(-1)

    keywords = _extract_cluster_keywords(titles, labels)

    # Collect cluster titles for deferred Ollama labelling
    cluster_title_map: dict[str, list[str]] = {}
    for cid in cluster_ids:
        key = f"{sent_group}_{cid}"
        cluster_title_map[key] = [t for t, l in zip(titles, labels) if l == cid]

    # Determine runner-up sentiment for gap contribution scoring
    _runner_up = {"positive": "negative", "negative": "positive", "neutral": "positive"}
    opposing = _runner_up.get(sent_group, "positive")

    storylines: list[dict[str, Any]] = []
    for cid in cluster_ids:
        cluster_articles = [a for a, l in zip(articles, labels) if l == cid]
        n = len(cluster_articles)

        weighted_score = sum(a.get("final_weight", 0.0) for a in cluster_articles)

        # Contribution score = sum(final_weight × (group_score − opposing_score))
        # Measures how much this cluster actually pushed toward the group sentiment
        contribution = 0.0
        for a in cluster_articles:
            w = a.get("final_weight", 0.0)
            raw = a.get("raw_scores", {})
            group_score = raw.get(sent_group, 0.0)
            opp_score = raw.get(opposing, 0.0)
            contribution += w * (group_score - opp_score)

        # Per-article sentiment breakdown (still informative within a group)
        sent_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for a in cluster_articles:
            raw = a.get("raw_scores", {})
            dom = get_dominant_label(raw) if raw else "neutral"
            sent_counts[dom] = sent_counts.get(dom, 0) + 1

        kw = keywords.get(cid, [])
        keyword_label = ", ".join(kw) if kw else f"cluster {cid}"

        # All titles in cluster, sorted by contribution (highest impact first)
        cluster_articles_sorted = sorted(
            cluster_articles,
            key=lambda a: a.get("final_weight", 0.0) * (
                a.get("raw_scores", {}).get(sent_group, 0.0)
                - a.get("raw_scores", {}).get(opposing, 0.0)
            ),
            reverse=True,
        )
        top_titles = [a.get("title", "") for a in cluster_articles_sorted]

        storylines.append({
            "label": "",                        # placeholder — filled after Ollama
            "keyword_label": keyword_label,
            "articles_count": n,
            "weighted_score": round(weighted_score, 4),
            "contribution_score": round(contribution, 4),
            "sentiment_group": sent_group,       # article-level sentiment group
            "sentiment": {
                "dominant": sent_group,
                "positive": sent_counts["positive"],
                "negative": sent_counts["negative"],
                "neutral": sent_counts["neutral"],
            },
            "top_titles": top_titles,
            "_cluster_key": f"{sent_group}_{cid}",
        })

    # Sort by contribution score — actual impact on prediction, not just timing
    storylines.sort(key=lambda s: s["contribution_score"], reverse=True)
    return storylines, group_other, cluster_title_map


def cluster_narratives(
    merged_articles: list[dict[str, Any]],
) -> dict[str, Any]:
    from .utils import get_dominant_label

    if len(merged_articles) < 3:
        logger.info("Too few articles (%d) for narrative clustering.", len(merged_articles))
        return {"storylines": [], "other_count": len(merged_articles), "method": "skipped"}

    # ── Step 1: split articles by article-level dominant sentiment ──
    by_sentiment: dict[str, list[dict[str, Any]]] = {
        "positive": [], "negative": [], "neutral": [],
    }
    for a in merged_articles:
        raw = a.get("raw_scores", {})
        dom = get_dominant_label(raw) if raw else "neutral"
        by_sentiment[dom].append(a)

    logger.info(
        "Sentiment split: %d positive, %d negative, %d neutral.",
        len(by_sentiment["positive"]),
        len(by_sentiment["negative"]),
        len(by_sentiment["neutral"]),
    )

    # ── Step 2: cluster within each sentiment group ────────────────
    all_storylines: list[dict[str, Any]] = []
    total_other = 0
    all_cluster_titles: dict[str, list[str]] = {}

    for sent_group, articles in by_sentiment.items():
        sl, other, ct = _build_group_storylines(articles, sent_group)
        all_storylines.extend(sl)
        total_other += other
        all_cluster_titles.update(ct)

    # ── Step 3: one Ollama labelling pass for all clusters ─────────
    ollama_labels = _ollama_label_clusters(all_cluster_titles)

    for sl in all_storylines:
        key = sl.pop("_cluster_key")
        ollama_label = ollama_labels.get(key, "")
        sl["label"] = ollama_label if ollama_label else sl["keyword_label"].title()

    # ── Step 4: final sort by contribution score ────────────────────
    all_storylines.sort(key=lambda s: s["contribution_score"], reverse=True)

    logger.info(
        "Narrative clustering (by sentiment): %d storylines + %d other from %d articles.",
        len(all_storylines), total_other, len(merged_articles),
    )

    return {
        "storylines": all_storylines,
        "other_count": total_other,
        "method": "tfidf_agglomerative_by_sentiment",
    }
