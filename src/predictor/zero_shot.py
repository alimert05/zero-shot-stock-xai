from __future__ import annotations

import json
import logging
import math
import re
from typing import Any

from transformers import pipeline

from config import (
    SENTIMENT_DEVICE,
    MODEL_NAME,
    PRIOR_DEBIASING_ENABLED,
    PRIOR_DEBIASING_ALPHA,
    SENTIMENT_CONFIDENCE_WEIGHTING,
    COVERAGE_COUNT_BOOST,
    RELEVANCE_RATIO_WEIGHTING,
    HEADLINE_ONLY_WEIGHT,
)
from predictor.abstention import apply_abstention

logger = logging.getLogger(__name__)

_deberta_pipeline = None

MODEL_DISPLAY_NAMES = {
    "facebook/bart-large-mnli": "BART Large MNLI",
    "roberta-large-mnli": "RoBERTa Large MNLI",
    "microsoft/deberta-large-mnli": "DeBERTa Large MNLI",
}
model = MODEL_DISPLAY_NAMES.get(MODEL_NAME, MODEL_NAME)


def _get_deberta_pipeline():
    global _deberta_pipeline
    if _deberta_pipeline is None:
        try:
            logger.info("Loading %s zero-shot classifier...", model)
            _deberta_pipeline = pipeline(
                "zero-shot-classification",
                model=MODEL_NAME,
                device=SENTIMENT_DEVICE,
            )
            logger.info("%s model loaded successfully", model)
        except Exception as exc:
            logger.error("Failed to load %s model: %s", model, exc)
            raise
    return _deberta_pipeline


_COMPANY_SUFFIXES = {
    "inc", "inc.", "corp", "corp.", "ltd", "ltd.", "co", "co.",
    "plc", "llc", "group", "holdings", "sa", "ag", "se", "nv",
    "the", "company",
}


def _title_matches(title: str, company_name: str, ticker: str | None) -> bool:
    title_lower = title.lower()

    # Full company name match (e.g. "Apple Inc." in title)
    if company_name.lower() in title_lower:
        return True

    # Ticker match — case-sensitive with word boundaries to avoid
    # short tickers (V, A, T) matching inside random words
    if ticker and re.search(rf"\b{re.escape(ticker)}\b", title):
        return True

    # Core-name match: strip common suffixes like Inc., Corp., Ltd.
    # so "Apple Inc." matches a title containing just "Apple"
    core_words = [w for w in company_name.lower().split() if w not in _COMPANY_SUFFIXES]
    if core_words:
        core_name = " ".join(core_words)
        if core_name in title_lower:
            return True

    return False


def _build_input_text(
    article: dict,
    include_title: bool,
    company_name: str,
    max_chars: int = 1500,
) -> str:
    title = article.get("title", "").strip()
    content = (article.get("content") or "").strip()

    if include_title:
        body = f"{title}. {content}" if content else title
    elif content:
        body = content
    else:
        body = title

    if not body:
        return ""

    text = f"News about {company_name}: {body}"
    return text[:max_chars]


_CLASS_TO_LABEL = {
    "positive": "positive financial outlook",
    "negative": "negative financial outlook",
    "neutral": "neutral financial outlook",
}
_CANDIDATE_LABELS = list(_CLASS_TO_LABEL.values())
_LABEL_TO_CLASS = {v.lower().strip(): k for k, v in _CLASS_TO_LABEL.items()}
_HYPOTHESIS_TEMPLATE = "This text is {} about the financial outlook."


def _classify_sentiment(text: str, company_name: str) -> dict[str, float]:
    pipe = _get_deberta_pipeline()

    result = pipe(
        text,
        candidate_labels=_CANDIDATE_LABELS,
        hypothesis_template=_HYPOTHESIS_TEMPLATE,
        multi_label=True,
    )

    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for label, score in zip(result["labels"], result["scores"]):
        cls = _LABEL_TO_CLASS.get(label.lower().strip())
        if cls:
            scores[cls] = float(score)

    return scores


def _batch_classify_sentiment(
    texts: list[str],
    batch_size: int = 32,
) -> list[dict[str, float]]:
    """Classify multiple texts in a single batched GPU call.

    Instead of N individual forward passes (one per article), this creates
    N × 3 NLI pairs and processes them in chunks of *batch_size*, dramatically
    reducing GPU kernel-launch overhead and Python-loop latency.
    """
    pipe = _get_deberta_pipeline()

    results = pipe(
        texts,
        candidate_labels=_CANDIDATE_LABELS,
        hypothesis_template=_HYPOTHESIS_TEMPLATE,
        multi_label=True,
        batch_size=batch_size,
    )

    # Single text → pipeline returns a dict instead of list
    if isinstance(results, dict):
        results = [results]

    all_scores: list[dict[str, float]] = []
    for result in results:
        scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for label, score in zip(result["labels"], result["scores"]):
            cls = _LABEL_TO_CLASS.get(label.lower().strip())
            if cls:
                scores[cls] = float(score)
        all_scores.append(scores)

    return all_scores


# ── Prior debiasing (Bayesian label-bias correction) ────────────────────

# Content-free financial sentences used to measure the model's inherent
# label preference.  These carry no directional sentiment — the model's
# output on them reveals its structural bias toward certain labels.
_PRIOR_PROBE_TEXTS = [
    "News about the company: A financial statement was released.",
    "News about the company: Quarterly earnings results were reported.",
    "News about the company: The stock market was active during the trading session.",
    "News about the company: Trading volume was recorded for the reporting period.",
    "News about the company: An analyst published a report covering the sector.",
    "News about the company: The annual shareholder meeting was held as scheduled.",
    "News about the company: Market participants observed the latest trading session.",
    "News about the company: The financial quarter has concluded for the fiscal year.",
]

_estimated_prior: dict[str, float] | None = None


def estimate_prior(batch_size: int = 32) -> dict[str, float]:
    """Estimate the model's inherent label bias using content-free probe texts.

    Feeds generic, sentiment-neutral financial sentences through the same
    zero-shot classifier (same labels, same hypothesis template) and averages
    the outputs to obtain the prior distribution P(label).

    The prior is computed once and cached for the duration of the session.

    Returns
    -------
    dict mapping class name to prior probability, e.g.
    {"positive": 0.50, "negative": 0.20, "neutral": 0.30}
    """
    global _estimated_prior
    if _estimated_prior is not None:
        return _estimated_prior

    logger.info("Estimating model prior from %d probe texts ...", len(_PRIOR_PROBE_TEXTS))
    all_scores = _batch_classify_sentiment(_PRIOR_PROBE_TEXTS, batch_size=batch_size)

    prior = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for scores in all_scores:
        for label in prior:
            prior[label] += scores[label]

    n = len(all_scores)
    prior = {label: val / n for label, val in prior.items()}

    _estimated_prior = prior
    logger.info(
        "Estimated prior: pos=%.4f neg=%.4f neu=%.4f",
        prior["positive"], prior["negative"], prior["neutral"],
    )
    return prior


def _debias_scores(
    raw_scores: dict[str, float],
    prior: dict[str, float],
    alpha: float = PRIOR_DEBIASING_ALPHA,
) -> dict[str, float]:
    """Apply dampened Bayesian prior correction to raw classification scores.

    For each label:  adjusted[label] = raw[label] / prior[label]^α
    Then renormalise so the adjusted scores sum to 1.

    α controls the correction strength:
        α = 1.0  → full debiasing (original Zhao et al. 2021)
        α = 0.5  → half-strength (gentler correction)
        α = 0.0  → no debiasing (raw scores unchanged)

    This removes the model's inherent bias: a label with a high prior
    (e.g. positive) is penalised, while a label with a low prior
    (e.g. negative) is boosted proportionally.  The damping factor α
    prevents over-correction when the prior estimate is noisy.
    """
    adjusted = {}
    for label in raw_scores:
        p = prior.get(label, 1.0)
        adjusted[label] = raw_scores[label] / max(p ** alpha, 1e-8)

    total = sum(adjusted.values())
    if total > 0:
        adjusted = {label: val / total for label, val in adjusted.items()}

    return adjusted


def predict_sentiment(
    articles_json_path: str,
    company_name: str | None = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    with open(articles_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    query = data.get("query", "")
    json_ticker = data.get("ticker")

    if not company_name:
        company_name = query
    if not ticker:
        ticker = json_ticker
    if not company_name:
        raise ValueError("company_name must be provided or present in articles.json query field")

    logger.info(
        "Running %s sentiment on %d articles (company=%s, ticker=%s)",
        model, len(articles), company_name, ticker
    )

    weighted_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    total_weight = 0.0
    article_sentiments: list[dict] = []
    article_weights: list[float] = []

    # ── Phase 1: Collect texts and metadata (CPU only) ──
    batch_texts: list[str] = []
    batch_meta: list[dict] = []  # parallel list: article index, weight, source_label

    for i, article in enumerate(articles):
        title = article.get("title", "")
        final_weight = float(article.get("final_weight", 1.0))

        include_title = _title_matches(title, company_name, ticker)
        text = _build_input_text(article, include_title=include_title, company_name=company_name)

        if not text:
            logger.debug("Skipping article (no title and no content): %s", title[:80])
            continue

        content_raw = article.get("content") or ""
        if include_title:
            source_label = "headline+content"
        elif content_raw.strip():
            source_label = "content-only"
        else:
            source_label = "title-fallback"

        coverage_count = int(article.get("coverage_count", 1))
        content_stats = article.get("content_stats", {})
        relevance_ratio = float(content_stats.get("relevance_ratio", 1.0))
        is_headline_only = not content_raw.strip()

        batch_texts.append(text)
        batch_meta.append({
            "idx": i,
            "title": title,
            "final_weight": final_weight,
            "source_label": source_label,
            "coverage_count": coverage_count,
            "relevance_ratio": relevance_ratio,
            "is_headline_only": is_headline_only,
        })

    # ── Phase 2: Batch GPU inference (single call for all articles) ──
    if batch_texts:
        all_scores = _batch_classify_sentiment(batch_texts, batch_size=32)
    else:
        all_scores = []

    # ── Phase 2.5: Prior debiasing (Bayesian label-bias correction) ──
    prior = None
    if PRIOR_DEBIASING_ENABLED and all_scores:
        prior = estimate_prior()
        all_debiased = [_debias_scores(s, prior) for s in all_scores]
    else:
        all_debiased = all_scores

    # ── Phase 3: Accumulate weighted scores ──
    for meta, raw, debiased in zip(batch_meta, all_scores, all_debiased):
        base_weight = meta["final_weight"]
        effective_weight = base_weight

        # ── Coverage count boost: log2(1 + coverage) ──
        coverage_boost = 1.0
        if COVERAGE_COUNT_BOOST:
            cc = meta["coverage_count"]
            coverage_boost = math.log2(1 + cc)          # cc=1→1.0, cc=3→2.0, cc=5→2.58
            effective_weight *= coverage_boost

        # ── Relevance ratio weighting: content quality from noise reducer ──
        relevance_ratio = 1.0
        if RELEVANCE_RATIO_WEIGHTING:
            relevance_ratio = meta["relevance_ratio"]       # in [0, 1]
            effective_weight *= relevance_ratio

        # ── Headline-only discount: less info = less trust ──
        headline_discount = 1.0
        if meta["is_headline_only"]:
            headline_discount = HEADLINE_ONLY_WEIGHT
            effective_weight *= headline_discount

        # ── Sentiment confidence weighting: margin between top-1 and top-2 ──
        sentiment_margin = 1.0
        if SENTIMENT_CONFIDENCE_WEIGHTING:
            sorted_scores = sorted(debiased.values(), reverse=True)
            sentiment_margin = sorted_scores[0] - sorted_scores[1]  # ∈ [0, 1]
            effective_weight *= sentiment_margin

        for label in weighted_scores:
            weighted_scores[label] += debiased[label] * effective_weight
        total_weight += effective_weight
        article_weights.append(effective_weight)

        detail = {
            "title": meta["title"],
            "base_weight": base_weight,
            "effective_weight": round(effective_weight, 4),
            "coverage_boost": round(coverage_boost, 4),
            "relevance_ratio": round(relevance_ratio, 4),
            "headline_discount": round(headline_discount, 4),
            "sentiment_margin": round(sentiment_margin, 4),
            "input_source": meta["source_label"],
            "raw_scores": raw,
            "weighted_scores": {
                k: round(v * effective_weight, 4) for k, v in debiased.items()
            },
        }
        if prior is not None:
            detail["debiased_scores"] = {k: round(v, 4) for k, v in debiased.items()}
        article_sentiments.append(detail)

        logger.info(
            "[%d/%d] (%s) %s -> pos=%.4f neg=%.4f neu=%.4f (w=%.3f->%.3f, cov=%.2f, rel=%.2f, hdl=%.2f, mar=%.2f)%s",
            meta["idx"] + 1, len(articles), meta["source_label"],
            meta["title"][:50],
            debiased["positive"], debiased["negative"], debiased["neutral"],
            base_weight, effective_weight, coverage_boost, relevance_ratio, headline_discount, sentiment_margin,
            " [debiased]" if prior else "",
        )

    if total_weight > 0:
        normalized_scores = {
            k: round(v / total_weight, 4) for k, v in weighted_scores.items()
        }
    else:
        normalized_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    abstention = apply_abstention(normalized_scores, article_weights)
    final_label = abstention["final_label"]

    result = {
        "query": query,
        "company_name": company_name,
        "ticker": ticker,
        "articles_analyzed": len(article_sentiments),
        "articles_total": len(articles),
        "total_weight": round(total_weight, 4),
        "prior_debiasing": {
            "enabled": prior is not None,
            "alpha": PRIOR_DEBIASING_ALPHA if prior is not None else None,
            "prior": {k: round(v, 4) for k, v in prior.items()} if prior else None,
        },
        "enhanced_weighting": {
            "sentiment_confidence": SENTIMENT_CONFIDENCE_WEIGHTING,
            "coverage_boost": COVERAGE_COUNT_BOOST,
            "relevance_ratio": RELEVANCE_RATIO_WEIGHTING,
            "headline_only_weight": HEADLINE_ONLY_WEIGHT,
        },
        "weighted_scores": {
            k: round(v, 4) for k, v in weighted_scores.items()
        },
        "normalized_scores": normalized_scores,
        "final_label": final_label,
        "final_confidence": normalized_scores[final_label],
        "article_details": article_sentiments,
        "abstention_test": abstention["abstention_test"],
    }

    logger.info(
        "Sentiment prediction complete: label=%s confidence=%.4f (%d articles analyzed)",
        final_label, normalized_scores[final_label], len(article_sentiments),
    )

    return result


def run_sentiment_prediction(
    articles_json_path: str,
    output_path: str | None = None,
    company_name: str | None = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    result = predict_sentiment(
        articles_json_path=articles_json_path,
        company_name=company_name,
        ticker=ticker,
    )

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info("Sentiment result saved to %s", output_path)

    _print_summary(result)
    return result


def _print_summary(result: dict) -> None:
    print(f"\n{'=' * 50}")
    print(f"  SENTIMENT PREDICTION RESULT ({model})")
    print(f"{'=' * 50}")
    print(f"  Company : {result['company_name']}")
    if result.get("ticker"):
        print(f"  Ticker  : {result['ticker']}")
    print(f"  Articles: {result['articles_analyzed']}/{result['articles_total']} matched")
    print(f"{'-' * 50}")
    print("  Normalized Scores (weighted by article importance):")
    for label in ["positive", "negative", "neutral"]:
        score = result["normalized_scores"][label]
        bar = "#" * int(score * 30)
        print(f"    {label:>8}: {score:.4f}  {bar}")
    print(f"{'-' * 50}")
    print(f"  FINAL LABEL : {result['final_label'].upper()}")
    print(f"  CONFIDENCE  : {result['final_confidence']:.4f}")

    abst = result.get("abstention_test", {})
    method = abst.get("method", "none")
    margin = abst.get("margin", 0.0)
    threshold = abst.get("threshold", 0.0)
    entropy = abst.get("entropy", 0.0)
    effective_n = abst.get("effective_n", 0.0)

    if method != "none":
        print(f"  ABSTAINED   : Yes ({method})")
        print(f"  MARGIN      : {margin:.4f}")
        print(f"  THRESHOLD   : {threshold:.4f}")
    else:
        print(f"  MARGIN      : {margin:.4f} (threshold={threshold:.4f})")

    print(f"  ENTROPY     : {entropy:.4f}")
    print(f"  EFFECTIVE N : {effective_n:.4f}")
    print(f"{'=' * 50}\n")