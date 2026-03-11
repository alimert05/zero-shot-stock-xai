from __future__ import annotations

import json
import logging
from typing import Any

from transformers import pipeline

from config import SENTIMENT_DEVICE, MODEL_NAME
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

    # Ticker match (e.g. "AAPL" in title)
    if ticker and ticker.lower() in title_lower:
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
        multi_label=False,
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
        multi_label=False,
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

        batch_texts.append(text)
        batch_meta.append({
            "idx": i,
            "title": title,
            "final_weight": final_weight,
            "source_label": source_label,
        })

    # ── Phase 2: Batch GPU inference (single call for all articles) ──
    if batch_texts:
        all_scores = _batch_classify_sentiment(batch_texts, batch_size=32)
    else:
        all_scores = []

    # ── Phase 3: Accumulate weighted scores ──
    for meta, scores in zip(batch_meta, all_scores):
        final_weight = meta["final_weight"]

        for label in weighted_scores:
            weighted_scores[label] += scores[label] * final_weight
        total_weight += final_weight
        article_weights.append(final_weight)

        article_sentiments.append({
            "title": meta["title"],
            "final_weight": final_weight,
            "input_source": meta["source_label"],
            "raw_scores": scores,
            "weighted_scores": {
                k: round(v * final_weight, 4) for k, v in scores.items()
            },
        })

        logger.info(
            "[%d/%d] (%s) %s -> pos=%.4f neg=%.4f neu=%.4f (w=%.3f)",
            meta["idx"] + 1, len(articles), meta["source_label"],
            meta["title"][:50],
            scores["positive"], scores["negative"], scores["neutral"],
            final_weight,
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