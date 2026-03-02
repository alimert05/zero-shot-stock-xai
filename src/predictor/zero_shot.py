from __future__ import annotations

import json
import logging
from typing import Any
from transformers import pipeline
from config import SENTIMENT_DEVICE, MODEL_NAME

logger = logging.getLogger(__name__)

_deberta_pipeline = None

if MODEL_NAME == "facebook/bart-large-mnli":
    model = "BART Large MNLI"
if MODEL_NAME == "roberta-large-mnli":
    model = "RoBERTa Large MNLI"
if MODEL_NAME == "microsoft/deberta-large-mnli":
    model = "DeBERTa Large MNLI"

def _get_deberta_pipeline():
    global _deberta_pipeline
    if _deberta_pipeline is None:
        try:
            logger.info(f"Loading {model} zero-shot classifier...")
            _deberta_pipeline = pipeline(
                "zero-shot-classification",
                model= MODEL_NAME,
                device= SENTIMENT_DEVICE,
            )
            logger.info(f"{model} model loaded successfully")
        except Exception as exc:
            logger.error(f"Failed to load {model} model: {exc}")
            raise
    return _deberta_pipeline


_COMPANY_SUFFIXES = {"inc", "inc.", "corp", "corp.", "ltd", "ltd.", "co",
                     "co.", "plc", "llc", "group", "holdings", "sa", "ag",
                     "se", "nv", "the", "company"}


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
    core_words = [w for w in company_name.lower().split()
                  if w not in _COMPANY_SUFFIXES]
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
    content = article.get("content") or ""
    content = content.strip()

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


def _classify_sentiment(text: str, company_name: str) -> dict[str, float]:
    pipe = _get_deberta_pipeline()

    # Company-aware market-impact labels.
    # Keep an explicit class->label mapping so parsing remains stable
    # even when label wording changes.
    class_to_label = {
        "negative": f"negative news about {company_name}'s stock price",
        "neutral": f"neutral news about {company_name}'s stock price",
        "positive": f"positive news about {company_name}'s stock price",
    }
    candidate_labels = list(class_to_label.values())
    label_to_class = {v.lower().strip(): k for k, v in class_to_label.items()}


    result = pipe(
        text,
        candidate_labels=candidate_labels,
        hypothesis_template="This text is about {}.",
    )

    scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    for label, score in zip(result["labels"], result["scores"]):
        cls = label_to_class.get(label.lower().strip())
        if cls:
            scores[cls] = score

    return scores


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
        f"Running {model} sentiment on {len(articles)} articles (company={company_name}, ticker={ticker})")

    weighted_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    total_weight = 0.0
    article_sentiments: list[dict] = []

    for i, article in enumerate(articles):
        title = article.get("title", "")
        final_weight = article.get("final_weight", 1.0)

        include_title = _title_matches(title, company_name, ticker)
        text = _build_input_text(article, include_title=include_title, company_name=company_name)

        if not text:
            logger.debug("Skipping article (no title and no content): %s", title[:80])
            continue

        scores = _classify_sentiment(text, company_name)

        for label in weighted_scores:
            weighted_scores[label] += scores[label] * final_weight
        total_weight += final_weight

        content_raw = article.get("content") or ""
        if include_title:
            source_label = "headline+content"
        elif content_raw.strip():
            source_label = "content-only"
        else:
            source_label = "title-fallback"

        article_sentiments.append({
            "title": title,
            "final_weight": final_weight,
            "input_source": source_label,
            "raw_scores": scores,
            "weighted_scores": {
                k: round(v * final_weight, 4) for k, v in scores.items()
            },
        })

        logger.info(
            "[%d/%d] (%s) %s -> pos=%.4f neg=%.4f neu=%.4f (w=%.3f)",
            i + 1, len(articles), source_label, title[:50],
            scores["positive"], scores["negative"], scores["neutral"],
            final_weight,
        )

    if total_weight > 0:
        normalized_scores = {
            k: round(v / total_weight, 4) for k, v in weighted_scores.items()
        }
    else:
        normalized_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}

    final_label = max(normalized_scores, key=normalized_scores.get)

    ABSTENTION_MARGIN = 0.005

    sorted_labels = sorted(normalized_scores, key=normalized_scores.get, reverse=True)
    top_label = sorted_labels[0]
    runner_up = sorted_labels[1]

    margin = round(
        normalized_scores[top_label] - normalized_scores[runner_up], 4
    )
    abstention_method = None

    if margin < ABSTENTION_MARGIN:
        final_label = "neutral"
        abstention_method = "margin"
        logger.info(
            "Abstention: margin %.4f < %.4f → neutral (top=%s, runner_up=%s)",
            margin, ABSTENTION_MARGIN, top_label, runner_up,
        )
    else:
        logger.info(
            "Margin check passed: %.4f ≥ %.4f → keep %s",
            margin, ABSTENTION_MARGIN, top_label,
        )

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
        "abstention_test": {
            "method": abstention_method if abstention_method else "none",
            "margin": margin,
            "threshold": ABSTENTION_MARGIN,
            "n_articles": len(article_sentiments),
        },
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
    print(f"\n{'='*50}")
    print(f"  SENTIMENT PREDICTION RESULT ({model})")
    print(f"{'='*50}")
    print(f"  Company : {result['company_name']}")
    if result.get("ticker"):
        print(f"  Ticker  : {result['ticker']}")
    print(f"  Articles: {result['articles_analyzed']}/{result['articles_total']} matched")
    print(f"{'-'*50}")
    print(f"  Normalized Scores (weighted by article importance):")
    for label in ["positive", "negative", "neutral"]:
        score = result["normalized_scores"][label]
        bar = "#" * int(score * 30)
        print(f"    {label:>8}: {score:.4f}  {bar}")
    print(f"{'-'*50}")
    print(f"  FINAL LABEL : {result['final_label'].upper()}")
    print(f"  CONFIDENCE  : {result['final_confidence']:.4f}")
    # Abstention test summary
    abst = result.get("abstention_test", {})
    method = abst.get("method", "none")
    margin = abst.get("margin", 0)
    threshold = abst.get("threshold", 0)
    if method != "none":
        print(f"  ABSTAINED   : Yes (margin {margin:.4f} < {threshold})")
    else:
        print(f"  MARGIN      : {margin:.4f} (threshold={threshold})")
    print(f"{'='*50}\n")