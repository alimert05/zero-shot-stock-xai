from __future__ import annotations

import json
import logging
from typing import Any

from predictor.abstention import apply_abstention
from config import OLLAMA_SENTIMENT_MODEL

logger = logging.getLogger(__name__)

# ── Prompt (FinGPT Instruction-Input-Answer format) ───────────────────────────
# Source: FinGPT framework (Yang et al., 2023)
# https://huggingface.co/FinGPT/fingpt-sentiment_llama2-13b_lora

OLLAMA_PROMPT_TEMPLATE = (
    "Instruction: What is the sentiment of this news? "
    "Please choose an answer from {{negative/neutral/positive}}\n"
    "Input: {text}\n"
    "Answer: "
)


# ── Text building (shared with finbert/fingpt) ───────────────────────────────

def _title_matches(title: str, company_name: str, ticker: str | None) -> bool:
    title_lower = title.lower()
    if company_name.lower() in title_lower:
        return True
    if ticker and ticker.lower() in title_lower:
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

    text = f"Sentiment for {company_name}: {body}"

    return text[:max_chars]


# ── Classification ────────────────────────────────────────────────────────────

def _classify_sentiment(text: str) -> dict[str, float]:
    """Classify a single text via Ollama. Returns hard scores {label: 0.90, others: 0.05}."""
    import ollama

    prompt = OLLAMA_PROMPT_TEMPLATE.format(text=text)

    response = ollama.chat(
        model=OLLAMA_SENTIMENT_MODEL,
        messages=[
            {"role": "user", "content": prompt},
        ],
        options={
            "temperature": 0.0,
            "num_predict": 10,
        },
    )

    answer = response["message"]["content"].strip().lower()

    if "positive" in answer:
        label = "positive"
    elif "negative" in answer:
        label = "negative"
    elif "neutral" in answer:
        label = "neutral"
    else:
        logger.warning(
            "Ollama (%s) unexpected answer: '%s', defaulting to neutral",
            OLLAMA_SENTIMENT_MODEL, answer,
        )
        label = "neutral"

    scores = {"positive": 0.05, "negative": 0.05, "neutral": 0.05}
    scores[label] = 0.90
    return scores


# ── Pipeline-level prediction ────────────────────────────────────────────────

def predict_sentiment(
    articles_json_path: str,
    company_name: str | None = None,
    ticker: str | None = None,
) -> dict[str, Any]:
    with open(articles_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    articles = data.get("articles", [])
    query = data.get("query", "")

    if not company_name:
        company_name = query
    if not company_name:
        raise ValueError("company_name must be provided or present in articles.json query field")

    logger.info(
        "Running Ollama (%s) sentiment on %d articles (company=%s, ticker=%s)",
        OLLAMA_SENTIMENT_MODEL, len(articles), company_name, ticker,
    )

    weighted_scores = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
    total_weight = 0.0
    article_sentiments: list[dict] = []
    article_weights: list[float] = []

    for i, article in enumerate(articles):
        title = article.get("title", "")
        final_weight = float(article.get("final_weight", 1.0))

        include_title = _title_matches(title, company_name, ticker)
        text = _build_input_text(article, include_title=include_title, company_name=company_name)

        if not text:
            logger.debug("Skipping article (no title and no content): %s", title[:80])
            continue

        scores = _classify_sentiment(text)

        for label in weighted_scores:
            weighted_scores[label] += scores[label] * final_weight
        total_weight += final_weight
        article_weights.append(final_weight)

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
    print(f"\n{'='*50}")
    print(f"  SENTIMENT PREDICTION RESULT ({OLLAMA_SENTIMENT_MODEL})")
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
    print(f"{'='*50}\n")
