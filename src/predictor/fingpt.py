from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

_fingpt_model = None
_fingpt_tokenizer = None

FINGPT_PROMPT = (
    "Instruction: What is the sentiment of this news? "
    "Please choose an answer from {{negative/neutral/positive}}\n"
    "Input: {text}\n"
    "Answer: "
)


def _get_fingpt_model():
    global _fingpt_model, _fingpt_tokenizer
    if _fingpt_model is None:
        try:
            import torch
            from transformers import LlamaForCausalLM, LlamaTokenizerFast
            from peft import PeftModel
            from config import (
                FINGPT_BASE_MODEL,
                FINGPT_LORA_MODEL,
                FINGPT_LOAD_IN_8BIT,
                SENTIMENT_DEVICE,
            )

            device = f"cuda:{SENTIMENT_DEVICE}" if SENTIMENT_DEVICE >= 0 else "cpu"

            logger.info(
                "Loading FinGPT: base=%s, lora=%s, 8bit=%s, device=%s",
                FINGPT_BASE_MODEL, FINGPT_LORA_MODEL, FINGPT_LOAD_IN_8BIT, device,
            )

            tokenizer = LlamaTokenizerFast.from_pretrained(
                FINGPT_BASE_MODEL, trust_remote_code=True,
            )
            tokenizer.pad_token = tokenizer.eos_token

            load_kwargs = {"trust_remote_code": True}
            if FINGPT_LOAD_IN_8BIT and SENTIMENT_DEVICE >= 0:
                load_kwargs["load_in_8bit"] = True
                load_kwargs["device_map"] = device
            else:
                load_kwargs["device_map"] = device
                load_kwargs["torch_dtype"] = torch.float16

            model = LlamaForCausalLM.from_pretrained(FINGPT_BASE_MODEL, **load_kwargs)
            model = PeftModel.from_pretrained(model, FINGPT_LORA_MODEL)
            model = model.eval()

            _fingpt_model = model
            _fingpt_tokenizer = tokenizer
            logger.info("FinGPT model loaded successfully")

        except Exception as exc:
            logger.error("Failed to load FinGPT model: %s", exc)
            raise

    return _fingpt_model, _fingpt_tokenizer


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


def _classify_sentiment(text: str) -> dict[str, float]:
    import torch

    model, tokenizer = _get_fingpt_model()

    prompt = FINGPT_PROMPT.format(text=text)
    tokens = tokenizer(
        prompt, return_tensors="pt", padding=True,
        truncation=True, max_length=512,
    )

    device = next(model.parameters()).device
    tokens = {k: v.to(device) for k, v in tokens.items()}

    with torch.no_grad():
        output = model.generate(
            **tokens,
            max_new_tokens=5,
            do_sample=False,
        )

    generated = tokenizer.decode(output[0], skip_special_tokens=True)

    answer = ""
    if "Answer: " in generated:
        answer = generated.split("Answer: ")[-1].strip().lower()

    if "positive" in answer:
        label = "positive"
    elif "negative" in answer:
        label = "negative"
    elif "neutral" in answer:
        label = "neutral"
    else:
        logger.warning("FinGPT unexpected answer: '%s', defaulting to neutral", answer)
        label = "neutral"

    scores = {"positive": 0.05, "negative": 0.05, "neutral": 0.05}
    scores[label] = 0.90
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

    if not company_name:
        company_name = query
    if not company_name:
        raise ValueError("company_name must be provided or present in articles.json query field")

    logger.info(
        "Running FinGPT sentiment on %d articles (company=%s, ticker=%s)",
        len(articles), company_name, ticker,
    )

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

        scores = _classify_sentiment(text)

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
    print(f"  SENTIMENT PREDICTION RESULT (FinGPT)")
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
    print(f"{'='*50}\n")
