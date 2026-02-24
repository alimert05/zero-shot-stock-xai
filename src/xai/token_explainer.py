from __future__ import annotations

import logging
from typing import Any

import numpy as np

from config import XAI_LIME_TOP_N, XAI_LIME_NUM_SAMPLES, XAI_LIME_NUM_FEATURES
from .utils import label_index, get_dominant_label, build_lime_noise_set, is_lime_noise_token

logger = logging.getLogger(__name__)

_lime_explainer = None


def _get_lime_explainer():
    global _lime_explainer
    if _lime_explainer is None:
        from lime.lime_text import LimeTextExplainer
        _lime_explainer = LimeTextExplainer(
            class_names=["positive", "negative", "neutral"],
            kernel_width=25,
            random_state=42,
        )
        logger.info("LIME explainer initialised.")
    return _lime_explainer


def _build_predict_fn(pipeline_callable, company_name: str):
    # Labels must match exactly what zero_shot._classify_sentiment uses on feature/xai
    labels = [
        f"negative sentiment toward {company_name}",
        f"this article is factual or neutral about {company_name}",
        f"positive sentiment toward {company_name}",
    ]

    def predict_proba(texts: list[str]) -> np.ndarray:
        results = pipeline_callable(
            texts,
            candidate_labels=labels,
            hypothesis_template="This text expresses {}.",
            batch_size=4,
        )
        # pipeline returns dict when given single string, list when given list
        if isinstance(results, dict):
            results = [results]

        probs = np.zeros((len(results), 3))
        for i, result in enumerate(results):
            for lbl, score in zip(result["labels"], result["scores"]):
                if "positive" in lbl:
                    probs[i, 0] = score
                elif "negative" in lbl:
                    probs[i, 1] = score
                elif "factual or neutral" in lbl:
                    probs[i, 2] = score
        return probs

    return predict_proba


def _select_top_articles(
    merged_articles: list[dict[str, Any]],
    top_n: int,
    predicted_label: str = "positive",
) -> list[dict[str, Any]]:
    """
    Select top_n articles for LIME, prioritising articles whose dominant
    sentiment matches the predicted label.

    Strategy:
      - Score = final_weight * raw_score_for_predicted_label
        (not max score — we care about the predicted label's confidence)
      - Take as many label-aligned articles as possible (up to top_n)
      - Fill remaining slots from opposing articles so we also explain
        what the model had to overcome, keeping rank order within each group
    """
    def aligned_score(a: dict) -> float:
        final_weight = a.get("final_weight", 0.0) or 0.0
        raw_scores = a.get("raw_scores", {})
        return final_weight * raw_scores.get(predicted_label, 0.0)

    def opposing_score(a: dict) -> float:
        final_weight = a.get("final_weight", 0.0) or 0.0
        raw_scores = a.get("raw_scores", {})
        dominant_score = max(raw_scores.values()) if raw_scores else 0.0
        return final_weight * dominant_score

    aligned   = [a for a in merged_articles if get_dominant_label(a.get("raw_scores", {})) == predicted_label]
    opposing  = [a for a in merged_articles if get_dominant_label(a.get("raw_scores", {})) != predicted_label]

    aligned.sort(key=aligned_score, reverse=True)
    opposing.sort(key=opposing_score, reverse=True)

    # Fill: take min(top_n, all aligned), then pad with top opposing
    n_aligned  = min(len(aligned), top_n)
    n_opposing = min(len(opposing), top_n - n_aligned)

    selected = aligned[:n_aligned] + opposing[:n_opposing]

    logger.debug(
        "LIME article selection: %d aligned + %d opposing (predicted=%s)",
        n_aligned, n_opposing, predicted_label,
    )
    return selected


def explain_tokens(
    merged_articles: list[dict[str, Any]],
    company_name: str,
    predicted_label: str,
    ticker: str = "",
    top_n: int = XAI_LIME_TOP_N,
    num_samples: int = XAI_LIME_NUM_SAMPLES,
    num_features: int = XAI_LIME_NUM_FEATURES,
) -> list[dict[str, Any]]:
    try:
        from predictor.zero_shot import _get_deberta_pipeline
        pipeline_callable = _get_deberta_pipeline()
    except Exception as exc:
        logger.error("Could not load zero-shot pipeline for LIME: %s", exc)
        return []

    explainer = _get_lime_explainer()
    predict_fn = _build_predict_fn(pipeline_callable, company_name)
    target_label_idx = label_index(predicted_label)
    all_titles = [a.get("title", "") for a in merged_articles]
    noise = build_lime_noise_set(company_name, ticker, article_titles=all_titles)

    top_articles = _select_top_articles(merged_articles, top_n, predicted_label)
    results = []

    for rank, article in enumerate(top_articles, start=1):
        title = article.get("title", "")
        content = article.get("content") or ""
        final_weight = article.get("final_weight", 0.0)
        raw_scores = article.get("raw_scores", {})

        # Reconstruct input text exactly as zero_shot._build_input_text does
        body = f"{title}. {content}" if content.strip() else title
        text_to_explain = f"News about {company_name}: {body}"
        text_to_explain = text_to_explain[:1500]

        if not text_to_explain.strip():
            logger.debug("Skipping LIME for empty article: %s", title[:60])
            continue

        dominant_raw = max(raw_scores.values()) if raw_scores else 0.0
        influence = _safe_round(final_weight * dominant_raw)

        try:
            logger.info(
                "LIME [%d/%d]: %s (influence=%.4f, samples=%d)",
                rank, len(top_articles), title[:60], influence, num_samples,
            )
            explanation = explainer.explain_instance(
                text_to_explain,
                predict_fn,
                num_features=num_features,
                num_samples=num_samples,
                labels=[target_label_idx],
            )

            raw_weights = explanation.as_list(label=target_label_idx)
            # raw_weights: list of (token, weight) — positive = supports predicted label

            token_weights = []
            for token, weight in raw_weights:
                token_weights.append({
                    "token": token,
                    "weight": round(weight, 6),
                    "direction": "supports" if weight > 0 else "opposes",
                })

            supporting = [t["token"] for t in sorted(token_weights, key=lambda x: x["weight"], reverse=True)
                          if t["weight"] > 0 and not is_lime_noise_token(t["token"], noise)][:5]
            opposing   = [t["token"] for t in sorted(token_weights, key=lambda x: x["weight"])
                          if t["weight"] < 0 and not is_lime_noise_token(t["token"], noise)][:5]

            results.append({
                "rank": rank,
                "title": title,
                "final_weight": _safe_round(final_weight),
                "influence_score": influence,
                "lime_label_explained": predicted_label,
                "token_weights": token_weights,
                "top_tokens_supporting": supporting,
                "top_tokens_opposing": opposing,
            })

        except Exception as exc:
            logger.warning("LIME failed for article '%s': %s", title[:60], exc)
            continue

    logger.info("Token explanation complete: %d articles explained.", len(results))
    return results


def _safe_round(v, d=4):
    try:
        return round(float(v), d)
    except Exception:
        return 0.0
