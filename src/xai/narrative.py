from __future__ import annotations

import logging
from typing import Any

from config import XAI_LLAMA_MODEL, XAI_LLAMA_TEMPERATURE, XAI_LLAMA_MAX_TOKENS, XAI_LLAMA_ENABLED

logger = logging.getLogger(__name__)

_SYSTEM_MESSAGE = (
    "You are a financial analyst assistant. "
    "Your task is ONLY to narrate pre-computed data provided to you. "
    "STRICT RULES — violating any rule makes your response invalid:\n"
    "1. Use ONLY facts explicitly stated in the DATA SUMMARY block. Do not infer, invent, or extrapolate.\n"
    "2. When describing confidence margin, copy the EXACT qualifier given (e.g. 'narrow margin', 'clear margin'). Do NOT substitute your own words like 'significant' or 'strong'.\n"
    "3. When describing article sentiment distribution, use ONLY the counts provided. Do NOT say 'majority' or 'most' unless the data says so.\n"
    "4. Do not use phrases like 'I think', 'likely', 'probably', 'may', 'seems', or 'appears'.\n"
    "5. Do not mention technical method names like LIME, SHAP, NLI, or transformer.\n"
    "6. Respond in exactly 2-3 sentences.\n"
    "7. Start your first sentence with 'The model predicted'."
)


def _margin_qualifier(margin: float) -> str:
    """Return a precise, grounded qualifier for the label margin so LLaMA copies it verbatim."""
    if margin < 0.10:
        return "narrow margin"
    elif margin < 0.25:
        return "moderate margin"
    else:
        return "clear margin"


def _build_prompt(
    prediction_result: dict[str, Any],
    article_explanation: dict[str, Any],
    pipeline_explanation: dict[str, Any],
    reliability: dict[str, Any],
    token_explanation: list[dict[str, Any]],
    company_name: str,
) -> str:
    final_label = prediction_result.get("final_label", "unknown").upper()
    final_confidence = prediction_result.get("final_confidence", 0.0)
    conf_pct = round(final_confidence * 100, 1)
    normalized = prediction_result.get("normalized_scores", {})
    pos_pct = round(normalized.get("positive", 0) * 100, 1)
    neg_pct = round(normalized.get("negative", 0) * 100, 1)
    neu_pct = round(normalized.get("neutral", 0) * 100, 1)
    articles_analyzed = prediction_result.get("articles_analyzed", 0)
    ticker = prediction_result.get("ticker") or ""
    W = pipeline_explanation.get("prediction_window_days", "?")
    overall_reliability = reliability.get("overall_reliability", "UNKNOWN")

    # Compute margin between top two labels and supply a grounded qualifier
    sorted_scores = sorted(normalized.values(), reverse=True)
    margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else sorted_scores[0]
    margin_qualifier = _margin_qualifier(margin)

    # Article sentiment breakdown counts (so LLaMA can't invent "majority")
    ranked = article_explanation.get("ranked_articles", [])
    sentiment_counts: dict[str, int] = {}
    for a in ranked:
        dom = a.get("dominant_sentiment", "unknown")
        sentiment_counts[dom] = sentiment_counts.get(dom, 0) + 1
    count_pos = sentiment_counts.get("positive", 0)
    count_neg = sentiment_counts.get("negative", 0)
    count_neu = sentiment_counts.get("neutral", 0)
    article_breakdown = (
        f"{count_pos} positive, {count_neg} negative, {count_neu} neutral"
        if ranked else "unknown"
    )

    # Top article
    top_article = ranked[0] if ranked else {}
    top_title = top_article.get("title", "N/A")
    top_weight = top_article.get("final_weight", 0.0)
    top_weight_share = round(top_article.get("weight_share", 0.0) * 100, 1)
    top_dominant = top_article.get("dominant_sentiment", "unknown").upper()
    top_cf = top_article.get("counterfactual", {})
    would_change = top_cf.get("label_would_change", False)
    change_str = "would change" if would_change else "would NOT change"

    # Pipeline averages
    avg_days = pipeline_explanation.get("avg_days_ago", 0)
    horizon_dist = pipeline_explanation.get("horizon_distribution", {})
    most_common_horizon = max(horizon_dist, key=horizon_dist.get) if horizon_dist else "UNKNOWN"
    most_common_count = horizon_dist.get(most_common_horizon, 0)
    horizon_pct = round(most_common_count / articles_analyzed * 100, 1) if articles_analyzed > 0 else 0

    # Reliability flags
    flags = reliability.get("flags", {})
    flag_lines = []
    for flag_name, flag_data in flags.items():
        if flag_data.get("flagged"):
            flag_lines.append(f"  - WARNING: {flag_data['message']}")
    flag_block = "\n".join(flag_lines) if flag_lines else "  (no warnings)"

    # LIME tokens (top article if available)
    lime_line = ""
    if token_explanation:
        top_lime = token_explanation[0]
        supporting = top_lime.get("top_tokens_supporting", [])
        if supporting:
            lime_line = f"- Key words supporting {final_label.lower()}: {', '.join(supporting[:5])}"

    # Pre-build the margin fact as a complete sentence fragment so LLaMA
    # reads it as data, not an instruction to copy a phrase.
    margin_fact = f"The label margin is {margin_qualifier} ({margin:.3f})."

    # Build a warning fact from ACTUAL flagged concerns (not the margin itself)
    active_warnings: list[str] = []
    for flag_name, flag_data in flags.items():
        if flag_data.get("flagged"):
            if flag_name == "source_diversity":
                active_warnings.append(
                    f"source concentration ({flag_data.get('top_domain', 'unknown')} "
                    f"has {flag_data.get('top_domain_share', 0) * 100:.0f}% of articles)"
                )
            elif flag_name == "timing_alignment":
                active_warnings.append("market-close time alignment is not applied")
            elif flag_name == "thin_evidence":
                active_warnings.append("evidence is thin (few articles)")
            elif flag_name == "weight_concentration":
                active_warnings.append("weight is concentrated in one article")
            elif flag_name == "low_confidence":
                active_warnings.append("confidence is below threshold")
            elif flag_name == "label_margin":
                active_warnings.append(f"the label was decided by a {margin_qualifier}")
            elif flag_name == "horizon_coverage":
                active_warnings.append(
                    f"news lookback ({flag_data.get('lookback_days', '?')} days) "
                    f"is shorter than the intended backward window "
                    f"({flag_data.get('intended_lookback_days', '?')} days)"
                )

    if active_warnings:
        warning_fact = (
            f"Reliability is {overall_reliability} due to: "
            + "; ".join(active_warnings) + "."
        )
    else:
        warning_fact = f"Reliability is {overall_reliability} with no concerns."

    prompt = f"""DATA (read-only — narrate these facts, do not add any others):
Company: {company_name}{f' ({ticker})' if ticker else ''}
Prediction label: {final_label} with a {conf_pct}% score share, based on {articles_analyzed} articles over {W} days.
Score breakdown: Positive {pos_pct}%, Negative {neg_pct}%, Neutral {neu_pct}%.
{margin_fact}
Top article: "{top_title}" ({top_dominant} sentiment, {top_weight_share}% of total weight). Removing it {change_str} the label.
Average article age: {avg_days:.1f} days.
{lime_line}
{warning_fact}

Write exactly 3 sentences starting with "The model predicted":
- Sentence 1: state the predicted label in ALL-CAPS exactly as given (e.g. "a NEUTRAL label"), score share (not "confidence"), and article count. The label must appear in UPPER CASE.
- Sentence 2: name the top article and its sentiment.
- Sentence 3: write exactly "The label margin is {margin_qualifier} ({margin:.3f}), but reliability is {overall_reliability} due to " followed by ALL specific reason(s) from the warning above — do not omit any."""

    return prompt


def _build_fallback_summary(
    prediction_result: dict[str, Any],
    article_explanation: dict[str, Any],
    reliability: dict[str, Any],
    company_name: str,
) -> str:
    final_label = prediction_result.get("final_label", "unknown")
    conf_pct = round(prediction_result.get("final_confidence", 0.0) * 100, 1)
    articles_analyzed = prediction_result.get("articles_analyzed", 0)
    overall = reliability.get("overall_reliability", "UNKNOWN")

    ranked = article_explanation.get("ranked_articles", [])
    top = ranked[0] if ranked else {}
    top_title = top.get("title", "N/A")
    top_weight = top.get("final_weight", 0.0)
    would_change = top.get("counterfactual", {}).get("label_would_change", False)
    change_str = "would change" if would_change else "would not change"

    # Compute margin qualifier for sentence clarity
    normalized = prediction_result.get("normalized_scores", {})
    sorted_scores = sorted(normalized.values(), reverse=True)
    margin = (sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
    margin_q = _margin_qualifier(margin)

    # Collect specific reliability concerns (not the margin — it's stated separately)
    flags = reliability.get("flags", {})
    concern_parts: list[str] = []
    if flags.get("source_diversity", {}).get("flagged"):
        sd = flags["source_diversity"]
        concern_parts.append(
            f"source concentration ({sd.get('top_domain', '?')} has "
            f"{sd.get('top_domain_share', 0) * 100:.0f}% of articles)"
        )
    if flags.get("timing_alignment", {}).get("flagged"):
        concern_parts.append("lack of market-close time alignment")
    if flags.get("thin_evidence", {}).get("flagged"):
        concern_parts.append("thin evidence (few articles)")
    if flags.get("weight_concentration", {}).get("flagged"):
        concern_parts.append("weight concentrated in one article")
    if flags.get("low_confidence", {}).get("flagged"):
        concern_parts.append("confidence below threshold")
    if flags.get("label_margin", {}).get("flagged"):
        concern_parts.append(f"the label was decided by a {margin_q}")
    if flags.get("horizon_coverage", {}).get("flagged"):
        hc = flags["horizon_coverage"]
        concern_parts.append(
            f"news lookback ({hc.get('lookback_days', '?')} days) is shorter than "
            f"the intended backward window ({hc.get('intended_lookback_days', '?')} days)"
        )

    if concern_parts and overall != "HIGH":
        reliability_sentence = (
            f"The label margin is {margin_q} ({margin:.3f}), "
            f"but reliability is {overall} due to {'; '.join(concern_parts)}."
        )
    else:
        reliability_sentence = (
            f"The label margin is {margin_q} ({margin:.3f}) "
            f"and prediction reliability is {overall}."
        )

    return (
        f"The model predicted a {final_label.upper()} label for {company_name} "
        f"with a {conf_pct}% score share based on {articles_analyzed} articles. "
        f"The most influential article was \"{top_title}\" (weight {top_weight:.3f}), "
        f"whose removal {change_str} the overall label. "
        f"{reliability_sentence}"
    )


def _validate_narrative(summary: str, prompt: str) -> tuple[bool, list[str]]:
    """
    Post-generation guard: flag hallucinated qualifiers that are NOT grounded
    in the prompt.  Returns (is_clean, list_of_violations).
    """
    violations: list[str] = []
    summary_lower = summary.lower()

    # 1. Magnitude qualifiers that LLaMA tends to invent for margin/confidence
    ungrounded_magnitude = ["significant", "strongly", "overwhelmingly", "substantially"]
    for word in ungrounded_magnitude:
        if word in summary_lower and word not in prompt.lower():
            violations.append(f"hallucinated magnitude qualifier: '{word}'")

    # 2. Distribution claims — only allowed if the exact phrase appears in prompt
    distribution_claims = ["majority", "most articles", "most of the articles", "out of 58", "out of 57", "out of 56"]
    for phrase in distribution_claims:
        if phrase in summary_lower and phrase not in prompt.lower():
            violations.append(f"unsupported distribution claim: '{phrase}'")

    # 3. Invented article count patterns like "34 out of 58" or "X articles had"
    import re
    count_pattern = re.compile(r"\d+\s+(?:out of|of the|articles? (?:had|were|showed))", re.IGNORECASE)
    for match in count_pattern.finditer(summary):
        if match.group(0).lower() not in prompt.lower():
            violations.append(f"invented article count: '{match.group(0)}'")

    # 4. Label must appear in ALL-CAPS (POSITIVE / NEGATIVE / NEUTRAL)
    label_match = re.search(r"Prediction label:\s*(\w+)", prompt)
    if label_match:
        expected_label = label_match.group(1)  # already UPPER in prompt
        # Flag if model wrote the label in lower/title case instead of ALL-CAPS
        title_case = expected_label.capitalize()  # e.g. "Neutral"
        lower_case = expected_label.lower()       # e.g. "neutral"
        for bad_form in [
            f"{title_case} sentiment",
            f"{title_case} prediction",
            f"{title_case} label",
            f"a {title_case} ",
            f"with {title_case} ",
            f"{lower_case} sentiment",
            f"{lower_case} prediction",
            f"{lower_case} label",
            f"a {lower_case} ",
            f"with {lower_case} ",
        ]:
            if bad_form in summary and expected_label not in summary:
                violations.append(
                    f"label not in ALL-CAPS: found '{bad_form.strip()}', "
                    f"expected '{expected_label}'"
                )
                break

    return len(violations) == 0, violations


def generate_narrative(
    prediction_result: dict[str, Any],
    article_explanation: dict[str, Any],
    pipeline_explanation: dict[str, Any],
    reliability: dict[str, Any],
    token_explanation: list[dict[str, Any]],
    company_name: str,
) -> dict[str, Any]:
    if not XAI_LLAMA_ENABLED:
        logger.info("LLaMA narrative disabled via config; using fallback.")
        fallback = _build_fallback_summary(
            prediction_result, article_explanation, reliability, company_name
        )
        return {
            "model": XAI_LLAMA_MODEL,
            "summary": fallback,
            "prompt_used": None,
            "ollama_available": False,
            "fallback_used": True,
            "fallback_summary": fallback,
            "hallucination_violations": [],
        }

    prompt = _build_prompt(
        prediction_result, article_explanation, pipeline_explanation,
        reliability, token_explanation, company_name,
    )

    fallback = _build_fallback_summary(
        prediction_result, article_explanation, reliability, company_name
    )

    try:
        import ollama

        logger.info("Calling Ollama (%s) for narrative summary...", XAI_LLAMA_MODEL)
        response = ollama.chat(
            model=XAI_LLAMA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_MESSAGE},
                {"role": "user",   "content": prompt},
            ],
            options={
                "temperature": XAI_LLAMA_TEMPERATURE,
                "num_predict": XAI_LLAMA_MAX_TOKENS,
                "top_p": 0.9,
            },
        )
        summary = response["message"]["content"].strip()
        logger.info("Narrative generated successfully.")

        is_clean, violations = _validate_narrative(summary, prompt)
        if not is_clean:
            logger.warning(
                "Narrative validation failed (%d violations): %s. "
                "Falling back to template summary.",
                len(violations), violations,
            )
            return {
                "model": XAI_LLAMA_MODEL,
                "summary": fallback,
                "prompt_used": prompt,
                "ollama_available": True,
                "fallback_used": True,
                "fallback_summary": fallback,
                "hallucination_violations": violations,
            }

        return {
            "model": XAI_LLAMA_MODEL,
            "summary": summary,
            "prompt_used": prompt,
            "ollama_available": True,
            "fallback_used": False,
            "fallback_summary": None,
            "hallucination_violations": [],
        }

    except Exception as exc:
        logger.warning("Ollama unavailable (%s); using fallback summary.", exc)
        return {
            "model": XAI_LLAMA_MODEL,
            "summary": fallback,
            "prompt_used": prompt,
            "ollama_available": False,
            "fallback_used": True,
            "fallback_summary": fallback,
            "hallucination_violations": [],
        }
