from __future__ import annotations
import re
import logging
from typing import Optional, List

logger = logging.getLogger(__name__)


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _ticker_regex(ticker: str) -> re.Pattern:
    t = re.escape(ticker.upper())
    return re.compile(rf"(?i)(\${t}\b|\({t}\)|\b{t}\b|\b{t}[-\.][A-Z]{{1,6}}\b)")


def _split_into_sentences(text: str) -> List[str]:
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def _contains_company_reference(
    text: str, 
    company_name: str, 
    ticker_re: Optional[re.Pattern]
) -> bool:
    if not text:
        return False
    
    txt = _normalize_spaces(text)
    low = txt.lower()
    
    # Check company name
    name_phrase = _normalize_spaces(company_name).lower()
    if name_phrase and name_phrase in low:
        return True
    
    # Check ticker
    if ticker_re and ticker_re.search(txt):
        return True
    
    return False


def reduce_content_noise(
    content: str,
    company_name: str,
    ticker: Optional[str] = None,
    keep_all_if_low_match: bool = True,
    min_sentences_threshold: int = 2
) -> tuple[Optional[str], dict]:
    
    if not content:
        return None, {"error": "no_content"}
    
    ticker_re = _ticker_regex(ticker) if ticker else None
    sentences = _split_into_sentences(content)
    
    if not sentences:
        return None, {"error": "no_sentences"}

    relevant_sentences = [
        sent for sent in sentences
        if _contains_company_reference(sent, company_name, ticker_re)
    ]
    
    total_sentences = len(sentences)
    relevant_count = len(relevant_sentences)
    relevance_ratio = relevant_count / total_sentences if total_sentences > 0 else 0
    
    metadata = {
        "total_sentences": total_sentences,
        "relevant_sentences": relevant_count,
        "relevance_ratio": relevance_ratio,
        "filtered": False
    }
    
    if relevant_count == 0:
        logger.debug(f"No relevant sentences for {company_name}, keeping article with no content")
        return None, {**metadata, "filter_action": "no_match"}
    
    elif relevant_count < min_sentences_threshold and keep_all_if_low_match:
        logger.debug(
            f"Only {relevant_count} sentences mention {company_name}, keeping all content for context"
        )
        return content, {**metadata, "filter_action": "kept_all_low_match"}
    
    else:
        filtered_content = " ".join(relevant_sentences)
        logger.debug(
            f"Filtered {company_name}: kept {relevant_count}/{total_sentences} sentences ({relevance_ratio:.1%})"
        )
        return filtered_content, {**metadata, "filtered": True, "filter_action": "filtered"}


def clean_articles_content(
    articles: List[dict],
    company_name: str,
    ticker: Optional[str] = None,
    keep_all_if_low_match: bool = True,
    min_sentences_threshold: int = 2
) -> List[dict]:
    
    if not articles:
        return []
    
    logger.info(
        f"Cleaning article content for {company_name} (ticker: {ticker or 'None'})"
    )
    
    stats = {
        "total_articles": 0,
        "articles_with_content": 0,
        "articles_filtered": 0,
        "articles_kept_all": 0,
        "articles_no_content_after": 0,
        "total_sentences_before": 0,
        "total_sentences_after": 0
    }
    
    for article in articles:
        content = article.get("content")
        
        if not content:
            stats["total_articles"] += 1
            continue
        
        stats["total_articles"] += 1
        stats["articles_with_content"] += 1

        cleaned_content, metadata = reduce_content_noise(
            content,
            company_name,
            ticker,
            keep_all_if_low_match,
            min_sentences_threshold
        )

        article["content"] = cleaned_content
        article["content_stats"] = metadata

        stats["total_sentences_before"] += metadata.get("total_sentences", 0)
        stats["total_sentences_after"] += metadata.get("relevant_sentences", 0)
        
        if metadata.get("filtered"):
            stats["articles_filtered"] += 1
        elif metadata.get("filter_action") == "kept_all_low_match":
            stats["articles_kept_all"] += 1
        
        if cleaned_content is None:
            stats["articles_no_content_after"] += 1

    logger.info(
        f"Content cleaning complete:\n"
        f"  Total articles: {stats['total_articles']}\n"
        f"  Articles with content: {stats['articles_with_content']}\n"
        f"  Articles filtered: {stats['articles_filtered']}\n"
        f"  Articles kept all (low match): {stats['articles_kept_all']}\n"
        f"  Articles with no content after: {stats['articles_no_content_after']}\n"
        f"  Sentences: {stats['total_sentences_before']} → {stats['total_sentences_after']}"
    )
    
    if stats['total_sentences_before'] > 0:
        reduction = (1 - stats['total_sentences_after'] / stats['total_sentences_before']) * 100
        logger.info(f"  Noise reduction: {reduction:.1f}%")
    
    return articles