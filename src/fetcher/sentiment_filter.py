from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple, Set

logger = logging.getLogger(__name__)


# ----------------------------
# Config
# ----------------------------

@dataclass(frozen=True)
class SentimentFilterConfig:
    # Hard drops by title patterns (very noisy article types)
    title_drop_patterns: Tuple[str, ...] = (
        r"fear\s*&\s*greed",
        r"morning memo",
        r"trade strategy",
        r"stocks to open",
        r"santa\s+rally",
        r"\bindex futures?\b",
        r"market clubhouse",
        r"what'?s driving markets",
        r"\bpre[- ]market\b",
        r"\bday trading\b",
        r"\blevels?\b.*\bspy\b|\bspy\b.*\blevels?\b",
    )

    # Strong trading-level detection inside content
    trading_content_patterns: Tuple[str, ...] = (
        r"\b(support|resistance|bull target|bear target|auctioning|breakout|pullback)\b",
        r"\b(high bull target|low bear target)\b",
        r"\b(price levels are updated every day)\b",
    )

    # Boilerplate or promo sentences to remove
    boilerplate_sentence_patterns: Tuple[str, ...] = (
        r"\ba newsletter built for market enthusiasts\b",
        r"\bthis content was partially produced\b",
        r"©\s*\d{4}\s*benzinga\.com\b",
        r"\bto add\s*benzinga news\b",
        r"\bread next\b",
        r"\bimage via\b|\bphoto courtesy\b",
        r"\bclick here\b",
        r"\bthank you for your support\b",
        r"\bwe may receive a commission\b",
        r"\bthis article was generated\b",
        r"\bmoney back guarantee\b",
        r"\bour methodology\b",
        r"\bwe recently compiled a list\b",
        r"\bcancel anytime\b",
        r"\bexclusive.*members\b",
        r"\bsign up\b.*\bnewsletter\b",
        r"\bdisclaimer\b",
    )

    # Event keywords: require at least N hits to keep (aggressive)
    min_event_hits: int = 2

    # If there are many other tickers and low event density, drop as roundup
    max_other_tickers: int = 2
    min_event_hits_if_roundup: int = 3

    # Minimum usable length after cleaning
    min_words_after_clean: int = 60

    # Numeric density threshold (when combined with trading patterns => drop)
    max_numeric_token_ratio: float = 0.07

    # Sentence extraction: keep up to this many best sentences
    max_sentences_out: int = 8

    # Leakage masking
    mask_prices_and_targets: bool = True


# ----------------------------
# Regex helpers
# ----------------------------

_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_PAREN_TICKER_RE = re.compile(
    r"\((?:NASDAQ|NYSE|AMEX|OTC|LSE|TSX|FWB|HKEX|ASX)\s*:\s*([A-Z0-9\.-]{1,10})\)",
    re.IGNORECASE,
)
_DOLLAR_TICKER_RE = re.compile(r"\$([A-Z]{1,6})\b")
_WS_RE = re.compile(r"\s+")

# Leakage-ish numeric patterns
_DOLLAR_AMOUNT_RE = re.compile(r"\$\s?\d+(?:\.\d+)?")
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s?%")
_PRICE_TARGET_RE = re.compile(r"(?i)\bprice target\b|\bPT\b")
_MARKET_CAP_RE = re.compile(r"(?i)\bmarket\s*cap(?:italization)?\b|\bvaluation\b")
_PE_RE = re.compile(r"(?i)\bP/E\b|\bPE ratio\b")
_BIG_NUMBER_RE = re.compile(r"\b\d{1,3}(?:,\d{3})+(?:\.\d+)?\b")


# Event keyword groups (counted as "hits")
_EVENT_GROUPS: Tuple[re.Pattern, ...] = (
    re.compile(r"(?i)\bearnings\b|\bEPS\b|\brevenue\b|\bguidance\b|\boutlook\b|\bmargins?\b"),
    re.compile(r"(?i)\bsales\b|\bdemand\b|\bship(?:ment|ped|ping)s?\b|\bupgrade cycle\b|\bchannel checks?\b"),
    re.compile(r"(?i)\bApple Intelligence\b|\bAI\b|\bSiri\b|\bChatGPT\b|\biOS\b|\bvision pro\b"),
    re.compile(r"(?i)\bregulator\b|\binvestigation\b|\bantitrust\b|\blawsuit\b|\bCMA\b|\bFTC\b|\bDoJ\b"),
    re.compile(r"(?i)\bpartnership\b|\bcollaboration\b|\bsupplier\b|\bchip\b|\bTSMC\b|\bBroadcom\b|\bQualcomm\b|\bmodem\b"),
    re.compile(r"(?i)\bChina\b|\bEurope\b|\bU\.K\.\b|\bUK\b|\bTariff\b"),
)


def _normalize_spaces(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").replace("\r", " ").replace("\n", " ").replace("\t", " ")).strip()


def _split_sentences(text: str) -> List[str]:
    txt = _normalize_spaces(text)
    if not txt:
        return []
    return [s.strip() for s in _SENT_SPLIT_RE.split(txt) if s.strip()]


def _numeric_token_ratio(text: str) -> float:
    toks = (text or "").split()
    if not toks:
        return 0.0
    numeric = sum(1 for t in toks if any(ch.isdigit() for ch in t))
    return numeric / max(1, len(toks))


def _extract_other_tickers(text: str, target_ticker: Optional[str]) -> Set[str]:
    tickers: Set[str] = set()
    for m in _PAREN_TICKER_RE.finditer(text or ""):
        tickers.add(m.group(1).upper())
    for m in _DOLLAR_TICKER_RE.finditer(text or ""):
        tickers.add(m.group(1).upper())

    if target_ticker:
        tickers.discard(target_ticker.upper())

    # Remove common indices/ETFs/crypto noise
    for t in {"SPY", "QQQ", "DIA", "IWM", "SPX", "DJI", "IXIC", "VIX", "BTC", "ETH"}:
        tickers.discard(t)

    return tickers


def _count_event_hits(text: str) -> int:
    return sum(1 for rgx in _EVENT_GROUPS if rgx.search(text or ""))


def _mentions_company(text: str, company_name: str, ticker: Optional[str]) -> bool:
    low = (text or "").lower()
    if company_name and company_name.lower() in low:
        return True
    if ticker and re.search(rf"(?i)\b{re.escape(ticker)}\b", text or ""):
        return True
    return False


# ----------------------------
# Cleaning and filtering
# ----------------------------

def strip_boilerplate_sentences(text: str, cfg: SentimentFilterConfig) -> str:
    sents = _split_sentences(text)
    if not sents:
        return ""

    boiler = [re.compile(p, re.IGNORECASE) for p in cfg.boilerplate_sentence_patterns]
    kept = [s for s in sents if not any(r.search(s) for r in boiler)]
    return _normalize_spaces(" ".join(kept))


def mask_price_leakage(text: str) -> str:
    """
    Masks common leakage patterns (prices, percents, market cap, P/E, price target).
    Keep it simple so you do not destroy meaning.
    """
    if not text:
        return ""

    t = text
    t = _DOLLAR_AMOUNT_RE.sub("<PRICE>", t)
    t = _PERCENT_RE.sub("<PCT>", t)
    t = _BIG_NUMBER_RE.sub("<NUM>", t)

    # Keep words, just mask the signal phrases a bit
    t = _PRICE_TARGET_RE.sub("<PRICE_TARGET>", t)
    t = _MARKET_CAP_RE.sub("<MARKET_CAP>", t)
    t = _PE_RE.sub("<PE>", t)

    return _normalize_spaces(t)


def extract_event_sentences(
    text: str,
    company_name: str,
    ticker: Optional[str],
    cfg: SentimentFilterConfig,
) -> str:
    """
    Keeps only sentences that are both:
      - company-linked (name or ticker)
      - event-ish (contain at least one event group hit)
    Falls back to first 3 sentences if everything is removed.
    """
    sents = _split_sentences(text)
    if not sents:
        return ""

    scored: List[Tuple[int, str]] = []
    for s in sents:
        if not _mentions_company(s, company_name, ticker):
            continue
        hits = _count_event_hits(s)
        if hits <= 0:
            continue
        scored.append((hits, s))

    if not scored:
        # fallback: keep a small, stable slice
        return _normalize_spaces(" ".join(sents[:3]))

    # Sort by event-hit count desc, keep top-K, preserve original order among chosen
    scored.sort(key=lambda x: x[0], reverse=True)
    top = set(s for _, s in scored[: cfg.max_sentences_out])

    out: List[str] = []
    for s in sents:
        if s in top:
            out.append(s)
        if len(out) >= cfg.max_sentences_out:
            break

    return _normalize_spaces(" ".join(out))


def is_low_signal_article(
    title: str,
    content: str,
    company_name: str,
    ticker: Optional[str],
    cfg: SentimentFilterConfig,
) -> Tuple[bool, str, Dict]:
    """
    Returns (drop?, reason, diagnostics)
    """
    title_n = _normalize_spaces(title)
    content_n = _normalize_spaces(content)

    # Hard drop by title patterns
    for pat in cfg.title_drop_patterns:
        if re.search(pat, title_n, flags=re.IGNORECASE):
            return True, "title_drop", {"matched": pat}

    # Trading-level content drop
    trading_hit = any(re.search(p, content_n, flags=re.IGNORECASE) for p in cfg.trading_content_patterns)
    num_ratio = _numeric_token_ratio(content_n)
    if trading_hit and num_ratio > cfg.max_numeric_token_ratio:
        return True, "trading_levels", {"numeric_ratio": round(num_ratio, 4)}

    # Must mention company somewhere
    if not _mentions_company(title_n + " " + content_n, company_name, ticker):
        return True, "no_company_mention", {}

    # Count event hits and roundups
    other_tickers = _extract_other_tickers(title_n + " " + content_n, ticker)
    hits = _count_event_hits(content_n)

    if len(other_tickers) > cfg.max_other_tickers and hits < cfg.min_event_hits_if_roundup:
        return True, "roundup_multi_ticker", {"other_tickers": sorted(other_tickers), "event_hits": hits}

    if hits < cfg.min_event_hits:
        return True, "low_event_density", {"event_hits": hits}

    # Length sanity check
    if len(content_n.split()) < cfg.min_words_after_clean:
        return True, "too_short_after_clean", {"words": len(content_n.split())}

    return False, "keep", {"event_hits": hits, "other_tickers": sorted(other_tickers), "numeric_ratio": round(num_ratio, 4)}


def aggressive_filter_for_sentiment(
    articles: List[dict],
    company_name: str,
    ticker: Optional[str] = None,
    cfg: Optional[SentimentFilterConfig] = None,
) -> List[dict]:
    """
    Aggressive filter to make content more valuable for sentiment classification.
    Mutates article["content"] (cleaned, event-sentence snippet, leakage-masked).
    Adds article["sentiment_filter"] diagnostics.
    """
    if cfg is None:
        cfg = SentimentFilterConfig()

    kept: List[dict] = []
    dropped = 0

    for a in articles or []:
        title = a.get("title") or ""
        content = a.get("content") or ""
        if not content:
            a["sentiment_filter"] = {"kept": False, "reason": "no_content"}
            dropped += 1
            continue

        # Step 1: strip boilerplate sentences
        c1 = strip_boilerplate_sentences(content, cfg)

        # Step 2: decide drop/keep
        drop, reason, diag = is_low_signal_article(title, c1, company_name, ticker, cfg)
        if drop:
            a["sentiment_filter"] = {"kept": False, "reason": reason, "diag": diag}
            dropped += 1
            continue

        # Step 3: extract event-focused sentences
        c2 = extract_event_sentences(c1, company_name, ticker, cfg)

        # Step 4: mask leakage if enabled
        if cfg.mask_prices_and_targets:
            c2 = mask_price_leakage(c2)

        a["content"] = c2
        a["sentiment_filter"] = {"kept": True, "reason": "keep", "diag": diag}
        kept.append(a)

    logger.info("aggressive_filter_for_sentiment: in=%d kept=%d dropped=%d", len(articles or []), len(kept), dropped)
    return kept
