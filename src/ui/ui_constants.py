"""
ui_constants.py - Shared colour palettes, label maps, and display constants.

Used by both the Streamlit front-end (app.py) and the Matplotlib chart
generators (xai/charts.py) so that visual styling is defined once.
"""
from __future__ import annotations

from weighting.event_classifier import (
    EVENT_FAMILY_TO_PRIOR_HORIZON,
    HORIZON_LABEL_TO_CATEGORY,
)

# Sentiment label colours
LABEL_COLOURS: dict[str, str] = {
    "positive": "#2ecc71",
    "negative": "#e74c3c",
    "neutral":  "#95a5a6",
}

# Evidence-quality rating colours
EVIDENCE_QUALITY_COLOURS: dict[str, str] = {
    "HIGH":   "#2ecc71",
    "MEDIUM": "#f39c12",
    "LOW":    "#e74c3c",
}

# Horizon category colours
HORIZON_COLOURS: dict[str, str] = {
    "IMMEDIATE":  "#e74c3c",
    "SHORT_TERM": "#f39c12",
    "DIFFUSION":  "#3498db",
    "LONG_TERM":  "#2ecc71",
}

# Horizon display labels
HORIZON_DISPLAY_LABELS: dict[str, str] = {
    "IMMEDIATE":  "Immediate (0-1 days)",
    "SHORT_TERM": "Short-term (2-5 days)",
    "DIFFUSION":  "Diffusion (6-10 days)",
    "LONG_TERM":  "Long-term (11-31 days)",
}

# Maps internal horizon codes (e.g. D1_IMMEDIATE) to their display label
_HORIZON_CODE_TO_DISPLAY: dict[str, str] = {
    code: HORIZON_DISPLAY_LABELS.get(cat, cat)
    for code, cat in HORIZON_LABEL_TO_CATEGORY.items()
}


def horizon_text_for_event(event_family: str) -> str:
    """Return a human-readable primary/secondary horizon string for an event family.

    Example output:
        "IMMEDIATE (0-1 days), secondary DIFFUSION (6-10 days)"
    """
    prior = EVENT_FAMILY_TO_PRIOR_HORIZON.get(event_family)
    if not prior:
        return "Unknown"
    pri_cat = HORIZON_LABEL_TO_CATEGORY.get(prior["primary"], "?")
    pri_label = HORIZON_DISPLAY_LABELS.get(pri_cat, pri_cat)
    parts = [pri_label.upper().split(" (")[0] + f" ({pri_label.split('(')[1]}" if "(" in pri_label else pri_cat]

    sec_code = prior.get("secondary")
    if sec_code:
        sec_cat = HORIZON_LABEL_TO_CATEGORY.get(sec_code, "?")
        sec_label = HORIZON_DISPLAY_LABELS.get(sec_cat, sec_cat)
        sec_text = sec_label.upper().split(" (")[0] + f" ({sec_label.split('(')[1]}" if "(" in sec_label else sec_cat
        parts.append(f"secondary {sec_text}")

    return ", ".join(parts)
