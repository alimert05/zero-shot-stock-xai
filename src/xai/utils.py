from __future__ import annotations

from typing import Any


def herfindahl_index(weights: list[float]) -> float:
    total = sum(weights)
    if total == 0:
        return 0.0
    shares = [w / total for w in weights]
    return round(sum(s ** 2 for s in shares), 6)


def safe_round(value: float | None, digits: int = 4) -> float:
    if value is None:
        return 0.0
    return round(float(value), digits)


def label_index(label: str) -> int:
    return {"positive": 0, "negative": 1, "neutral": 2}.get(label, 0)


def top_n_by_key(
    items: list[dict],
    key: str,
    n: int,
    reverse: bool = True,
) -> list[dict]:
    return sorted(items, key=lambda x: x.get(key, 0), reverse=reverse)[:n]


def get_dominant_label(raw_scores: dict[str, float]) -> str:
    return max(raw_scores, key=raw_scores.get)


def scores_to_pct(scores: dict[str, float]) -> dict[str, float]:
    return {k: round(v * 100, 2) for k, v in scores.items()}
