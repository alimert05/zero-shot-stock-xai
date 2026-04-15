"""Benchmark zero-shot NLI models on Financial PhraseBank CSV files.

Supports DeBERTa, RoBERTa, FinBERT, FinGPT, and Ollama-based models.

Usage (benchmark mode):
    python -m testing.evaluation.phrasebank_benchmark --model <model_name>
    python -m testing.evaluation.phrasebank_benchmark --model <model_name> --config <config_name>
    python -m testing.evaluation.phrasebank_benchmark --model <model_name> --max-samples 200

Usage (label tuning mode - K-fold CV with 112 template x label configs):
    python -m testing.evaluation.phrasebank_benchmark --mode tune --model <model_name>
    python -m testing.evaluation.phrasebank_benchmark --mode tune --model <model_name> --resume
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from config import (
    FPB_PATH, FPB_DATASETS_PATH, SENTIMENT_DEVICE,
    DECISION_THRESHOLD_ENABLED,
    MODEL_NAME,
    SENTIMENT_MODEL,
)

logger = logging.getLogger(__name__)

DEFAULT_DATASETS = [
    str(FPB_DATASETS_PATH / "financial_phrasebank_sentences_50agree.csv"),
    str(FPB_DATASETS_PATH / "financial_phrasebank_sentences_75agree.csv"),
    str(FPB_DATASETS_PATH / "financial_phrasebank_sentences_allagree.csv"),
]

_MODEL_TAG = SENTIMENT_MODEL.replace("/", "_").replace("-", "_")
DEFAULT_OUTPUT = FPB_PATH / f"phrasebank_model_benchmark_{_MODEL_TAG}.json"
CHECKPOINT_PATH = FPB_PATH / f"label_tuning_checkpoint_{_MODEL_TAG}.json"
LABELS = ["positive", "negative", "neutral"]

# Coarse screen uses the largest dataset (50agree) for stable estimates.
COARSE_DATASET_IDX = 0


# Hypothesis templates x label maps - combinatorial search space.
# Cross-product yields 14 x 8 = 112 configurations.

HYPOTHESIS_TEMPLATES: dict[str, str] = {
    "financial_statement": "This financial statement is {}.",
    "overall_sentiment": "The overall sentiment indicates {}.",
    "expresses_sentiment": "This text expresses {} sentiment.",
    "tone_financial": "The tone of this financial text is {}.",
    "news_for_company": "This news is {} for the company.",
    "financial_assessment": "The financial assessment in this text is {}.",
    "reflects_outlook": "This statement reflects a {} outlook for the company.",
    "market_signal": "The market signal from this text is {}.",
    "sentiment_of_text": "The sentiment of this text is {}.",
    "conveys_message": "This financial news conveys a {} message.",
    "authors_attitude": "The author's attitude in this text is {}.",
    "about_outlook": "This text is {} about the financial outlook.",
    "implication": "The implication of this statement is {}.",
    "investor_implications": "This news has {} implications for investors.",
}

LABEL_MAPS: dict[str, dict[str, str]] = {
    "simple": {
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral",
    },
    "outlook": {
        "positive": "positive financial outlook",
        "negative": "negative financial outlook",
        "neutral": "neutral financial outlook",
    },
    "optimistic": {
        "positive": "optimistic",
        "negative": "pessimistic",
        "neutral": "neutral",
    },
    "good_bad": {
        "positive": "good",
        "negative": "bad",
        "neutral": "neutral",
    },
    "favorable": {
        "positive": "favorable",
        "negative": "unfavorable",
        "neutral": "neutral",
    },
    "bullish": {
        "positive": "bullish",
        "negative": "bearish",
        "neutral": "neutral",
    },
    "growth": {
        "positive": "growth",
        "negative": "decline",
        "neutral": "stable",
    },
    "confidence": {
        "positive": "confident",
        "negative": "concerned",
        "neutral": "uncertain",
    },
}


def generate_configs() -> list[dict[str, Any]]:
    """Generate cross-product of hypothesis templates x label maps."""
    configs: list[dict[str, Any]] = []
    for t_idx, (t_name, template) in enumerate(HYPOTHESIS_TEMPLATES.items()):
        for lm_idx, (lm_name, label_map) in enumerate(LABEL_MAPS.items()):
            configs.append({
                "config_id": f"t{t_idx + 1:02d}_lm{lm_idx + 1:02d}",
                "template_name": t_name,
                "label_map_name": lm_name,
                "name": f"{t_name}__{lm_name}",
                "hypothesis_template": template,
                "label_map": label_map,
            })
    return configs


# Helpers

def normalize_label(value: str | int | None) -> str:
    """Normalize labels to the canonical set: positive, negative, neutral."""
    if value is None:
        return "neutral"

    raw = str(value).strip().lower()
    if raw in {"2", "label_2", "pos", "positive"}:
        return "positive"
    if raw in {"0", "label_0", "neg", "negative"}:
        return "negative"
    if raw in {"1", "label_1", "neu", "neutral"}:
        return "neutral"

    if "positive" in raw:
        return "positive"
    if "negative" in raw:
        return "negative"
    if "neutral" in raw:
        return "neutral"
    return "neutral"


@dataclass
class DatasetSplit:
    path: str
    name: str
    texts: list[str]
    labels: list[str]


def load_phrasebank_csv(path: str, max_samples: int | None = None) -> DatasetSplit:
    """Load a Financial PhraseBank CSV file and return texts with labels."""
    texts: list[str] = []
    labels: list[str] = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sentence = (row.get("sentence") or "").strip()
            if not sentence:
                continue

            label_text = row.get("label_text")
            label_value = label_text if label_text else row.get("label")
            label = normalize_label(label_value)

            texts.append(sentence)
            labels.append(label)

            if max_samples and len(texts) >= max_samples:
                break

    if not texts:
        raise ValueError(f"No samples loaded from dataset: {path}")

    return DatasetSplit(
        path=path,
        name=Path(path).name,
        texts=texts,
        labels=labels,
    )


# Splitting

def stratified_split(
    texts: list[str],
    labels: list[str],
    test_ratio: float = 0.5,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Stratified split preserving label distribution.

    Returns (dev_texts, dev_labels, test_texts, test_labels).
    """
    rng = random.Random(seed)

    by_label: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        by_label.setdefault(label, []).append(i)

    dev_idx: list[int] = []
    test_idx: list[int] = []

    for _label, indices in sorted(by_label.items()):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        split_point = max(1, int(len(shuffled) * (1 - test_ratio)))
        dev_idx.extend(shuffled[:split_point])
        test_idx.extend(shuffled[split_point:])

    rng.shuffle(dev_idx)
    rng.shuffle(test_idx)

    return (
        [texts[i] for i in dev_idx],
        [labels[i] for i in dev_idx],
        [texts[i] for i in test_idx],
        [labels[i] for i in test_idx],
    )


def stratified_kfold(
    labels: list[str],
    n_folds: int = 5,
    seed: int = 42,
) -> list[tuple[list[int], list[int]]]:
    """Stratified K-fold cross-validation (indices only).

    Returns list of (train_indices, val_indices) for each fold.
    Every sample appears in exactly one validation fold.
    """
    rng = random.Random(seed)

    by_label: dict[str, list[int]] = {}
    for i, label in enumerate(labels):
        by_label.setdefault(label, []).append(i)

    # Shuffle each label group and distribute round-robin into folds
    folds: list[list[int]] = [[] for _ in range(n_folds)]
    for _label, indices in sorted(by_label.items()):
        shuffled = indices[:]
        rng.shuffle(shuffled)
        for i, idx in enumerate(shuffled):
            folds[i % n_folds].append(idx)

    for fold in folds:
        rng.shuffle(fold)

    result: list[tuple[list[int], list[int]]] = []
    for val_fold_idx in range(n_folds):
        val_indices = folds[val_fold_idx]
        train_indices: list[int] = []
        for k in range(n_folds):
            if k != val_fold_idx:
                train_indices.extend(folds[k])
        rng.shuffle(train_indices)
        result.append((train_indices, val_indices))

    return result


# Dataset metadata visualisation

BAR_WIDTH = 30


def _bar(ratio: float, width: int = BAR_WIDTH) -> str:
    """Render a text-based progress bar from a 0-1 ratio."""
    filled = round(ratio * width)
    return "#" * filled + "." * (width - filled)


def _print_label_dist(dist: Counter, total: int, indent: str = "    ") -> None:
    """Print a per-label distribution with count, percentage, and visual bar."""
    for label in LABELS:
        count = dist.get(label, 0)
        pct = (count / total * 100) if total > 0 else 0.0
        bar = _bar(count / total if total > 0 else 0.0)
        print(f"{indent}{label:<10}: {count:>6}  ({pct:>5.1f}%)  |{bar}|")


def print_dataset_metadata(
    datasets: list[DatasetSplit],
    title: str = "DATASET OVERVIEW",
) -> None:
    """Print visual metadata for one or more datasets."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

    for ds in datasets:
        total = len(ds.labels)
        dist = Counter(ds.labels)

        print(f"\n  File  : {ds.name}")
        print(f"  Total : {total} samples")
        print()
        _print_label_dist(dist, total, indent="    ")

        # Imbalance summary
        majority_label = max(dist, key=dist.get)
        majority_pct = dist[majority_label] / total * 100
        minority_label = min(dist, key=dist.get)
        minority_count = dist[minority_label]
        ratio = dist[majority_label] / max(minority_count, 1)
        print(f"\n    Majority class : {majority_label} ({majority_pct:.1f}%)")
        print(f"    Imbalance ratio: {ratio:.1f}:1 ({majority_label} vs {minority_label})")
        print("-" * 70)

    print()


def _short_ds_name(name: str) -> str:
    """Extract short dataset tag like '50agree' from a full filename."""
    for tag in ("50agree", "75agree", "allagree"):
        if tag in name:
            return tag
    return name[:12]


def print_holdout_metadata(
    dev_labels: list[str],
    test_labels: list[str],
    dataset_name: str,
    test_ratio: float,
    n_folds: int,
    seed: int,
) -> None:
    """Print visual metadata for holdout + CV split."""
    dev_dist = Counter(dev_labels)
    test_dist = Counter(test_labels)
    dev_total = len(dev_labels)
    test_total = len(test_labels)
    total = dev_total + test_total

    print("\n" + "=" * 70)
    print("  HOLDOUT + CV SPLIT")
    print("=" * 70)
    print(f"  Dataset : {dataset_name}")
    print(f"  Total   : {total} samples")
    print(
        f"  Holdout : {(1 - test_ratio) * 100:.0f}% dev / "
        f"{test_ratio * 100:.0f}% test  (seed={seed})"
    )
    print(f"  CV      : {n_folds}-fold on dev set (~{dev_total // n_folds} samples/fold)")

    print(f"\n  DEV SET ({dev_total} samples)")
    _print_label_dist(dev_dist, dev_total)

    print(f"\n  TEST SET ({test_total} samples) -- held out until Stage 3")
    _print_label_dist(test_dist, test_total)

    # Stratification check
    print(f"\n  Stratification check (dev% vs test%):")
    for label in LABELS:
        d_pct = (dev_dist.get(label, 0) / dev_total * 100) if dev_total > 0 else 0
        t_pct = (test_dist.get(label, 0) / test_total * 100) if test_total > 0 else 0
        delta = d_pct - t_pct
        print(f"    {label:<10}: dev={d_pct:>5.1f}%  test={t_pct:>5.1f}%  delta={delta:>+5.1f}%")

    print("=" * 70 + "\n")


# Metrics

def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    """Compute accuracy, macro precision/recall/F1, and per-class metrics."""
    if len(y_true) != len(y_pred):
        raise ValueError(f"Mismatched lengths: y_true={len(y_true)} y_pred={len(y_pred)}")

    total = len(y_true)
    if total == 0:
        return {
            "accuracy": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f1": 0.0,
            "f1_scores": {label: 0.0 for label in LABELS},
            "per_class": {},
            "total": 0,
            "correct": 0,
        }

    labels_present = sorted(set(LABELS) | set(y_true) | set(y_pred))
    confusion = {actual: {pred: 0 for pred in labels_present} for actual in labels_present}
    for actual, pred in zip(y_true, y_pred):
        confusion[actual][pred] += 1

    correct = sum(1 for actual, pred in zip(y_true, y_pred) if actual == pred)
    accuracy = correct / total

    per_class: dict[str, dict[str, Any]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []

    for label in labels_present:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels_present if other != label)
        fn = sum(confusion[label][other] for other in labels_present if other != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

        if support > 0:
            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1)

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    f1_scores = {
        label: per_class.get(label, {}).get("f1", 0.0)
        for label in LABELS
    }

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "f1_scores": f1_scores,
        "per_class": per_class,
        "confusion_matrix": confusion,
        "total": total,
        "correct": correct,
    }

def _chunks(items: list[str], size: int) -> list[list[str]]:
    """Split a list into fixed-size sublists for batched processing."""
    return [items[i:i + size] for i in range(0, len(items), size)]


class DeBERTaPredictor:
    def __init__(self, device: int, batch_size: int):
        """Initialise DeBERTa predictor with the zero-shot NLI pipeline."""
        from transformers import pipeline
        from predictors.zero_shot import _CLASS_TO_LABEL, _HYPOTHESIS_TEMPLATE, _LABEL_TO_CLASS

        self.batch_size = batch_size
        self.max_length = 512
        self.label_map = _CLASS_TO_LABEL          # {"positive": "bullish financial outlook", ...}
        self.reverse_map = _LABEL_TO_CLASS         # {"bullish financial outlook": "positive", ...}
        self.candidate_labels = list(self.label_map.values())
        self.hypothesis_template = _HYPOTHESIS_TEMPLATE

        logger.info("Loading DeBERTa (zero-shot) model...")
        self.pipe = pipeline(
            "zero-shot-classification",
            model="microsoft/deberta-large-mnli",
            device=device,
        )
        if getattr(self.pipe.tokenizer, "model_max_length", 0) > 10000:
            self.pipe.tokenizer.model_max_length = self.max_length
        logger.info("DeBERTa model ready")

    def predict(self, texts: list[str]) -> list[str]:
        """Classify texts using DeBERTa zero-shot NLI."""
        predictions: list[str] = []
        for batch in _chunks(texts, self.batch_size):
            outputs = self.pipe(
                batch,
                candidate_labels=self.candidate_labels,
                hypothesis_template=self.hypothesis_template,
                multi_label=True,
                batch_size=self.batch_size,
                truncation=True,
                max_length=512,
            )
            if isinstance(outputs, dict):
                outputs = [outputs]

            for item in outputs:
                nli_labels = item.get("labels", [])
                nli_scores = item.get("scores", [])

                # Map NLI labels -> canonical scores
                scores = {}
                for lbl, sc in zip(nli_labels, nli_scores):
                    canonical = self.reverse_map.get(lbl.lower().strip(), None)
                    if canonical:
                        scores[canonical] = sc
                for cls in ("positive", "negative", "neutral"):
                    scores.setdefault(cls, 0.0)

                # Apply abstention (thresholds + dynamic margin if enabled)
                from predictors.abstention import apply_abstention
                abstention_result = apply_abstention(scores, article_weights=[1.0])

                predictions.append(abstention_result["final_label"])
        return predictions

    def predict_with_config(
        self,
        texts: list[str],
        config: dict[str, Any],
    ) -> list[str]:
        """Predict using a specific label configuration (for tuning)."""
        label_map = config["label_map"]
        candidate_labels = list(label_map.values())
        reverse_map = {v.lower().strip(): k for k, v in label_map.items()}
        hypothesis_template = config["hypothesis_template"]

        predictions: list[str] = []
        for batch in _chunks(texts, self.batch_size):
            outputs = self.pipe(
                batch,
                candidate_labels=candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=True,
                batch_size=self.batch_size,
                truncation=True,
                max_length=512,
            )
            if isinstance(outputs, dict):
                outputs = [outputs]

            for item in outputs:
                top_nli_label = item.get("labels", ["neutral"])[0]
                canonical = reverse_map.get(
                    top_nli_label.lower().strip(), "neutral"
                )
                predictions.append(canonical)
        return predictions


class RoBERTaPredictor:
    """Zero-shot NLI predictor using RoBERTa-Large-MNLI."""

    def __init__(self, device: int, batch_size: int):
        """Initialise RoBERTa predictor with the zero-shot NLI pipeline."""
        from transformers import pipeline

        self.batch_size = batch_size
        self.max_length = 512

        logger.info("Loading RoBERTa (zero-shot) model...")
        self.pipe = pipeline(
            "zero-shot-classification",
            model="roberta-large-mnli",
            device=device,
        )
        if getattr(self.pipe.tokenizer, "model_max_length", 0) > 10000:
            self.pipe.tokenizer.model_max_length = self.max_length
        logger.info("RoBERTa model ready")

    def predict(self, texts: list[str]) -> list[str]:
        """Classify texts using RoBERTa zero-shot NLI."""
        predictions: list[str] = []
        for batch in _chunks(texts, self.batch_size):
            outputs = self.pipe(
                batch,
                candidate_labels=["positive", "negative", "neutral"],
                multi_label=True,
                batch_size=self.batch_size,
                truncation=True,
                max_length=512,
            )
            if isinstance(outputs, dict):
                outputs = [outputs]
            for item in outputs:
                top_label = item.get("labels", ["neutral"])[0]
                predictions.append(top_label)
        return predictions

    def predict_with_config(
        self,
        texts: list[str],
        config: dict[str, Any],
    ) -> list[str]:
        """Predict using a specific label configuration (for tuning)."""
        label_map = config["label_map"]
        candidate_labels = list(label_map.values())
        reverse_map = {v.lower().strip(): k for k, v in label_map.items()}
        hypothesis_template = config["hypothesis_template"]

        predictions: list[str] = []
        for batch in _chunks(texts, self.batch_size):
            outputs = self.pipe(
                batch,
                candidate_labels=candidate_labels,
                hypothesis_template=hypothesis_template,
                multi_label=True,
                batch_size=self.batch_size,
                truncation=True,
                max_length=512,
            )
            if isinstance(outputs, dict):
                outputs = [outputs]

            for item in outputs:
                top_nli_label = item.get("labels", ["neutral"])[0]
                canonical = reverse_map.get(
                    top_nli_label.lower().strip(), "neutral"
                )
                predictions.append(canonical)
        return predictions


class OllamaPredictor:
    """Predict sentiment using a local LLM via Ollama (Llama 3.1 8B / Mistral 7B)."""

    def __init__(self, model_name: str):
        """Initialise Ollama predictor with the specified model name."""
        import ollama as _ollama
        self.ollama = _ollama
        self.model_name = model_name
        logger.info("OllamaPredictor ready (model=%s)", model_name)

    def predict(self, texts: list[str]) -> list[str]:
        """Classify texts using Ollama LLM."""
        from predictors.ollama_predictor import OLLAMA_PROMPT_TEMPLATE

        predictions: list[str] = []
        for i, text in enumerate(texts):
            prompt = OLLAMA_PROMPT_TEMPLATE.format(text=text)
            response = self.ollama.chat(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.0, "num_predict": 10},
            )
            answer = response["message"]["content"].strip().lower()

            if "positive" in answer:
                label = "positive"
            elif "negative" in answer:
                label = "negative"
            elif "neutral" in answer:
                label = "neutral"
            else:
                logger.warning("Ollama (%s) unexpected answer: '%s', defaulting to neutral", self.model_name, answer)
                label = "neutral"

            predictions.append(label)

            if (i + 1) % 100 == 0:
                logger.info("OllamaPredictor progress: %d/%d", i + 1, len(texts))

        return predictions


class FinBERTPredictor:
    """Predict sentiment using FinBERT (ProsusAI/finbert)."""

    def __init__(self, device: int):
        """Initialise FinBERT predictor with the sentiment-analysis pipeline."""
        from transformers import pipeline
        logger.info("Loading FinBERT model...")
        self.pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=device)
        logger.info("FinBERT model ready")

    def predict(self, texts: list[str]) -> list[str]:
        """Classify texts using FinBERT."""
        predictions: list[str] = []
        for text in texts:
            results = self.pipe(text, top_k=None, truncation=True, max_length=512)
            best = max(results, key=lambda x: x["score"])
            predictions.append(best["label"])
        return predictions


class FinGPTPredictor:
    """Predict sentiment using FinGPT (Llama-2-13B + LoRA)."""

    def __init__(self, device: int):
        """Initialise FinGPT predictor with the LoRA-adapted model."""
        from predictors.fingpt import _get_fingpt_model, FINGPT_PROMPT
        self.model, self.tokenizer = _get_fingpt_model()
        self.prompt_template = FINGPT_PROMPT
        self.device = device
        logger.info("FinGPT model ready for FPB benchmark")

    def predict(self, texts: list[str]) -> list[str]:
        """Classify texts using FinGPT."""
        import torch

        predictions: list[str] = []
        for i, text in enumerate(texts):
            prompt = self.prompt_template.format(text=text)
            tokens = self.tokenizer(
                prompt, return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            )
            device = next(self.model.parameters()).device
            tokens = {k: v.to(device) for k, v in tokens.items()}

            with torch.no_grad():
                output = self.model.generate(**tokens, max_new_tokens=5, do_sample=False)

            generated = self.tokenizer.decode(output[0], skip_special_tokens=True)
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

            predictions.append(label)

            if (i + 1) % 100 == 0:
                logger.info("FinGPTPredictor progress: %d/%d", i + 1, len(texts))

        return predictions

Predictor = DeBERTaPredictor | RoBERTaPredictor | OllamaPredictor | FinBERTPredictor | FinGPTPredictor

# Benchmark mode

def _resolve_config(config_name: str | None) -> dict[str, Any] | None:
    """Look up a config by its composite name (e.g. 'about_outlook__outlook')."""
    if config_name is None:
        return None
    all_configs = generate_configs()
    for cfg in all_configs:
        if cfg["name"] == config_name:
            return cfg
    available = [c["name"] for c in all_configs]
    raise ValueError(
        f"Config '{config_name}' not found. Available configs:\n"
        + "\n".join(f"  - {n}" for n in available)
    )


def _create_predictor(args: argparse.Namespace):
    """Create the appropriate predictor based on --model argument."""
    model = getattr(args, "model", "deberta")

    if model == "deberta":
        return DeBERTaPredictor(device=args.device, batch_size=args.batch_size)
    elif model == "roberta":
        return RoBERTaPredictor(device=args.device, batch_size=args.batch_size)
    elif model == "ollama-llama3":
        return OllamaPredictor(model_name="llama3.1:8b")
    elif model == "ollama-mistral":
        return OllamaPredictor(model_name="mistral:7b")
    elif model == "finbert":
        return FinBERTPredictor(device=args.device)
    elif model == "fingpt":
        return FinGPTPredictor(device=args.device)
    else:
        raise ValueError(f"Unknown model: {model}")


def _model_display_name(model: str) -> str:
    """Map a short model key to its full display name."""
    names = {
        "deberta": "microsoft/deberta-large-mnli",
        "roberta": "roberta-large-mnli",
        "ollama-llama3": "llama3.1:8b (Ollama)",
        "ollama-mistral": "mistral:7b (Ollama)",
        "finbert": "ProsusAI/finbert",
        "fingpt": "FinGPT (Llama-2-13B + LoRA)",
    }
    return names.get(model, model)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run FPB benchmark on all datasets and return results dict."""
    datasets = [load_phrasebank_csv(path, max_samples=args.max_samples) for path in args.datasets]

    print_dataset_metadata(datasets, title="BENCHMARK DATASETS")

    model = getattr(args, "model", "deberta")

    # Resolve optional tuning config for benchmark mode
    config = None
    if model in ("deberta", "roberta"):
        config = _resolve_config(getattr(args, "config", None))
        if config:
            logger.info(
                "Benchmark using tuned config: %s  template='%s'  labels=%s",
                config["name"],
                config["hypothesis_template"],
                list(config["label_map"].values()),
            )
            print(f"\n  Config: {config['name']}")
            print(f"  Template: {config['hypothesis_template']}")
            print(f"  Labels: {list(config['label_map'].values())}\n")
        else:
            logger.info("Benchmark using default config (financial_statement + simple labels)")

    predictor = _create_predictor(args)
    model_name = _model_display_name(model)

    report: dict[str, Any] = {
        "metadata": {
            "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "config_name": config["name"] if config else "default (argmax)",
            "hypothesis_template": config["hypothesis_template"] if config else "N/A",
            "candidate_labels": list(config["label_map"].values()) if config else ["positive", "negative", "neutral"],
            "datasets": [d.path for d in datasets],
            "max_samples": args.max_samples,
            "device": args.device,
            "decision_threshold_enabled": DECISION_THRESHOLD_ENABLED if model in ("deberta", "roberta") else False,
        },
        "results": {},
        "failures": [],
    }

    for dataset in datasets:
        logger.info("Evaluating %s on dataset=%s samples=%d", model_name, dataset.name, len(dataset.texts))
        started = time.time()
        try:
            if config and model in ("deberta", "roberta"):
                y_pred = predictor.predict_with_config(dataset.texts, config)
            else:
                y_pred = predictor.predict(dataset.texts)
            metrics = compute_metrics(dataset.labels, y_pred)
        except Exception as exc:
            logger.exception(
                "Evaluation failed for dataset=%s: %s",
                dataset.name,
                exc,
            )
            report["failures"].append({
                "dataset": dataset.path,
                "stage": "predict",
                "error": str(exc),
            })
            continue

        duration = round(time.time() - started, 2)
        report["results"][dataset.name] = {
            "dataset_path": dataset.path,
            "samples": len(dataset.texts),
            "label_distribution": dict(Counter(dataset.labels)),
            "duration_seconds": duration,
            "metrics": metrics,
        }

    return report


def print_report(report: dict[str, Any]) -> None:
    """Print a formatted FPB benchmark report to stdout."""
    metadata = report.get("metadata", {})
    model_name = metadata.get("model", "unknown")
    print("\n" + "=" * 90)
    print(f"  FPB BENCHMARK RESULTS  - {model_name}")
    print("=" * 90)
    config_name = metadata.get("config_name", "default")
    template = metadata.get("hypothesis_template", "N/A")
    labels = metadata.get("candidate_labels", [])
    dt_enabled = metadata.get("decision_threshold_enabled", False)
    print(f"  Config: {config_name}")
    print(f"  Template: {template}")
    print(f"  Labels: {labels}")
    print(f"  Decision Thresholds: {'ON' if dt_enabled else 'OFF'}")
    print("-" * 90)

    print(
        f"  {'Dataset':<50} {'N':>6} "
        f"{'Acc':>8} {'Prec':>8} {'Recall':>8} {'MacroF1':>8}"
    )
    print("-" * 90)

    any_row = False
    for dataset_name, result in report.get("results", {}).items():
        metrics = result["metrics"]
        print(
            f"  {dataset_name:<50} {result['samples']:>6} "
            f"{metrics['accuracy']:>8.4f} {metrics['macro_precision']:>8.4f} "
            f"{metrics['macro_recall']:>8.4f} {metrics['macro_f1']:>8.4f}"
        )
        any_row = True

    if not any_row:
        print("  No successful evaluation runs.")

    print("-" * 90)
    print("  Per-class F1 scores")
    print("-" * 90)
    for dataset_name, result in report.get("results", {}).items():
        f1_scores = result["metrics"]["f1_scores"]
        print(
            f"  {dataset_name:<50} "
            f"pos={f1_scores.get('positive', 0.0):.4f} "
            f"neg={f1_scores.get('negative', 0.0):.4f} "
            f"neu={f1_scores.get('neutral', 0.0):.4f}"
        )

    failures = report.get("failures", [])
    if failures:
        print("-" * 90)
        print(f"  Failures: {len(failures)}")
        for failure in failures:
            stage = failure.get("stage", "unknown")
            dataset = Path(failure["dataset"]).name if "dataset" in failure else "-"
            print(f"    stage={stage} dataset={dataset} error={failure['error']}")

    print("=" * 90 + "\n")


# Checkpointing

def save_checkpoint(data: dict[str, Any], path: Path = CHECKPOINT_PATH) -> None:
    """Atomically save checkpoint for crash recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def load_checkpoint(path: Path = CHECKPOINT_PATH) -> dict[str, Any] | None:
    """Load a tuning checkpoint from disk if it exists."""
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Stage 1: Coarse Screen

def run_coarse_screen(
    predictor: Predictor,
    configs: list[dict[str, Any]],
    dev_texts: list[str],
    dev_labels: list[str],
    n_folds: int,
    seed: int,
    top_k: int = 25,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Evaluate all configs on fold-0 of the coarse dataset.

    Returns (all_results_sorted, surviving_config_dicts).
    """
    folds = stratified_kfold(dev_labels, n_folds=n_folds, seed=seed)
    _, val_indices = folds[0]
    val_texts = [dev_texts[i] for i in val_indices]
    val_labels = [dev_labels[i] for i in val_indices]

    logger.info(
        "Stage 1: Coarse screen -- %d configs x fold-0 (%d val samples)",
        len(configs), len(val_texts),
    )

    results: list[dict[str, Any]] = []
    stage_start = time.time()

    for idx, config in enumerate(configs):
        logger.info("Coarse [%d/%d] %s", idx + 1, len(configs), config["name"])
        started = time.time()
        y_pred = predictor.predict_with_config(val_texts, config)
        duration = time.time() - started
        metrics = compute_metrics(val_labels, y_pred)

        results.append({
            "config_id": config["config_id"],
            "config_name": config["name"],
            "template_name": config["template_name"],
            "label_map_name": config["label_map_name"],
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "f1_scores": metrics["f1_scores"],
            "duration_seconds": round(duration, 2),
        })

    results.sort(key=lambda r: r["macro_f1"], reverse=True)

    stage_duration = time.time() - stage_start
    logger.info(
        "Coarse screen done in %.1fs -- top-%d selected from %d configs",
        stage_duration, min(top_k, len(configs)), len(configs),
    )

    survivor_ids = {r["config_id"] for r in results[:top_k]}
    surviving_configs = [c for c in configs if c["config_id"] in survivor_ids]

    return results, surviving_configs


# Stage 2: Full Cross-Validation

def run_cross_validation(
    predictor: Predictor,
    configs: list[dict[str, Any]],
    dataset_splits: list[dict[str, Any]],
    n_folds: int,
    seed: int,
    resume_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full K-fold CV for surviving configs across all datasets."""
    completed: set[tuple[str, str, int]] = set()
    fold_results: list[dict[str, Any]] = []

    if resume_data and resume_data.get("stage") == "cv":
        fold_results = list(resume_data.get("cv_fold_results", []))
        for entry in fold_results:
            completed.add((entry["config_id"], entry["dataset"], entry["fold"]))
        logger.info("Resumed from checkpoint: %d completed fold evaluations", len(completed))

    total_runs = len(configs) * n_folds * len(dataset_splits)
    done = len(completed)

    logger.info(
        "Stage 2: Full CV -- %d configs x %d folds x %d datasets = %d runs (%d done)",
        len(configs), n_folds, len(dataset_splits), total_runs, done,
    )

    stage_start = time.time()

    for config in configs:
        for ds in dataset_splits:
            folds = stratified_kfold(ds["dev_labels"], n_folds=n_folds, seed=seed)
            for fold_idx, (_, val_indices) in enumerate(folds):
                key = (config["config_id"], ds["name"], fold_idx)
                if key in completed:
                    continue

                done += 1
                val_texts = [ds["dev_texts"][i] for i in val_indices]
                val_labels = [ds["dev_labels"][i] for i in val_indices]

                logger.info(
                    "CV [%d/%d] config=%s ds=%s fold=%d/%d",
                    done, total_runs,
                    config["name"],
                    _short_ds_name(ds["name"]),
                    fold_idx + 1, n_folds,
                )

                started = time.time()
                y_pred = predictor.predict_with_config(val_texts, config)
                duration = time.time() - started
                metrics = compute_metrics(val_labels, y_pred)

                fold_results.append({
                    "config_id": config["config_id"],
                    "config_name": config["name"],
                    "template_name": config["template_name"],
                    "label_map_name": config["label_map_name"],
                    "dataset": ds["name"],
                    "fold": fold_idx,
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "f1_scores": metrics["f1_scores"],
                    "duration_seconds": round(duration, 2),
                })

                if done % 10 == 0:
                    save_checkpoint({"stage": "cv", "cv_fold_results": fold_results})

    stage_duration = time.time() - stage_start
    logger.info("Stage 2 done in %.1fs", stage_duration)

    # Final checkpoint save
    save_checkpoint({"stage": "cv", "cv_fold_results": fold_results})

    summary = _aggregate_cv_results(fold_results, configs, dataset_splits)

    return {"fold_results": fold_results, "summary": summary}


def _aggregate_cv_results(
    fold_results: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    dataset_splits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate fold-level results into mean +/- std per config."""
    dataset_names = [ds["name"] for ds in dataset_splits]

    by_config: dict[str, list[dict]] = {}
    for entry in fold_results:
        by_config.setdefault(entry["config_id"], []).append(entry)

    summaries: list[dict[str, Any]] = []
    for config in configs:
        cid = config["config_id"]
        entries = by_config.get(cid, [])
        if not entries:
            continue

        per_dataset: dict[str, dict[str, float]] = {}
        all_f1s: list[float] = []

        for ds_name in dataset_names:
            ds_entries = [e for e in entries if e["dataset"] == ds_name]
            f1s = [e["macro_f1"] for e in ds_entries]
            accs = [e["accuracy"] for e in ds_entries]

            if f1s:
                mean_f1 = sum(f1s) / len(f1s)
                std_f1 = (sum((x - mean_f1) ** 2 for x in f1s) / len(f1s)) ** 0.5
                mean_acc = sum(accs) / len(accs)
                per_dataset[ds_name] = {
                    "mean_f1": round(mean_f1, 4),
                    "std_f1": round(std_f1, 4),
                    "mean_acc": round(mean_acc, 4),
                    "n_folds": len(f1s),
                }
                all_f1s.extend(f1s)

        if all_f1s:
            overall_mean = sum(all_f1s) / len(all_f1s)
            overall_std = (sum((x - overall_mean) ** 2 for x in all_f1s) / len(all_f1s)) ** 0.5
        else:
            overall_mean = 0.0
            overall_std = 0.0

        summaries.append({
            "config_id": cid,
            "config_name": config["name"],
            "template_name": config["template_name"],
            "label_map_name": config["label_map_name"],
            "overall_mean_f1": round(overall_mean, 4),
            "overall_std_f1": round(overall_std, 4),
            "per_dataset": per_dataset,
            "fold_f1_values": all_f1s,
        })

    summaries.sort(key=lambda s: s["overall_mean_f1"], reverse=True)
    return summaries


# Stage 3: Final Test Evaluation

def run_final_test(
    predictor: Predictor,
    best_config: dict[str, Any],
    dataset_splits: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate the best config on held-out test sets (never seen before)."""
    logger.info("Stage 3: Final test -- config=%s", best_config["name"])

    test_results: dict[str, Any] = {}
    for ds in dataset_splits:
        started = time.time()
        y_pred = predictor.predict_with_config(ds["test_texts"], best_config)
        duration = time.time() - started
        metrics = compute_metrics(ds["test_labels"], y_pred)

        test_results[ds["name"]] = {
            "macro_f1": metrics["macro_f1"],
            "accuracy": metrics["accuracy"],
            "f1_scores": metrics["f1_scores"],
            "per_class": metrics["per_class"],
            "total": metrics["total"],
            "correct": metrics["correct"],
            "duration_seconds": round(duration, 2),
        }

        logger.info(
            "Test %s: F1=%.4f Acc=%.4f",
            _short_ds_name(ds["name"]), metrics["macro_f1"], metrics["accuracy"],
        )

    return test_results


# Statistical significance

def paired_t_test(
    scores_a: list[float],
    scores_b: list[float],
) -> dict[str, Any]:
    """Two-sided paired t-test on fold-level F1 scores."""
    n = len(scores_a)
    if n != len(scores_b) or n < 2:
        return {"t_stat": 0.0, "p_value": 1.0, "significant_005": False, "n": n}

    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    mean_d = sum(diffs) / n
    var_d = sum((d - mean_d) ** 2 for d in diffs) / (n - 1)
    se = (var_d / n) ** 0.5 if var_d > 0 else 1e-10
    t_stat = mean_d / se
    df = n - 1

    try:
        from scipy.stats import t as t_dist
        p_value = float(t_dist.sf(abs(t_stat), df) * 2)
    except ImportError:
        # Rough approximation without scipy
        abs_t = abs(t_stat)
        if abs_t > 4.0:
            p_value = 0.001
        elif abs_t > 3.0:
            p_value = 0.01
        elif abs_t > 2.5:
            p_value = 0.02
        elif abs_t > 2.0:
            p_value = 0.05
        elif abs_t > 1.5:
            p_value = 0.15
        elif abs_t > 1.0:
            p_value = 0.30
        else:
            p_value = 0.50

    return {
        "t_stat": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant_005": p_value < 0.05,
        "mean_diff": round(mean_d, 4),
        "n_pairs": n,
    }


# Effect analysis

def _compute_effect_analysis(fold_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute average F1 per template and per label_map across all folds."""
    by_template: dict[str, list[float]] = {}
    by_label_map: dict[str, list[float]] = {}

    for entry in fold_results:
        by_template.setdefault(entry["template_name"], []).append(entry["macro_f1"])
        by_label_map.setdefault(entry["label_map_name"], []).append(entry["macro_f1"])

    template_effects = {
        name: round(sum(vals) / len(vals), 4)
        for name, vals in sorted(
            by_template.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True,
        )
    }

    label_map_effects = {
        name: round(sum(vals) / len(vals), 4)
        for name, vals in sorted(
            by_label_map.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True,
        )
    }

    return {
        "by_template": template_effects,
        "by_label_map": label_map_effects,
    }


# Label tuning orchestrator

def run_label_tuning(args: argparse.Namespace) -> dict[str, Any]:
    """Three-stage label tuning: coarse screen -> K-fold CV -> final test."""
    configs = generate_configs()
    logger.info("Generated %d configs (%d templates x %d label maps)",
                len(configs), len(HYPOTHESIS_TEMPLATES), len(LABEL_MAPS))

    # Load all datasets, split into dev/test (80/20 holdout)
    dataset_splits: list[dict[str, Any]] = []
    for path in args.tune_datasets:
        ds = load_phrasebank_csv(path, max_samples=args.max_samples)
        dev_texts, dev_labels, test_texts, test_labels = stratified_split(
            ds.texts, ds.labels,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
        dataset_splits.append({
            "name": ds.name,
            "path": path,
            "total": len(ds.texts),
            "dev_texts": dev_texts,
            "dev_labels": dev_labels,
            "test_texts": test_texts,
            "test_labels": test_labels,
        })
        print_holdout_metadata(
            dev_labels, test_labels, ds.name,
            args.test_ratio, args.n_folds, args.seed,
        )

    predictor = _create_predictor(args)

    # Stage 1: Coarse screen on 50agree dev set
    coarse_ds = dataset_splits[min(COARSE_DATASET_IDX, len(dataset_splits) - 1)]
    coarse_results, surviving_configs = run_coarse_screen(
        predictor, configs,
        coarse_ds["dev_texts"], coarse_ds["dev_labels"],
        n_folds=args.n_folds, seed=args.seed,
        top_k=args.coarse_top_k,
    )

    print_coarse_report(coarse_results, args.coarse_top_k)

    if args.stage == "coarse":
        return {
            "metadata": _build_metadata(args, dataset_splits, configs, surviving_configs),
            "coarse_screen": coarse_results,
        }

    # Stage 2: Full K-fold CV on surviving configs
    resume_data = load_checkpoint() if args.resume else None
    cv_data = run_cross_validation(
        predictor, surviving_configs, dataset_splits,
        n_folds=args.n_folds, seed=args.seed,
        resume_data=resume_data,
    )

    cv_summary = cv_data["summary"]
    print_cv_report(cv_summary, dataset_splits)

    # Stage 3: Final test with best config
    best_config_id = cv_summary[0]["config_id"]
    best_config = next(c for c in configs if c["config_id"] == best_config_id)

    test_results = run_final_test(predictor, best_config, dataset_splits)

    # Statistical significance: top-1 vs top-2
    significance: dict[str, Any] = {}
    if len(cv_summary) >= 2:
        significance = paired_t_test(
            cv_summary[0]["fold_f1_values"],
            cv_summary[1]["fold_f1_values"],
        )
        significance["config_a"] = cv_summary[0]["config_name"]
        significance["config_b"] = cv_summary[1]["config_name"]

    # Effect analysis
    effect_analysis = _compute_effect_analysis(cv_data["fold_results"])

    print_test_report(test_results, best_config, cv_summary[0])
    print_effect_analysis(effect_analysis)
    if significance:
        print_significance(significance)

    # Clean up checkpoint
    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    report: dict[str, Any] = {
        "metadata": _build_metadata(args, dataset_splits, configs, surviving_configs),
        "coarse_screen": coarse_results,
        "cv_summary": [
            {k: v for k, v in s.items() if k != "fold_f1_values"}
            for s in cv_summary
        ],
        "best_config": {
            "config_id": best_config["config_id"],
            "config_name": best_config["name"],
            "template_name": best_config["template_name"],
            "label_map_name": best_config["label_map_name"],
            "hypothesis_template": best_config["hypothesis_template"],
            "label_map": best_config["label_map"],
            "cv_mean_f1": cv_summary[0]["overall_mean_f1"],
            "cv_std_f1": cv_summary[0]["overall_std_f1"],
        },
        "test_results": test_results,
        "significance": significance,
        "effect_analysis": effect_analysis,
    }

    return report


def _build_metadata(
    args: argparse.Namespace,
    dataset_splits: list[dict[str, Any]],
    configs: list[dict[str, Any]],
    surviving_configs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build the metadata dict saved alongside benchmark results."""
    return {
        "run_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": MODEL_NAME,
        "datasets": [ds["path"] for ds in dataset_splits],
        "dataset_names": [ds["name"] for ds in dataset_splits],
        "total_configs": len(configs),
        "coarse_survivors": len(surviving_configs),
        "n_folds": args.n_folds,
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "splits": {
            ds["name"]: {
                "total": ds["total"],
                "dev_samples": len(ds["dev_texts"]),
                "test_samples": len(ds["test_texts"]),
            }
            for ds in dataset_splits
        },
    }


# Reporting

def print_coarse_report(
    coarse_results: list[dict[str, Any]],
    top_k: int,
) -> None:
    """Print coarse screening results."""
    print("\n" + "=" * 100)
    print("  STAGE 1: COARSE SCREEN (fold-0 of 50agree)")
    print("=" * 100)
    print(f"  Total configs: {len(coarse_results)}  |  Survivors: top-{top_k}")
    print()
    print(
        f"  {'Rank':>4}  {'Config':<40} {'F1':>7} {'Acc':>7} "
        f"{'Template':<20} {'LabelMap':<12}"
    )
    print("-" * 100)

    for i, r in enumerate(coarse_results):
        marker = " *" if i < top_k else ""
        print(
            f"  {i + 1:>4}  {r['config_name']:<40} "
            f"{r['macro_f1']:>7.4f} {r['accuracy']:>7.4f} "
            f"{r['template_name']:<20} {r['label_map_name']:<12}{marker}"
        )

    if len(coarse_results) > top_k:
        cutoff = coarse_results[top_k - 1]["macro_f1"]
        eliminated = coarse_results[top_k]["macro_f1"]
        print(f"\n  Cutoff F1: {cutoff:.4f}  |  First eliminated: {eliminated:.4f}")

    print("=" * 100 + "\n")


def print_cv_report(
    cv_summary: list[dict[str, Any]],
    dataset_splits: list[dict[str, Any]],
) -> None:
    """Print K-fold CV results."""
    dataset_names = [ds["name"] for ds in dataset_splits]
    short_names = [_short_ds_name(n) for n in dataset_names]

    print("\n" + "=" * 110)
    print("  STAGE 2: K-FOLD CROSS-VALIDATION RESULTS")
    print("=" * 110)

    # Dynamic header
    ds_headers = "".join(f" {sn:>10}" for sn in short_names)
    print(f"  {'Rank':>4}  {'Config':<40} {'Mean F1':>8} {'Std':>6}{ds_headers}")
    print("-" * 110)

    for i, s in enumerate(cv_summary):
        ds_cols = ""
        for ds_name in dataset_names:
            pd = s.get("per_dataset", {}).get(ds_name, {})
            mean = pd.get("mean_f1", 0)
            std = pd.get("std_f1", 0)
            ds_cols += f" {mean:>.4f}+{std:<.2f}"

        marker = " <-- best" if i == 0 else ""
        print(
            f"  {i + 1:>4}  {s['config_name']:<40} "
            f"{s['overall_mean_f1']:>8.4f} {s['overall_std_f1']:>6.4f}{ds_cols}{marker}"
        )

    print("=" * 110 + "\n")


def print_test_report(
    test_results: dict[str, Any],
    best_config: dict[str, Any],
    cv_top: dict[str, Any],
) -> None:
    """Print final held-out test results."""
    print("\n" + "=" * 100)
    print("  STAGE 3: FINAL TEST RESULTS (held-out 20%)")
    print("=" * 100)
    print(f"  Best config   : {best_config['name']}")
    print(f"  Template      : {best_config['hypothesis_template']}")
    lm = best_config["label_map"]
    print(f"  Labels        : pos=\"{lm['positive']}\"  neg=\"{lm['negative']}\"  neu=\"{lm['neutral']}\"")
    print(f"  CV Mean F1    : {cv_top['overall_mean_f1']:.4f} +/- {cv_top['overall_std_f1']:.4f}")

    print()
    print(f"  {'Dataset':<15} {'Test F1':>9} {'Test Acc':>9} {'F1-pos':>8} {'F1-neg':>8} {'F1-neu':>8}  {'N':>6}")
    print("-" * 100)

    all_f1s: list[float] = []
    all_accs: list[float] = []

    for ds_name, metrics in test_results.items():
        sn = _short_ds_name(ds_name)
        f1s = metrics.get("f1_scores", {})
        print(
            f"  {sn:<15} {metrics['macro_f1']:>9.4f} {metrics['accuracy']:>9.4f} "
            f"{f1s.get('positive', 0):>8.4f} {f1s.get('negative', 0):>8.4f} "
            f"{f1s.get('neutral', 0):>8.4f}  {metrics['total']:>6}"
        )
        all_f1s.append(metrics["macro_f1"])
        all_accs.append(metrics["accuracy"])

    if all_f1s:
        avg_f1 = sum(all_f1s) / len(all_f1s)
        avg_acc = sum(all_accs) / len(all_accs)
        print("-" * 100)
        print(f"  {'AVERAGE':<15} {avg_f1:>9.4f} {avg_acc:>9.4f}")

    # CV vs Test gap
    cv_f1 = cv_top["overall_mean_f1"]
    if all_f1s:
        test_avg_f1 = sum(all_f1s) / len(all_f1s)
        gap = cv_f1 - test_avg_f1
        print(f"\n  CV-Test gap (overfit indicator): {gap:+.4f}")

    print("=" * 100 + "\n")


def print_effect_analysis(effect_analysis: dict[str, Any]) -> None:
    """Print template and label map effect analysis."""
    print("\n" + "=" * 80)
    print("  EFFECT ANALYSIS")
    print("=" * 80)

    print("\n  Template Effect (avg F1 across all label maps, sorted):")
    print("-" * 80)
    for name, f1 in effect_analysis.get("by_template", {}).items():
        bar = _bar(f1, width=20)
        print(f"    {name:<30} {f1:.4f}  |{bar}|")

    print("\n  Label Map Effect (avg F1 across all templates, sorted):")
    print("-" * 80)
    for name, f1 in effect_analysis.get("by_label_map", {}).items():
        bar = _bar(f1, width=20)
        print(f"    {name:<30} {f1:.4f}  |{bar}|")

    print("=" * 80 + "\n")


def print_significance(significance: dict[str, Any]) -> None:
    """Print statistical significance test result."""
    print("\n" + "-" * 80)
    print("  STATISTICAL SIGNIFICANCE (paired t-test: top-1 vs top-2)")
    print("-" * 80)
    print(f"  Config A (top-1): {significance.get('config_a', '?')}")
    print(f"  Config B (top-2): {significance.get('config_b', '?')}")
    print(f"  t-statistic     : {significance.get('t_stat', 0):.4f}")
    print(f"  p-value         : {significance.get('p_value', 1):.6f}")
    print(f"  Mean diff (A-B) : {significance.get('mean_diff', 0):+.4f}")
    print(f"  N pairs         : {significance.get('n_pairs', 0)}")

    if significance.get("significant_005"):
        print("  Result          : SIGNIFICANT at alpha=0.05")
    else:
        print("  Result          : NOT significant at alpha=0.05")

    print("-" * 80 + "\n")


# CLI

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for benchmark and tune modes."""
    parser = argparse.ArgumentParser(
        description="Benchmark zero-shot and baseline sentiment models on Financial PhraseBank datasets.",
    )

    # Mode
    parser.add_argument(
        "--mode",
        choices=["benchmark", "tune"],
        default="benchmark",
        help=(
            "'benchmark' (default): evaluate the selected model on full datasets. "
            "'tune': find the best label config with K-fold CV.",
        ),
    )

    # Shared options
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Batch size for NLI model inference.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for DeBERTa inference.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=SENTIMENT_DEVICE,
        help="Transformers device index. Use -1 for CPU.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_OUTPUT),
        help="Output JSON file path.",
    )

    # Model selection
    parser.add_argument(
        "--model",
        choices=["deberta", "roberta", "ollama-llama3", "ollama-mistral", "finbert", "fingpt"],
        default="deberta",
        help="Which model to benchmark (default: deberta).",
    )

    # Benchmark-mode options
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="One or more CSV dataset paths (default: the three Financial PhraseBank splits).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Name of a tuning config to use for benchmark mode "
            "(e.g. 'about_outlook__outlook'). Format: <template_name>__<label_map_name>. "
            "If not set, uses the hardcoded default (financial_statement + simple labels)."
        ),
    )

    # Tune-mode options
    parser.add_argument(
        "--tune-datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="CSV dataset(s) for label tuning (default: all three FPB agreement splits).",
    )
    parser.add_argument(
        "--stage",
        choices=["coarse", "full"],
        default="full",
        help="'coarse': only run Stage 1 (screen all configs). 'full' (default): all 3 stages.",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=5,
        help="Number of CV folds (default: 5).",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="Fraction held out for final test set (default: 0.20).",
    )
    parser.add_argument(
        "--coarse-top-k",
        type=int,
        default=25,
        help="Number of configs to keep after coarse screen (default: 25).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (Stage 2 crash recovery).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splits (default: 42).",
    )

    return parser.parse_args()


def main() -> None:
    """Entry point for benchmark and label-tuning modes."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    args = parse_args()

    # Derive model tag from CLI --model (overrides config default)
    model_tag = getattr(args, "model", _MODEL_TAG)

    if args.mode == "tune":
        report = run_label_tuning(args)

        output_path = FPB_PATH / f"phrasebank_label_tuning_{model_tag}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Tuning results written to %s", output_path)
    else:
        report = run_benchmark(args)

        # Use CLI --output if explicitly provided, otherwise derive from model tag
        if args.output != str(DEFAULT_OUTPUT):
            output_path = Path(args.output)
        else:
            output_path = FPB_PATH / f"phrasebank_model_benchmark_{model_tag}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        logger.info("Benchmark results written to %s", output_path)
        print_report(report)


if __name__ == "__main__":
    main()
