"""Central configuration for the sentiment analysis pipeline."""

from __future__ import annotations

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "fetch.logs"
DATA_PATH = PROJECT_ROOT / "data"
TEMP_PATH = DATA_PATH / "temp"
EVAL_PATH = DATA_PATH / "evaluation"
ARTICLE_CACHE_PATH = DATA_PATH / "articles" / "articles_with_noise_reduction_deberta"

# Sub-directories under evaluation/
EVAL_RESULTS_PATH       = EVAL_PATH / "evaluation_results"
CORRELATION_PATH        = EVAL_PATH / "correlation_results"
FPB_PATH                = EVAL_PATH / "FPB"
DATASET_PATH            = EVAL_PATH / "pipeline_evaluation_dataset"
THRESHOLD_PATH          = EVAL_PATH / "threshold_calibration"
FPB_DATASETS_PATH       = DATA_PATH / "financial_phrasebank_datasets"

# Pipeline intermediary files
JSON_PATH      = TEMP_PATH / "articles.json"
PRED_JSON_PATH = TEMP_PATH / "predictions.json"
PREDS_PATH     = TEMP_PATH / "predictions"
FINBERT_PREDS  = PREDS_PATH / "finbert_result.json"
FINGPT_PREDS   = PREDS_PATH / "fingpt_result.json"
ZEROSHOT_PREDS = PREDS_PATH / "zeroshot_result.json"
OLLAMA_PREDS   = PREDS_PATH / "ollama_result.json"

TEMP_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT_LIMIT = 30

# ── Api & Network ────────────────────────────────────────────────────────────
FINNHUB_API_KEY = "d5rvt19r01qq2th0b8sgd5rvt19r01qq2th0b8t0"

NOISE_REDUCTION_MODEL = "microsoft/deberta-large-mnli" 

IMPACT_HORIZON_MODEL = "microsoft/deberta-large-mnli"
IMPACT_HORIZON_DEVICE = 0

SENTIMENT_DEVICE = 0 

# ── Model Selection ───────────────────────────────────────────────────────────

# # finbert config
SENTIMENT_MODEL = "ProsusAI/finbert"
SENTIMENT_MAX_LENGTH = 512  

#fingpt config
# SENTIMENT_MODEL = "fingpt"
# FINGPT_BASE_MODEL = "NousResearch/Llama-2-13b-hf"
# FINGPT_LORA_MODEL = "FinGPT/fingpt-sentiment_llama2-13b_lora"
# FINGPT_LOAD_IN_8BIT = True

# zero-shot config


# ollama LLM config (Llama 3.1 8B or Mistral 7B)
# SENTIMENT_MODEL = "ollama-llama3"
OLLAMA_SENTIMENT_MODEL = "llama3.1:8b"
# SENTIMENT_MODEL = "ollama-mistral"
# OLLAMA_SENTIMENT_MODEL = "mistral:7b"

SENTIMENT_MODEL = "zero-shot"
# MODEL_NAME = "microsoft/deberta-large-mnli" # microsoft deberta large mnli
MODEL_NAME = "roberta-large-mnli" # facebookAI roberta mnli

# ── Xai Configuration ────────────────────────────────────────────────────────
XAI_ENABLED                    = True
XAI_EXPLANATIONS_PATH          = PROJECT_ROOT / "data" / "xai_explanations"
XAI_OUTPUT_PATH                = XAI_EXPLANATIONS_PATH / "xai_result.json"
XAI_SUMMARY_PATH               = XAI_EXPLANATIONS_PATH / "xai_summary.txt"
XAI_EXPLANATIONS_PATH.mkdir(parents=True, exist_ok=True)

XAI_LIME_TOP_N                 = 5
XAI_LIME_NUM_SAMPLES           = 300
XAI_LIME_NUM_FEATURES          = 20

XAI_THIN_EVIDENCE_THRESHOLD    = 5
XAI_CONCENTRATION_THRESHOLD    = 0.4
XAI_MARGIN_THRESHOLD           = 0.15

XAI_LLAMA_MODEL                = "llama3.2:3b"
XAI_LLAMA_TEMPERATURE          = 0.1
XAI_LLAMA_MAX_TOKENS           = 200
XAI_LLAMA_ENABLED              = True

# Evidence quality — source diversity & timing
XAI_SOURCE_CONCENTRATION_THRESHOLD = 0.60   # flag if top domain > 60% of articles
XAI_MIN_UNIQUE_SOURCES             = 2      # flag if fewer unique domains

# ── Aggregation & Post-Processing ────────────────────────────────────────────
COVERAGE_COUNT_BOOST           = True    # boost multi-source articles via log2(1 + coverage)
HEADLINE_ONLY_WEIGHT           = 0.5     # discount for headline-only articles (no body content)

# Decision thresholds (tuned on tune set via grid search, macro F1)
# Applied before dynamic abstention margin.
# positive must exceed tau_pos, negative must exceed tau_neg, else neutral.
DECISION_THRESHOLD_ENABLED     = True
DECISION_THRESHOLD_POS         = 0.56
DECISION_THRESHOLD_NEG         = 0.26

# Dynamic abstention margin (entropy-adaptive safety net)
DYNAMIC_MARGIN_ENABLED         = False

# ── Market Timezone Alignment ────────────────────────────────────────────────
MARKET_TIMEZONE    = "America/New_York"      # NYSE / NASDAQ timezone
MARKET_CLOSE_HOUR  = 16                      # 4:00 PM ET
