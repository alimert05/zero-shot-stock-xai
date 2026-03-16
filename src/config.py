import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "fetch.logs"
DATA_PATH = PROJECT_ROOT / "data"
TEMP_PATH = DATA_PATH / "temp"
PRED_PATH = DATA_PATH / "test_results"
ARTICLE_CACHE_PATH = DATA_PATH / "articles_with_noise_reduction_deberta"
PRED_JSON_PATH = PRED_PATH / "predictions.json"
JSON_PATH = PROJECT_ROOT / "data" / "temp" / "articles.json"
FINBERT_PREDS = PROJECT_ROOT / "data" / "predictions" / "finbert_result.json"
FINGPT_PREDS = PROJECT_ROOT / "data" / "predictions" / "fingpt_result.json"
ZEROSHOT_PREDS = PROJECT_ROOT / "data" / "predictions" / "zeroshot_result.json"
OLLAMA_PREDS = PROJECT_ROOT / "data" / "predictions" / "ollama_result.json"

TEMP_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT_LIMIT = 30

# API Configuration
FINNHUB_API_KEY = "d5rvt19r01qq2th0b8sgd5rvt19r01qq2th0b8t0"

NOISE_REDUCTION_MODEL = "microsoft/deberta-large-mnli" 

IMPACT_HORIZON_MODEL = "microsoft/deberta-large-mnli"
IMPACT_HORIZON_DEVICE = 0

SENTIMENT_DEVICE = 0 

# # finbert config
# SENTIMENT_MODEL = "ProsusAI/finbert"
# SENTIMENT_MAX_LENGTH = 512  

#fingpt config
# SENTIMENT_MODEL = "fingpt"
FINGPT_BASE_MODEL = "NousResearch/Llama-2-13b-hf"
FINGPT_LORA_MODEL = "FinGPT/fingpt-sentiment_llama2-13b_lora"
FINGPT_LOAD_IN_8BIT = True

# zero-shot config


# ollama LLM config (Llama 3.1 8B or Mistral 7B)
# SENTIMENT_MODEL = "ollama-llama3"
# OLLAMA_SENTIMENT_MODEL = "llama3.1:8b"
# SENTIMENT_MODEL = "ollama-mistral"
OLLAMA_SENTIMENT_MODEL = "mistral:7b"

SENTIMENT_MODEL = "zero-shot"
MODEL_NAME = "microsoft/deberta-large-mnli" # microsoft deberta large mnli
# MODEL_NAME = "roberta-large-mnli" # facebookAI roberta mnli

# XAI Configuration
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
XAI_LOW_CONFIDENCE_THRESHOLD   = 0.45

XAI_LLAMA_MODEL                = "llama3.2:3b"
XAI_LLAMA_TEMPERATURE          = 0.1
XAI_LLAMA_MAX_TOKENS           = 200
XAI_LLAMA_ENABLED              = True

# Reliability — source diversity & timing
XAI_SOURCE_CONCENTRATION_THRESHOLD = 0.60   # flag if top domain > 60% of articles
XAI_MIN_UNIQUE_SOURCES             = 2      # flag if fewer unique domains

# Prior debiasing (Bayesian correction for zero-shot label bias)
PRIOR_DEBIASING_ENABLED        = False
PRIOR_DEBIASING_ALPHA          = 1.0       # damping: 0=off, 1=full, 0.5=half-strength

# Enhanced article weighting
SENTIMENT_CONFIDENCE_WEIGHTING = False   # disabled: scores already encode uncertainty; double-counts
COVERAGE_COUNT_BOOST           = True    # boost multi-source articles via log2(1 + coverage)
RELEVANCE_RATIO_WEIGHTING      = False   # disabled: noise reducer already filters; double-penalty
HEADLINE_ONLY_WEIGHT           = 0.5     # discount for headline-only articles (no body content)

# Decision thresholds (tuned on tune set via grid search, macro F1)
# Applied before dynamic abstention margin.
# positive must exceed tau_pos, negative must exceed tau_neg, else neutral.
DECISION_THRESHOLD_ENABLED     = False
DECISION_THRESHOLD_POS         = 0.56
DECISION_THRESHOLD_NEG         = 0.42

# Dynamic abstention margin (entropy-adaptive safety net)
DYNAMIC_MARGIN_ENABLED         = True

NEUTRAL_THRESHOLD              = 0.003     # ±0.3% close-to-close return band ?

# Market timezone alignment
MARKET_TIMEZONE    = "America/New_York"      # NYSE / NASDAQ timezone
MARKET_CLOSE_HOUR  = 16                      # 4:00 PM ET
