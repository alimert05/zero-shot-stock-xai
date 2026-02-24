import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "fetch.logs"
DATA_PATH = PROJECT_ROOT / "data"
TEMP_PATH = DATA_PATH / "temp"
PRED_PATH = DATA_PATH / "test_results"
PRED_JSON_PATH = PRED_PATH / "predictions.json"
JSON_PATH = PROJECT_ROOT / "data" / "temp" / "articles.json"
FINBERT_PREDS = PROJECT_ROOT / "data" / "predictions" / "finbert_result.json"
FINGPT_PREDS = PROJECT_ROOT / "data" / "predictions" / "fingpt_result.json"
ZEROSHOT_PREDS = PROJECT_ROOT / "data" / "predictions" / "zeroshot_result.json"

TEMP_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT_LIMIT = 30

# API Configuration
FINNHUB_API_KEY = "d5rvt19r01qq2th0b8sgd5rvt19r01qq2th0b8t0"

NOISE_REDUCTION_MODEL = "roberta-large-mnli" 

IMPACT_HORIZON_MODEL = "roberta-large-mnli" 
IMPACT_HORIZON_DEVICE = 0

SENTIMENT_DEVICE = 0 

# finbert config
# SENTIMENT_MODEL = "ProsusAI/finbert"
# SENTIMENT_MAX_LENGTH = 512  

#fingpt config
# SENTIMENT_MODEL = "fingpt"
# FINGPT_BASE_MODEL = "NousResearch/Llama-2-13b-hf"
# FINGPT_LORA_MODEL = "FinGPT/fingpt-sentiment_llama2-13b_lora"
# FINGPT_LOAD_IN_8BIT = True

# zero-shot config
SENTIMENT_MODEL = "zero-shot"
# MODEL_NAME = "facebook/bart-large-mnli" # facebook bart mnli
MODEL_NAME = "roberta-large-mnli" # facebookAI roberta mnli
# MODEL_NAME = "microsoft/deberta-large-mnli/" # microsoft deberta large mnli

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

# Trading action policy
XAI_ACTION_MIN_CONFIDENCE      = 0.55
XAI_ACTION_MIN_MARGIN          = 0.05
NEUTRAL_THRESHOLD              = 0.003      # ±0.3% close-to-close return band

# Market timezone alignment
MARKET_TIMEZONE    = "America/New_York"      # NYSE / NASDAQ timezone
MARKET_CLOSE_HOUR  = 16                      # 4:00 PM ET
