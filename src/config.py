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
BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

WEIGHT_COMBINE_METHOD = "weighted_avg"

NOISE_REDUCTION_ENABLED = True
IMPACT_HORIZON_ENABLED = True
IMPACT_HORIZON_MODEL = "roberta-large-mnli"
IMPACT_HORIZON_DEVICE = 0

RECENCY_IMPORTANCE = 0.4
HORIZON_IMPORTANCE = 0.6

SENTIMENT_DEVICE = 0 

# finbert config
# SENTIMENT_MODEL = "ProsusAI/finbert"
# SENTIMENT_MAX_LENGTH = 512  

#fingpt config
SENTIMENT_MODEL = "fingpt"
FINGPT_BASE_MODEL = "NousResearch/Llama-2-13b-hf"
FINGPT_LORA_MODEL = "FinGPT/fingpt-sentiment_llama2-13b_lora"
FINGPT_LOAD_IN_8BIT = True

# zero-shot config
# SENTIMENT_MODEL = "zero-shot"
MODEL_NAME = "facebook/bart-large-mnli" # facebook bart mnli
# MODEL_NAME = "roberta-large-mnli" # facebookAI roberta mnli
# MODEL_NAME = "microsoft/deberta-large-mnli" # microsoft deberta large mnli

THEMES = [
    "ECON_STOCKMARKET",
    "ECON_CENTRALBANK",
    "ECON_INTEREST_RATES",
    "EPU_CATS_FINANCIAL_REGULATION",
    "EPU_ECONOMY",

    # "ECON_INFLATION",
    # "ECON_WORLDCURRENCIES",
    # "ECON_CURRENCY_EXCHANGE_RATE",
    # "EPU_POLICY",
    # "EPU_CATS_MONETARY_POLICY",
    # "ECON_TAXATION",
    # "ECON_IPO",
]