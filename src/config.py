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

TEMP_PATH.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

REQUEST_TIMEOUT_LIMIT = 30

# API Configuration
BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

THEMES = [
    "ECON_STOCKMARKET",
    "ECON_CENTRALBANK",
    "ECON_INTEREST_RATES",
    # "ECON_INFLATION",
    # "ECON_WORLDCURRENCIES",
    # "ECON_CURRENCY_EXCHANGE_RATE",
    "EPU_ECONOMY",
    # "EPU_POLICY",
    "EPU_CATS_FINANCIAL_REGULATION",
    # "EPU_CATS_MONETARY_POLICY",
    # "ECON_TAXATION",
    # "ECON_IPO",
]

# Impact Horizon Configuration
IMPACT_HORIZON_ENABLED = True  # Set to False to disable impact horizon classification
IMPACT_HORIZON_MODEL = "MoritzLaurer/deberta-v3-base-zeroshot-v2.0"
IMPACT_HORIZON_DEVICE = -1  # -1 for CPU, 0 for GPU
 
# Weight combination method: "weighted_avg", "multiplicative", "geometric"
WEIGHT_COMBINE_METHOD = "weighted_avg"
 
# Importance weights for weighted_avg method (should sum to 1.0)
RECENCY_IMPORTANCE = 0.4
HORIZON_IMPORTANCE = 0.6

