from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_PATH = LOG_DIR / "fetch.logs"

LOG_DIR.mkdir(parents=True, exist_ok=True)

# API Configuration
BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

