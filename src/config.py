from pathlib import Path
import os

# Base Directories
BASE_DIR = Path(__file__).resolve().parent.parent # market_mind root
DATA_DIR = BASE_DIR / "data"

DATA_PATHS = {
    "bronze": DATA_DIR / "bronze",
    "silver": DATA_DIR / "silver",
    "gold": DATA_DIR / "gold"
}

# Ensure directories exist
for path in DATA_PATHS.values():
    path.mkdir(parents=True, exist_ok=True)

# Market Parameters
MARKET_PARAMS = {
    "RISK_FREE_RATE": 0.04,
    "TRADING_DAYS": 252,
    "HISTORY_PERIOD": "2y"
}

# Model Parameters
MODEL_PARAMS = {
    "GNN_HIDDEN_DIM": 16,
    "SEQ_LEN": 10,
    "TRAIN_SPLIT": 0.8
}

# Tickers Universe (V1)
TICKERS = [
    # Tech
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA", "AMD", "INTC", "CRM", "ADBE", "CSCO",
    # Finance
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP", "BLK",
    # Healthcare
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY", "TMO", "DHR",
    # Consumer
    "PG", "KO", "PEP", "COST", "WMT", "TGT", "HD", "MCD", "NKE", "SBUX",
    # Communication/Media
    "DIS", "NFLX", "CMCSA", "TMUS", "VZ", "T",
    # Industrial/Energy
    "XOM", "CVX", "BA", "CAT", "GE", "HON", "UPS", "UNP"
]
