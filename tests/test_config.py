import sys
from pathlib import Path
from src.config import DATA_PATHS, MARKET_PARAMS, TICKERS

def test_config_structure():
    assert "bronze" in DATA_PATHS
    assert "silver" in DATA_PATHS
    assert "gold" in DATA_PATHS
    
    assert "RISK_FREE_RATE" in MARKET_PARAMS
    assert "TRADING_DAYS" in MARKET_PARAMS
    
    assert isinstance(TICKERS, list)
    assert len(TICKERS) > 0

def test_paths_exist():
    # Config creates paths on import, so they should exist
    for key, path in DATA_PATHS.items():
        assert path.exists()
        assert path.is_dir()
