import pytest
import sys
from pathlib import Path

# Add project root to sys.path so we can import src
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

@pytest.fixture
def mock_tickers():
    return ["AAPL", "MSFT"]

@pytest.fixture
def mock_bronze_path(tmp_path):
    d = tmp_path / "data" / "bronze"
    d.mkdir(parents=True)
    return d
