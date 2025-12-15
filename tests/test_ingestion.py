import pytest
import pandas as pd
from unittest.mock import MagicMock, patch
from src.ingestion.market_data import MarketDataFetcher

@patch("src.ingestion.market_data.yf.download")
def test_fetch_history_success(mock_download, mock_bronze_path):
    # Setup Mock
    mock_df = pd.DataFrame({"Close": [100, 101]}, index=[0, 1])
    mock_download.return_value = mock_df
    
    fetcher = MarketDataFetcher(bronze_path=mock_bronze_path)
    result = fetcher.fetch_history(["AAPL"], period="1d")
    
    assert result is not None
    assert not result.empty
    assert mock_download.called
    
    # Verify file was saved
    saved_files = list(mock_bronze_path.glob("*.parquet"))
    assert len(saved_files) == 1

@patch("src.ingestion.market_data.yf.download")
def test_fetch_history_empty(mock_download, mock_bronze_path):
    # Setup Mock to return empty
    mock_download.return_value = pd.DataFrame()
    
    fetcher = MarketDataFetcher(bronze_path=mock_bronze_path)
    result = fetcher.fetch_history(["AAPL"], period="1d")
    
    assert result is None
    # No file should be saved
    saved_files = list(mock_bronze_path.glob("*.parquet"))
    assert len(saved_files) == 0
