import yfinance as yf
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import List, Optional

class MarketDataFetcher:
    """
    Fetches raw market data (OHLCV) using yfinance and saves it to the Bronze layer.
    Follows Medallion Architecture: Bronze = Raw, Immutable.
    """
    def __init__(self, bronze_path: str = "market_mind/data/bronze"):
        self.bronze_path = Path(bronze_path)
        self.bronze_path.mkdir(parents=True, exist_ok=True)

    def fetch_history(self, tickers: List[str], period: str = "1mo") -> Optional[pd.DataFrame]:
        """
        Fetches historical data for a list of tickers.
        
        Args:
            tickers: List of symbol strings (e.g., ['AAPL', 'NVDA'])
            period: valid yfinance period string (e.g., '1mo', '1y', 'max')
            
        Returns:
            pd.DataFrame: The raw data fetched, or None if empty.
        """
        print(f"Fetching data for {len(tickers)} tickers: {tickers} over {period}...")
        
        try:
            # auto_adjust=True fixes the FutureWarning and gives us adjusted close by default
            data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, progress=False)
            
            if data.empty:
                print("Warning: No data fetched.")
                return None
            
            self._save_to_bronze(data, period)
            return data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return None

    def _save_to_bronze(self, data: pd.DataFrame, period: str):
        """Saves the raw dataframe to a parquet file with a timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"market_data_{period}_{timestamp}.parquet"
        file_path = self.bronze_path / filename
        
        # Save as parquet for efficiency and schema preservation
        data.to_parquet(file_path)
        print(f"Saved raw data to {file_path}")

if __name__ == "__main__":
    # V1 Scale Up: Diverse Universe (Tech, Finance, Healthcare, Energy, Consumer, Industrial)
    v1_tickers = [
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
    
    fetcher = MarketDataFetcher()
    fetcher.fetch_history(v1_tickers, period="2y")
