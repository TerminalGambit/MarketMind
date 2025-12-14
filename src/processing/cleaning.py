import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

class MarketDataProcessor:
    """
    Silver Layer Processor: Cleans and transforms raw market data.
    - Aligns timestamps
    - Handles missing values (Forward Fill)
    - Computes basic features (Log Returns, Volatility)
    """
    def __init__(self, bronze_path: str = "market_mind/data/bronze", silver_path: str = "market_mind/data/silver"):
        self.bronze_path = Path(bronze_path)
        self.silver_path = Path(silver_path)
        self.silver_path.mkdir(parents=True, exist_ok=True)

    def load_latest_bronze(self) -> pd.DataFrame:
        """Finds and loads the most recent market data parquet file from Bronze."""
        files = list(self.bronze_path.glob("market_data_*.parquet"))
        if not files:
            raise FileNotFoundError("No market data found in Bronze layer.")
        
        # Sort by creation time (or filename timestamp)
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading raw data from: {latest_file}")
        return pd.read_parquet(latest_file)

    def process(self):
        """Runs the cleaning pipeline."""
        df = self.load_latest_bronze()
        
        # 1. Handling Missing Data
        # Forward fill is standard for financial time series (last known price)
        df_clean = df.ffill().bfill()
        
        # 2. Feature Engineering
        # Calculate Log Returns for each ticker's 'Close' price
        # df structure from yfinance is MultiIndex (Price, Ticker)
        
        # We need to handle the MultiIndex columns correctly
        # Structure is usually: Level 0 = [Close, High, ...], Level 1 = [AAPL, NVDA...]
        # Or Level 0 = [AAPL, ...], Level 1 = [Close, ...] depending on download args.
        # simpler approach: iterate if needed or use cross-section
        
        # Let's verify structure. The header usually has 'Ticker' as level 1.
        # If 'Close' is a column level, we extract it.
        
        silvers = {}
        
        try:
             # Check if we have MultiIndex columns
            if isinstance(df.columns, pd.MultiIndex):
                # Validating if 'Close' is in level 0
                if 'Close' in df.columns.get_level_values(0):
                    closes = df['Close']
                else:
                    # Maybe Ticker is Level 0?
                    # Let's try to infer or assume standard format
                    closes = df.xs('Close', level=1, axis=1) 
            else:
                 # Flat dataframe (single ticker case usually, but we downloaded groups)
                if 'Close' in df.columns:
                    closes = df[['Close']]
                else:
                    raise ValueError("Could not find 'Close' prices in dataframe.")

            # Calculate Log Returns
            log_returns = np.log(closes / closes.shift(1))
            
            # 3. Aggregation & Saving
            # We save the cleaned prices AND returns
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            clean_path = self.silver_path / f"market_prices_clean_{timestamp}.parquet"
            returns_path = self.silver_path / f"market_returns_{timestamp}.parquet"
            
            df_clean.to_parquet(clean_path)
            log_returns.to_parquet(returns_path)
            
            print(f"Saved Silver data:\n  - {clean_path}\n  - {returns_path}")
            print(f"Sample Returns:\n{log_returns.tail(3)}")
            
        except Exception as e:
            print(f"Error processing data: {e}")
            # Debugging: print columns
            print(f"Columns: {df.columns}")

if __name__ == "__main__":
    processor = MarketDataProcessor()
    processor.process()
