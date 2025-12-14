import pandas as pd
import numpy as np
from pathlib import Path

class FeatureEngineer:
    """
    Silver Layer: Technical Analysis Feature Engineering.
    Enriches 'market_prices_clean' with indicators like RSI, MACD, Bollinger Bands.
    """
    def __init__(self, silver_path: str = "market_mind/data/silver"):
        self.silver_path = Path(silver_path)

    def load_clean_prices(self) -> pd.DataFrame:
        files = list(self.silver_path.glob("market_prices_clean_*.parquet"))
        if not files: return pd.DataFrame()
        latest = max(files, key=lambda f: f.stat().st_mtime)
        return pd.read_parquet(latest)

    def compute_rsi(self, series: pd.Series, window=14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss.replace(0, np.nan) 
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50) # Neutral fill

    def compute_macd(self, series: pd.Series) -> pd.Series:
        exp1 = series.ewm(span=12, adjust=False).mean()
        exp2 = series.ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        # We return the histogram (Distance between MACD and Signal)
        return macd_line - signal_line

    def compute_bollinger_bands(self, series: pd.Series, window=20) -> pd.Series:
        sma = series.rolling(window=window).mean()
        std = series.rolling(window=window).std()
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        # Return %B (Position within bands)
        # 0 = Lower Band, 1 = Upper Band
        width = upper - lower
        percent_b = (series - lower) / width.replace(0, np.nan)
        return percent_b.fillna(0.5)

    def generate_features(self):
        df_prices = self.load_clean_prices().ffill() # Forward fill any holes
        if df_prices.empty:
            print("No price data found.")
            return

        # Handling MultiIndex from yfinance
        if isinstance(df_prices.columns, pd.MultiIndex):
            # Check if 'Close' is in the first or second level
            # Usually yfinance gives (Price, Ticker) or (Ticker, Price) depending on version/args
            # Based on the error tuple ('AMD', 'Open'), it seems to be (Ticker, PriceType)
            # Let's inspect the first column
            first_col = df_prices.columns[0]
            print(f"Detected MultiIndex columns: {first_col}")
            
            # If (Ticker, Price), we need to select xs or swaplevel
            # But wait, ('AMD', 'Open') implies column is (Ticker, PriceType)
            # Let's try to find 'Close' in the second level
            try:
                # xs(key, axis=1, level=1, drop_level=True)
                df_close = df_prices.xs('Close', axis=1, level=1)
            except KeyError:
                # Maybe it is (Price, Ticker)?
                try:
                    df_close = df_prices.xs('Close', axis=1, level=0)
                except KeyError:
                    print("Could not find 'Close' price level.")
                    return
        else:
            # Flat columns, assume they are tickers and values are Close prices
            df_close = df_prices

        print(f"Generating Technical Indicators for {len(df_close.columns)} tickers...")
        
        feature_frames = []
        
        for ticker in df_close.columns:
            series = df_close[ticker]
            # Ensure series is numeric
            series = pd.to_numeric(series, errors='coerce')
            
            # Compute Indicators
            rsi = self.compute_rsi(series)
            macd = self.compute_macd(series)
            bb = self.compute_bollinger_bands(series)
            
            print(f"  Shapes - RSI: {len(rsi)}, MACD: {len(macd)}, BB: {len(bb)}")
            
            # Combine into a MultiIndex DataFrame or Panel
            # For simplicity in this V1, let's create a wide format for each feature, 
            # OR a long format. GNN expects [Node, Time, Features].
            # Let's save separate Parquet files for each feature type for modularity.
            
            # Only storing them in memory for now to save a Unified Feature Map?
            # Let's create a combined dataframe: Ticker | Date | RSI | MACD | BB
            
            df_feat = pd.DataFrame({
                'Ticker': ticker,
                'Close': series,
                'RSI': rsi,
                'MACD': macd,
                'BB_Pct': bb
            })
            feature_frames.append(df_feat)
            
        full_df = pd.concat(feature_frames)
        
        # Save enriched data
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        out_path = self.silver_path / f"market_features_v1_{timestamp}.parquet"
        
        # We'll pivot or keep long based on needs. Long is good for GNN masking.
        # But our GNN loader might expect Wide.
        # Let's save this Long format as the "Enriched Silver Source"
        full_df.to_parquet(out_path)
        print(f"Saved {len(full_df)} rows of enriched features to {out_path}")
        return out_path

if __name__ == "__main__":
    fe = FeatureEngineer()
    fe.generate_features()
