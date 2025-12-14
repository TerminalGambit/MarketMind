import pandas as pd
import numpy as np
from pathlib import Path
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
import warnings

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore")

class StatsEngine:
    """
    Analytics Component: Rigorous Statistical Hypothesis Testing.
    - Stationarity Checks (ADF Test)
    - Causality Checks (Granger Causality)
    """
    def __init__(self, silver_path: str = "market_mind/data/silver"):
        self.silver_path = Path(silver_path)

    def load_returns(self) -> pd.DataFrame:
        files = list(self.silver_path.glob("market_returns_*.parquet"))
        if not files: return pd.DataFrame()
        latest = max(files, key=lambda f: f.stat().st_mtime)
        return pd.read_parquet(latest)

    def load_prices(self) -> pd.DataFrame:
        files = list(self.silver_path.glob("market_prices_clean_*.parquet"))
        if not files: return pd.DataFrame()
        latest = max(files, key=lambda f: f.stat().st_mtime)
        return pd.read_parquet(latest)

    def check_stationarity(self):
        """
        Runs Augmented Dickey-Fuller (ADF) test.
        H0: The time series has a unit root (is non-stationary).
        H1: The time series is stationary.
        """
        df_ret = self.load_returns().dropna()
        print("\n=== Hypothesis Test: Stationarity (ADF) ===")
        print("H0: Series is Non-Stationary (Random Walk)")
        print("H1: Series is Stationary (Mean Reverting)")
        
        results = []
        for ticker in df_ret.columns:
            series = df_ret[ticker].values
            # ADF test
            adf_result = adfuller(series)
            p_value = adf_result[1]
            is_stationary = p_value < 0.05
            
            results.append({
                "Ticker": ticker,
                "ADF Statistic": round(adf_result[0], 4),
                "p-value": round(p_value, 6),
                "Stationary (95%)": is_stationary
            })
            
        res_df = pd.DataFrame(results)
        print(res_df)
        return res_df

    def test_granger_causality(self, output_target: str = "AAPL", max_lag: int = 3):
        """
        Tests if other tickers 'Granger Cause' the target ticker.
        """
        df_ret = self.load_returns().dropna()
        if output_target not in df_ret.columns:
            print(f"Target {output_target} not found in data.")
            return

        print(f"\n=== Hypothesis Test: Granger Causality (Target: {output_target}) ===")
        print(f"Does X provide statistically significant info about future {output_target}?")
        
        predictors = [c for c in df_ret.columns if c != output_target]
        
        for predictor in predictors:
            # Combine into 2D array: [Target, Predictor]
            data = df_ret[[output_target, predictor]]
            
            print(f"\nTesting: Does {predictor} -> {output_target}?")
            try:
                # Statsmodels granger causality returns a dict of results for each lag
                gc_res = grangercausalitytests(data, maxlag=max_lag, verbose=False)
                
                # We check the p-value of the F-test for each lag
                significant = False
                for lag in range(1, max_lag + 1):
                    # getting F-test p-value (params_ftest) -> index 0 is F-stat, 1 is p-value
                    p_val = gc_res[lag][0]['ssr_ftest'][1]
                    if p_val < 0.05:
                        print(f"  * Significant at Lag {lag} (p={p_val:.4f})")
                        significant = True
                
                if not significant:
                    print("  - No significant causality found.")
                    
            except Exception as e:
                print(f"  Error: {e}")

if __name__ == "__main__":
    stats = StatsEngine()
    stats.check_stationarity()
    stats.test_granger_causality(output_target="AAPL")
