import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

class EconometricModels:
    """
    Analytics Component: Linear Models & Factor Analysis.
    Uses Lasso (L1) and Ridge (L2) regression to identify drivers of stock returns.
    """
    def __init__(self, silver_path: str = "market_mind/data/silver"):
        self.silver_path = Path(silver_path)

    def load_returns(self) -> pd.DataFrame:
        files = list(self.silver_path.glob("market_returns_*.parquet"))
        if not files: return pd.DataFrame()
        latest = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading returns from: {latest}")
        return pd.read_parquet(latest).dropna()

    def run_factor_analysis(self, target_ticker: str = "AAPL"):
        """
        Regresses Target ~ Features (all other tickers).
        Uses Lasso to select key features (sparse coefficients).
        """
        df = self.load_returns()
        if target_ticker not in df.columns:
            print(f"Target {target_ticker} not found.")
            return

        print(f"\n=== Factor Analysis (Lasso Regression) for {target_ticker} ===")
        
        # X = All other tickers, y = Target
        X = df.drop(columns=[target_ticker])
        y = df[target_ticker]
        
        if X.empty:
            print("Not enough data for regression.")
            return

        # Train/Test logic not strictly needed for 'Analysis', but good practice
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        # Lasso Model (L1 regularization forces weak coefficients to zero)
        # Alpha is the regularization strength. 
        model = Lasso(alpha=0.0001) 
        model.fit(X_train, y_train)
        
        # Evaluation
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Model R-Squared (Out-of-Sample): {r2:.4f}")
        
        # Feature Importance
        coeffs = pd.Series(model.coef_, index=X.columns)
        
        # Sort by absolute magnitude
        coeffs = coeffs.iloc[(-coeffs.abs()).argsort()]
        
        print("\nKey Drivers (Coefficients):")
        print(coeffs[coeffs != 0])
        
        if (coeffs == 0).all():
            print("  (Lasso shrunk all coefficients to zero - try lower alpha or check correlations)")

        return coeffs

if __name__ == "__main__":
    econ = EconometricModels()
    econ.run_factor_analysis(target_ticker="AAPL")
    econ.run_factor_analysis(target_ticker="NVDA")
