import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

class Backtester:
    """
    Backtesting Engine for Market-Mind.
    Simulates a Vectorized Long/Short Strategy based on Alpha Signals.
    """
    def __init__(self, gold_path: str = "market_mind/data/gold", silver_path: str = "market_mind/data/silver"):
        self.gold_path = Path(gold_path)
        self.silver_path = Path(silver_path)
        self.output_path = self.gold_path / "backtest_results.png"

    def load_data(self):
        # 1. Load Predictions (Signal)
        pred_path = self.gold_path / "backtest_predictions.parquet"
        if not pred_path.exists():
            raise FileNotFoundError("Predictions not found. Run GNN first.")
        
        preds = pd.read_parquet(pred_path)
        
        # 2. Load Actual Returns (Truth)
        # We need the returns for the SAME PERIOD as predictions.
        # The silver layer has the full history.
        files = list(self.silver_path.glob("market_returns_*.parquet"))
        latest = max(files, key=lambda f: f.stat().st_mtime)
        returns = pd.read_parquet(latest)
        
        # Align Data
        # Predictions at t are for t+1 return.
        # So we trade at t (close) to capture return at t+1.
        # Let's align indices.
        
        # Ensure datetimes
        preds.index = pd.to_datetime(preds.index)
        returns.index = pd.to_datetime(returns.index)
        
        print(f"Preds range: {preds.index.min()} to {preds.index.max()}")
        print(f"Returns range: {returns.index.min()} to {returns.index.max()}")
        
        common_dates = preds.index.intersection(returns.index)
        print(f"Common dates: {len(common_dates)}")
        
        if len(common_dates) == 0:
            print("Error: No overlapping dates found between Predictions and Returns.")
            return pd.DataFrame(), pd.DataFrame()

        preds = preds.loc[common_dates]
        returns = returns.loc[common_dates]
        
        return preds, returns

    def run_strategy(self, long_k=5, short_k=5):
        print(f"Running Backtest: Long Top {long_k} / Short Bottom {short_k}...")
        preds, returns = self.load_data()
        
        portfolio_returns = []
        
        # Vectorized Loop (or iterrows for clarity in pedagogical code)
        # Detailed logging for first few days
        
        for date in preds.index:
            # Signal for this day (predicting tomorrow's move)
            daily_signal = preds.loc[date]
            daily_truth = returns.loc[date] # Actual return achieved this day
            
            # Note: GNN predicts 'Next Day Return'.
            # Did we align correctly?
            # GNN: Input(t) -> Output(t+1).
            # The saved prediction at index 'date' corresponds to the target return at 'date'.
            # (Because in GNN we did: test_dates = dates[split_idx+1:])
            # So preds.loc[date] is the PREDICTED return for 'date'.
            # And returns.loc[date] is the ACTUAL return for 'date'.
            # Perfect.
            
            # Ranking
            sorted_tickers = daily_signal.sort_values(ascending=False)
            
            longs = sorted_tickers.head(long_k).index
            shorts = sorted_tickers.tail(short_k).index
            
            # Returns
            long_ret = daily_truth.loc[longs].mean()
            short_ret = daily_truth.loc[shorts].mean()
            
            # Market Neutral Strategy Return
            # Return = (Long_Ret - Short_Ret) / 2 (assuming 100% GMV, 50% Long, 50% Short)
            # Or effectively: 0.5 * Long + 0.5 * (-Short)
            
            # If short_k is 0, Long Only
            if short_k == 0:
                day_ret = long_ret
            else:
                day_ret = 0.5 * long_ret - 0.5 * short_ret
                
            portfolio_returns.append(day_ret)
            
        # Metrics
        equity_curve = pd.Series(portfolio_returns, index=preds.index).cumsum()
        
        total_return = equity_curve.iloc[-1]
        daily_std = np.std(portfolio_returns)
        sharpe = np.mean(portfolio_returns) / daily_std * np.sqrt(252) if daily_std != 0 else 0
        
        # Drawdown
        rolling_max = equity_curve.cummax()
        drawdown = equity_curve - rolling_max
        max_drawdown = drawdown.min()
        
        print(f"\n=== Backtest Performance ===")
        print(f"Total Return: {total_return * 100:.2f}%")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Max Drawdown: {max_drawdown * 100:.2f}%")
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Compare vs Benchmark (SPY equivalent or Mean Market)
        market_return = returns.mean(axis=1).cumsum() # Equal weight index
        
        plt.plot(equity_curve, label='Market-Mind AI Strategy', color='#00ff88', linewidth=2)
        plt.plot(market_return, label='Equal Weight Market', color='gray', linestyle='--', alpha=0.6)
        
        plt.title(f"Identify Alpha (Market Neutral) - Sharpe: {sharpe:.2f}", fontsize=14)
        plt.ylabel("Cumulative Returns")
        plt.grid(True, alpha=0.2)
        plt.legend()
        
        print(f"Saving Equity Curve to {self.output_path}...")
        plt.savefig(self.output_path)
        plt.close()
        
if __name__ == "__main__":
    bt = Backtester()
    bt.run_strategy(long_k=5, short_k=5)
