import pandas as pd
import numpy as np
from typing import Dict, Union

class BenchmarkEngine:
    """
    Analytics Component: Comparative Statistics.
    Standardizes performance measurement across different strategies (MVO, HRP, GNN, Market).
    """
    def __init__(self, risk_free_rate: float = 0.04):
        self.rf = risk_free_rate
        self.trading_days = 252

    def calculate_metrics(self, returns: pd.Series, name: str = "Strategy") -> Dict[str, float]:
        """
        Computes key financial metrics for a return series.
        """
        if returns.empty:
            return {}

        # 1. Total & Annualized Return
        total_ret = (1 + returns).prod() - 1
        n_years = len(returns) / self.trading_days
        ann_ret = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 2. Volatility (Annualized)
        volatility = returns.std() * np.sqrt(self.trading_days)

        # 3. Sharpe Ratio
        sharpe = (ann_ret - self.rf) / volatility if volatility > 0 else 0

        # 4. Sortino Ratio (Downside Deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(self.trading_days)
        sortino = (ann_ret - self.rf) / downside_std if downside_std > 0 else 0

        # 5. Max Drawdown
        cum_ret = (1 + returns).cumprod()
        rolling_max = cum_ret.cummax()
        drawdown = (cum_ret - rolling_max) / rolling_max
        max_dd = drawdown.min()

        # 6. Calmar Ratio
        calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

        # 7. Win Rate
        win_rate = len(returns[returns > 0]) / len(returns)

        return {
            "Strategy": name,
            "Total Return": total_ret,
            "Annualized Return": ann_ret,
            "Volatility": volatility,
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "Max Drawdown": max_dd,
            "Calmar Ratio": calmar,
            "Win Rate": win_rate
        }

    def compare_strategies(self, strategy_returns: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        Aggregates metrics for multiple strategies into a comparison table.
        Args:
            strategy_returns: Dictionary { 'MVO': pd.Series(...), 'Market': pd.Series(...) }
        """
        results = []
        for name, series in strategy_returns.items():
            metrics = self.calculate_metrics(series, name)
            results.append(metrics)
        
        df = pd.DataFrame(results).set_index("Strategy")
        
        # Formatting for display (return clean float df, but print formatted)
        print("\n=== Strategy Performance Benchmark ===")
        print(df.style.format({
            "Total Return": "{:.2%}",
            "Annualized Return": "{:.2%}",
            "Volatility": "{:.2%}",
            "Sharpe Ratio": "{:.2f}",
            "Sortino Ratio": "{:.2f}",
            "Max Drawdown": "{:.2%}",
            "Calmar Ratio": "{:.2f}",
            "Win Rate": "{:.2%}"
        }).to_string())
        
        return df

if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=252)
    # Strategy A: Low Vol, Low Return
    ret_a = pd.Series(np.random.normal(0.0002, 0.005, 252), index=dates)
    # Strategy B: High Vol, High Return
    ret_b = pd.Series(np.random.normal(0.0005, 0.015, 252), index=dates)
    
    engine = BenchmarkEngine()
    engine.compare_strategies({
        "Conservative (A)": ret_a,
        "Aggressive (B)": ret_b
    })
