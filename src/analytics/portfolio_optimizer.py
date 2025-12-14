import pandas as pd
import numpy as np
import scipy.optimize as sco
import matplotlib.pyplot as plt
from pathlib import Path

class PortfolioOptimizer:
    """
    Phase 5: Modern Portfolio Theory (MPT) Engine.
    Calculates Efficient Frontier and Optimal Weights.
    """
    def __init__(self, silver_path: str = "market_mind/data/silver", gold_path: str = "market_mind/data/gold"):
        self.silver_path = Path(silver_path)
        self.gold_path = Path(gold_path)
        self.rf_rate = 0.04 # Risk Free Rate (4%)

    def load_returns(self):
        files = list(self.silver_path.glob("market_returns_*.parquet"))
        if not files:
            raise FileNotFoundError("Returns data not found.")
        latest = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading returns from: {latest.name}")
        df = pd.read_parquet(latest)
        return df

    def portfolio_performance(self, weights, mean_returns, cov_matrix):
        returns = np.sum(mean_returns * weights) * 252
        std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        return returns, std

    def neg_sharpe_ratio(self, weights, mean_returns, cov_matrix, rf_rate):
        p_ret, p_std = self.portfolio_performance(weights, mean_returns, cov_matrix)
        return - (p_ret - rf_rate) / p_std

    def optimize(self):
        print("Running Mean-Variance Optimization...")
        df = self.load_returns()
        
        # Data Stats
        mean_returns = df.mean()
        cov_matrix = df.cov()
        num_assets = len(mean_returns)
        tickers = df.columns.tolist()
        
        # Constraints & Bounds
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0.0, 1.0) for asset in range(num_assets))
        init_guess = num_assets * [1. / num_assets,]
        
        # Max Sharpe Optimization
        print("Maximizing Sharpe Ratio...")
        opts = sco.minimize(self.neg_sharpe_ratio, init_guess, args=(mean_returns, cov_matrix, self.rf_rate),
                            method='SLSQP', bounds=bounds, constraints=constraints)
        
        opt_weights = opts.x
        opt_ret, opt_std = self.portfolio_performance(opt_weights, mean_returns, cov_matrix)
        opt_sharpe = (opt_ret - self.rf_rate) / opt_std
        
        print(f"\n=== Optimal Portfolio (Max Sharpe) ===")
        print(f"Expected Return: {opt_ret*100:.2f}%")
        print(f"Volatility: {opt_std*100:.2f}%")
        print(f"Sharpe Ratio: {opt_sharpe:.2f}")
        
        # Top Holdings
        w_series = pd.Series(opt_weights, index=tickers)
        top_holdings = w_series[w_series > 0.01].sort_values(ascending=False)
        print("\nTop Holdings (>1%):")
        print(top_holdings.head(10))
        
        # Generate Efficient Frontier (Simulated)
        print("\nGenerating Efficient Frontier (Monte Carlo)...")
        num_portfolios = 5000
        results = np.zeros((3, num_portfolios))
        
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            p_ret, p_std = self.portfolio_performance(weights, mean_returns, cov_matrix)
            results[0,i] = p_std
            results[1,i] = p_ret
            results[2,i] = (p_ret - self.rf_rate) / p_std
            
        # Plot
        plt.figure(figsize=(10, 6))
        plt.scatter(results[0,:], results[1,:], c=results[2,:], cmap='viridis', marker='o', s=10, alpha=0.3)
        plt.colorbar(label='Sharpe Ratio')
        plt.scatter(opt_std, opt_ret, marker='*', color='r', s=500, label='Max Sharpe')
        
        # Plot individual assets
        # plt.scatter(np.sqrt(np.diag(cov_matrix))*np.sqrt(252), mean_returns*252, marker='x', color='black', alpha=0.5)
        
        plt.title('Efficient Frontier (Mean-Variance Optimization)')
        plt.xlabel('Volatility (Risk)')
        plt.ylabel('Expected Return')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_plot = self.gold_path / "efficient_frontier.png"
        plt.savefig(output_plot)
        print(f"Saved frontier plot to {output_plot}")
        plt.close()
        
        # Save Weights
        output_csv = self.gold_path / "optimal_portfolio.csv"
        top_holdings.to_csv(output_csv)
        print(f"Saved weights to {output_csv}")

if __name__ == "__main__":
    opt = PortfolioOptimizer()
    opt.optimize()
