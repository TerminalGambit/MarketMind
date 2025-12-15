import pandas as pd
import numpy as np
from src.analytics.portfolio_optimizer import PortfolioOptimizer
from src.analytics.advanced_optimizer import AdvancedOptimizer
from src.analytics.benchmark import BenchmarkEngine

def run_benchmark():
    print("=== Starting Strategy Benchmark (Out-of-Sample) ===")
    
    # 1. Load Data
    mvo_gen = PortfolioOptimizer()
    try:
        full_returns = mvo_gen.load_returns()
    except FileNotFoundError:
        print("No return data found. Please run market_data.py and features.py first.")
        return

    # 2. Train/Test Split (70/30)
    split_idx = int(len(full_returns) * 0.7)
    train_df = full_returns.iloc[:split_idx]
    test_df = full_returns.iloc[split_idx:]
    
    print(f"Total Days: {len(full_returns)}")
    print(f"Training:   {len(train_df)} days")
    print(f"Testing:    {len(test_df)} days")
    
    # 3. Strategy Optimization (on Train data)
    adv_opt = AdvancedOptimizer()
    strategies = {}
    
    # --- A. Equal Weight (Benchmark) ---
    print("optimizing: Equal Weight...")
    n_assets = len(train_df.columns)
    strategies['Equal Weight'] = pd.Series(1/n_assets, index=train_df.columns)
    
    # --- B. Mean-Variance (Max Sharpe) ---
    print("optimizing: MVO (Max Sharpe)...")
    strategies['MVO'] = mvo_gen.get_max_sharpe_weights(train_df)
    
    # --- C. Hierarchical Risk Parity (HRP) ---
    print("optimizing: HRP...")
    strategies['HRP'] = adv_opt.get_hrp_weights(train_df)
    
    # --- D. CVaR (Min Expected Shortfall) ---
    print("optimizing: CVaR (Min ES)...")
    strategies['CVaR'] = adv_opt.get_cvar_weights(train_df)
    
    # --- E. Black-Litterman (with Dummy View) ---
    print("optimizing: Black-Litterman...")
    # View: Top 1 asset from training momentum performs well? 
    # Let's just say "AAPL" (if exists) will outperform by 5%
    if 'AAPL' in train_df.columns:
        views = {'AAPL': 0.05}
    else:
        # Pick first asset
        first_asset = train_df.columns[0]
        views = {first_asset: 0.05}
    
    strategies['Black-Litterman'] = adv_opt.get_black_litterman_weights(train_df, view_dict=views)

    # 4. Out-of-Sample Performance
    print("\nCalculating Out-of-Sample Performance...")
    results = {}
    
    for name, weights in strategies.items():
        # Ensure alignment
        aligned_weights = weights.reindex(test_df.columns).fillna(0)
        
        # Calculate portfolio returns: R_p = Sum(w_i * R_i)
        # test_df is [Time x Assets]
        port_ret = test_df.dot(aligned_weights)
        results[name] = port_ret
        
    # 5. Review with Engine
    engine = BenchmarkEngine()
    engine.compare_strategies(results)
    
if __name__ == "__main__":
    run_benchmark()
