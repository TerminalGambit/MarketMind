import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.analytics.portfolio_optimizer import PortfolioOptimizer
from src.analytics.advanced_optimizer import AdvancedOptimizer

def plot_simple_strategy(name, returns, save_path):
    """
    Create a simple cumulative returns plot for a single strategy,
    matching the style of backtest_results.png
    """
    # Calculate cumulative returns
    cum_returns = (1 + returns).cumprod()
    
    # Color scheme - same as before
    color_map = {
        'Black-Litterman': '#A23B72',  # Purple
        'Equal Weight': '#F18F01',     # Orange
        'HRP': '#C73E1D',              # Red
        'MVO': '#6A994E',              # Green
        'CVaR': '#BC4B51'              # Dark red
    }
    color = color_map.get(name, '#2E86AB')
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Plot cumulative returns
    ax.plot(cum_returns.index, cum_returns.values, 
            linewidth=3, color=color, label=name, alpha=0.9)
    
    # Fill area under curve
    ax.fill_between(cum_returns.index, 1, cum_returns.values, 
                     alpha=0.2, color=color)
    
    # Add baseline
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1.5, 
               label='Initial Investment ($1)')
    
    # Styling
    ax.set_title(f'{name} - Cumulative Returns (Out-of-Sample)', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Date', fontsize=13, fontweight='bold')
    ax.set_ylabel('Portfolio Value ($)', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'${y:.2f}'))
    
    # Add final value annotation
    final_value = cum_returns.iloc[-1]
    total_return = (final_value - 1) * 100
    
    ax.annotate(f'Final Value: ${final_value:.2f}\nReturn: {total_return:+.2f}%', 
                xy=(cum_returns.index[-1], final_value),
                xytext=(-120, 20), textcoords='offset points',
                fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor=color, 
                         alpha=0.85, edgecolor='white', linewidth=2),
                color='white',
                arrowprops=dict(arrowstyle='->', color=color, lw=2))
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path.name}")

def run_simple_plots():
    print("=== Generating Simple Individual Strategy Plots ===\n")
    
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
    print(f"Testing:    {len(test_df)} days\n")
    
    # 3. Strategy Optimization (on Train data)
    adv_opt = AdvancedOptimizer()
    strategies = {}
    
    # --- A. Equal Weight (Benchmark) ---
    print("Optimizing: Equal Weight...")
    n_assets = len(train_df.columns)
    strategies['Equal Weight'] = pd.Series(1/n_assets, index=train_df.columns)
    
    # --- B. Mean-Variance (Max Sharpe) ---
    print("Optimizing: MVO (Max Sharpe)...")
    strategies['MVO'] = mvo_gen.get_max_sharpe_weights(train_df)
    
    # --- C. Hierarchical Risk Parity (HRP) ---
    print("Optimizing: HRP...")
    strategies['HRP'] = adv_opt.get_hrp_weights(train_df)
    
    # --- D. CVaR (Min Expected Shortfall) ---
    print("Optimizing: CVaR (Min ES)...")
    strategies['CVaR'] = adv_opt.get_cvar_weights(train_df)
    
    # --- E. Black-Litterman (with Dummy View) ---
    print("Optimizing: Black-Litterman...")
    if 'AAPL' in train_df.columns:
        views = {'AAPL': 0.05}
    else:
        first_asset = train_df.columns[0]
        views = {first_asset: 0.05}
    
    strategies['Black-Litterman'] = adv_opt.get_black_litterman_weights(train_df, view_dict=views)

    # 4. Calculate Out-of-Sample Performance
    print("\nCalculating Out-of-Sample Performance...")
    results = {}
    
    for name, weights in strategies.items():
        aligned_weights = weights.reindex(test_df.columns).fillna(0)
        port_ret = test_df.dot(aligned_weights)
        results[name] = port_ret
    
    # 5. Generate Individual Plots
    output_dir = Path("data/gold/simple_individual")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating simple plots in: {output_dir}\n")
    
    for name, returns in results.items():
        safe_name = name.replace(' ', '_').replace('-', '_').lower()
        save_path = output_dir / f"{safe_name}_simple.png"
        plot_simple_strategy(name, returns, save_path)
    
    print(f"\n✓ All plots generated successfully!")
    print(f"📁 Location: {output_dir.absolute()}")

if __name__ == "__main__":
    run_simple_plots()
