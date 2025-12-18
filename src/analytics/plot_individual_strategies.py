import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from src.analytics.portfolio_optimizer import PortfolioOptimizer
from src.analytics.advanced_optimizer import AdvancedOptimizer
from src.analytics.benchmark import BenchmarkEngine

def plot_individual_strategy(name, returns, metrics, save_path):
    """
    Create a detailed plot for a single strategy showing:
    - Cumulative returns
    - Drawdown
    - Key metrics
    """
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3, height_ratios=[2, 1.5, 0.8])
    
    # Calculate cumulative returns and drawdown
    cum_returns = (1 + returns).cumprod()
    rolling_max = cum_returns.cummax()
    drawdown = (cum_returns - rolling_max) / rolling_max
    
    # Color scheme
    color = '#2E86AB'  # Blue
    if name == 'Black-Litterman':
        color = '#A23B72'  # Purple
    elif name == 'Equal Weight':
        color = '#F18F01'  # Orange
    elif name == 'HRP':
        color = '#C73E1D'  # Red
    elif name == 'MVO':
        color = '#6A994E'  # Green
    elif name == 'CVaR':
        color = '#BC4B51'  # Dark red
    
    # 1. Cumulative Returns (large plot)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(cum_returns.index, cum_returns.values, linewidth=2.5, color=color, label=name)
    ax1.fill_between(cum_returns.index, 1, cum_returns.values, alpha=0.3, color=color)
    ax1.axhline(y=1, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.set_title(f'{name} - Cumulative Returns', fontsize=16, fontweight='bold', pad=15)
    ax1.set_ylabel('Portfolio Value ($1 Initial)', fontsize=12)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.legend(loc='upper left', fontsize=11)
    
    # Add final value annotation
    final_value = cum_returns.iloc[-1]
    ax1.annotate(f'Final: ${final_value:.2f}', 
                xy=(cum_returns.index[-1], final_value),
                xytext=(-60, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.7, edgecolor='none'),
                color='white')
    
    # 2. Drawdown
    ax2 = fig.add_subplot(gs[1, :])
    ax2.fill_between(drawdown.index, 0, drawdown.values * 100, color='#C73E1D', alpha=0.6)
    ax2.plot(drawdown.index, drawdown.values * 100, color='#8B0000', linewidth=1.5)
    ax2.set_title('Drawdown', fontsize=14, fontweight='bold', pad=10)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    
    # Add max drawdown annotation
    max_dd_idx = drawdown.idxmin()
    max_dd_val = drawdown.min() * 100
    ax2.annotate(f'Max DD: {max_dd_val:.2f}%', 
                xy=(max_dd_idx, max_dd_val),
                xytext=(20, -20), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='#C73E1D', alpha=0.8, edgecolor='none'),
                color='white',
                arrowprops=dict(arrowstyle='->', color='#8B0000', lw=1.5))
    
    # 3. Performance Metrics (left)
    ax3 = fig.add_subplot(gs[2, 0])
    ax3.axis('off')
    
    metrics_text = f"""
    RETURN METRICS
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    Total Return:           {metrics['Total Return']:.2%}
    Annualized Return:      {metrics['Annualized Return']:.2%}
    Volatility:             {metrics['Volatility']:.2%}
    """
    
    ax3.text(0.05, 0.9, metrics_text, transform=ax3.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.9, pad=8))
    
    # 4. Risk Metrics (right)
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.axis('off')
    
    risk_text = f"""
    RISK METRICS
    ━━━━━━━━━━━━━━━━━━━━━━━━━
    Sharpe Ratio:           {metrics['Sharpe Ratio']:.2f}
    Sortino Ratio:          {metrics['Sortino Ratio']:.2f}
    Max Drawdown:           {metrics['Max Drawdown']:.2%}
    Calmar Ratio:           {metrics['Calmar Ratio']:.2f}
    Win Rate:               {metrics['Win Rate']:.2%}
    """
    
    ax4.text(0.05, 0.9, risk_text, transform=ax4.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f0f0f0', alpha=0.9, pad=8))
    
    # Overall title
    fig.suptitle(f'Strategy Performance Analysis: {name}', 
                fontsize=18, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✓ Saved: {save_path}")

def run_individual_plots():
    print("=== Generating Individual Strategy Plots ===\n")
    
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
    
    # 5. Calculate Metrics
    engine = BenchmarkEngine()
    all_metrics = {}
    for name, returns in results.items():
        metrics = engine.calculate_metrics(returns, name)
        all_metrics[name] = metrics
    
    # 6. Generate Individual Plots
    output_dir = Path("data/gold/individual_strategies")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating individual plots in: {output_dir}\n")
    
    for name, returns in results.items():
        safe_name = name.replace(' ', '_').replace('-', '_').lower()
        save_path = output_dir / f"{safe_name}_performance.png"
        plot_individual_strategy(name, returns, all_metrics[name], save_path)
    
    print(f"\n✓ All plots generated successfully!")
    print(f"📁 Location: {output_dir.absolute()}")

if __name__ == "__main__":
    run_individual_plots()
