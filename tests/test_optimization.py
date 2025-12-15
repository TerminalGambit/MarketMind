import pytest
import pandas as pd
import numpy as np
from src.analytics.portfolio_optimizer import PortfolioOptimizer
from src.analytics.advanced_optimizer import AdvancedOptimizer

@pytest.fixture
def mock_returns():
    """Generates synthetic returns for testing"""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=100)
    data = np.random.normal(0.0005, 0.01, (100, 4)) # 4 Assets
    df = pd.DataFrame(data, columns=["A", "B", "C", "D"], index=dates)
    return df

def test_mvo_initialization():
    opt = PortfolioOptimizer()
    assert opt.rf_rate == 0.04

def test_hrp_weights_sum_to_one(mock_returns):
    opt = AdvancedOptimizer()
    weights = opt.get_hrp_weights(mock_returns)
    
    assert isinstance(weights, pd.Series)
    assert len(weights) == 4
    # Check weights sum to ~1
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
    # Check weights are non-negative (long only)
    assert (weights >= 0).all()

def test_hrp_clustering(mock_returns):
    # Create two perfectly correlated assets
    mock_returns['E'] = mock_returns['A'] 
    
    opt = AdvancedOptimizer()
    weights = opt.get_hrp_weights(mock_returns)
    
    # HRP should handle collinearity without crashing
    assert len(weights) == 5
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
    
    # A and E are identical, should basically split the risk/weight of that cluster
    # They should have very similar weights
    assert np.isclose(weights['A'], weights['E'], atol=0.01)

def test_black_litterman(mock_returns):
    opt = AdvancedOptimizer()
    # View: Asset 'A' will return 10% (very high)
    views = {'A': 0.10}
    
    weights = opt.get_black_litterman_weights(mock_returns, view_dict=views)
    
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
    # A should have high weight due to bullish view
    assert weights['A'] > 0.25 # Heuristic check

def test_cvar_optimization(mock_returns):
    opt = AdvancedOptimizer()
    weights = opt.get_cvar_weights(mock_returns, alpha=0.95)
    
    assert np.isclose(weights.sum(), 1.0, atol=1e-5)
    assert (weights >= -1e-5).all() # Non-negative (allowing for float precision)

