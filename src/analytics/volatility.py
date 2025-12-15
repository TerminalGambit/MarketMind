import pandas as pd
import numpy as np
from arch import arch_model
from pathlib import Path

class VolatilityModel:
    """
    Analytics Component: Volatility Modeling.
    Uses GARCH(1,1) to estimate and forecast conditional volatility.
    """
    def __init__(self):
        pass

    def fit_garch(self, returns: pd.Series, p=1, q=1):
        """
        Fits a GARCH(p,q) model to the returns series.
        Returns:
            res: The fitted model result object.
        """
        # Rescale returns to percentage for numerical stability in optimizer
        # scaling by 100 often helps GARCH convergence
        returns_scaled = returns * 100
        
        # 'GARCH' model with Constant Mean
        am = arch_model(returns_scaled, vol='Garch', p=p, q=q, dist='Normal')
        res = am.fit(update_freq=0, disp='off')
        
        return res

    def get_conditional_volatility(self, returns: pd.Series):
        """
        Returns the conditional volatility (sigma) from the fitted model.
        """
        res = self.fit_garch(returns)
        # Rescale back (since we scaled input by 100, vol is also scaled by 100)
        cond_vol = res.conditional_volatility / 100
        return cond_vol

    def forecast_volatility(self, returns: pd.Series, horizon=5):
        """
        Forecasts volatility for the next H days.
        """
        res = self.fit_garch(returns)
        forecasts = res.forecast(horizon=horizon)
        # Rescale back
        var_forecast = forecasts.variance.dropna().iloc[-1]
        vol_forecast = np.sqrt(var_forecast) / 100
        return vol_forecast

if __name__ == "__main__":
    # Test
    np.random.seed(42)
    # Synthetic returns with volatility clustering
    T = 1000
    w = np.random.normal(0, 1, T)
    eps = np.zeros(T)
    sigma_sq = np.zeros(T)
    
    alpha0 = 0.1
    alpha1 = 0.1
    beta1 = 0.8
    
    for t in range(1, T):
        sigma_sq[t] = alpha0 + alpha1*(eps[t-1]**2) + beta1*sigma_sq[t-1]
        eps[t] = w[t] * np.sqrt(sigma_sq[t])
        
    returns = pd.Series(eps, index=pd.date_range("2020-01-01", periods=T))
    
    vol = VolatilityModel()
    print("Fitting GARCH(1,1)...")
    cvol = vol.get_conditional_volatility(returns)
    print("Conditional Volatility (Last 5):")
    print(cvol.tail())
    
    fcast = vol.forecast_volatility(returns, horizon=5)
    print("\nForecast (Next 5 Days):")
    print(fcast)
