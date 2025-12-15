import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from pathlib import Path

class AdvancedOptimizer:
    """
    Analytics Component: Advanced Portfolio Construction.
    Implements Hierarchical Risk Parity (HRP) to address MVO limitations (instability).
    """
    def __init__(self):
        pass

    def get_hrp_weights(self, returns: pd.DataFrame) -> pd.Series:
        """
        Computes HRP weights for a given dataframe of returns.
        Steps:
        1. Clustering (Linkage)
        2. Quasi-Diagonalization (Sorting)
        3. Recursive Bisection (Allocation)
        """
        cov = returns.cov()
        corr = returns.corr()
        
        # 1. Clustering
        # Distance Matrix d(i,j) = sqrt(2*(1-rho))
        dist = np.sqrt(2 * (1 - corr))
        dist_condensed = squareform(dist, checks=False) # Enhance robustness/speed
        
        # Linkage: Single is standard for HRP
        link = sch.linkage(dist_condensed, 'single')
        
        # 2. Quasi-Diagonalization
        # Sort indices so similar assets are adjacent
        sorted_ix = self.get_quasi_diag(link)
        sorted_ix = corr.index[sorted_ix].tolist()
        
        # Reorder covariance
        hrp_cov = cov.loc[sorted_ix, sorted_ix]
        
        # 3. Recursive Bisection
        weights = self.get_rec_bisection(hrp_cov, sorted_ix)
        return weights

    def get_quasi_diag(self, link):
        """
        Sorts the linkage matrix so clustered items are adjacent in the list.
        """
        link = link.astype(int)
        sort_ix = pd.Series([link[-1, 0], link[-1, 1]])
        num_items = link[-1, 3] # Total items
        
        while sort_ix.max() >= num_items:
            sort_ix.index = range(0, sort_ix.shape[0] * 2, 2) # Make space
            df0 = sort_ix[sort_ix >= num_items] # Clusters to expansion
            i = df0.index
            j = df0.values - num_items
            sort_ix[i] = link[j, 0] # Left child
            df0 = pd.Series(link[j, 1], index=i + 1) # Right child
            sort_ix = pd.concat([sort_ix, df0])
            sort_ix = sort_ix.sort_index()
            sort_ix.index = range(sort_ix.shape[0])
            
        return sort_ix.tolist()

    def get_rec_bisection(self, cov, sort_ix):
        """
        Recursively allocates weight to clusters based on Inverse Variance.
        """
        w = pd.Series(1.0, index=sort_ix)
        c_items = [sort_ix] # List of clusters (initially one big cluster)
        
        while len(c_items) > 0:
            # Pop the first cluster split
            c_items = [i[j:k] for i in c_items for j, k in ((0, len(i) // 2), (len(i) // 2, len(i))) if len(i) > 1]
            
            for i in range(0, len(c_items), 2): # Iterate pairs of sub-clusters
                c1 = c_items[i] 
                c2 = c_items[i+1]
                
                # Variance of the cluster/sub-portfolio
                v1 = self.get_cluster_var(cov, c1)
                v2 = self.get_cluster_var(cov, c2)
                
                # Split factor alpha = 1 - v1 / (v1 + v2) 
                # (Lower variance gets higher weight)
                alpha = 1 - v1 / (v1 + v2)
                
                w[c1] *= alpha
                w[c2] *= 1 - alpha
                
        return w

    def get_cluster_var(self, cov, c_items):
        """
        Computes the variance of a cluster, assuming equal risk contribution within it.
        """
        cov_slice = cov.loc[c_items, c_items]
        ivp = 1. / np.diag(cov_slice)
        ivp /= ivp.sum()
        w_vec = ivp.reshape(-1, 1)
        var = np.dot(np.dot(w_vec.T, cov_slice), w_vec)[0, 0]
        return var

    def get_black_litterman_weights(self, returns: pd.DataFrame, market_caps: pd.Series = None, 
                                   view_dict: dict = None, confidences: list = None, risk_aversion: float = 2.5) -> pd.Series:
        """
        Computes Black-Litterman weights.
        Combines Market Equilibrium (Prior) with Investor Views (Likelihood) to form Posterior.
        
        Args:
            returns: Asset returns dataframe.
            market_caps: Series of market caps for equilibrium weights. If None, assumes equal weight equilibrium.
            view_dict: Dictionary of {ticker: expected_return}. E.g. {'AAPL': 0.05} (5% excess return).
            confidences: List of confidence levels for views [0, 1]. Defaults to high confidence if None.
            risk_aversion: Lambda parameter for market equilibrium.
            
        Returns:
            pd.Series: Optimal weights based on Posterior estimates.
        """
        cov = returns.cov()
        tickers = returns.columns
        
        # 1. Market Equilibrium (Prior)
        if market_caps is None:
            # Fallback: Equal weights as market equilibrium
            w_mkt = pd.Series(1/len(tickers), index=tickers)
        else:
            w_mkt = market_caps / market_caps.sum()
            # Align with tickers
            w_mkt = w_mkt.reindex(tickers).fillna(0)
            
        # Implied Equilibrium Returns (Pi = lambda * Sigma * w_mkt)
        pi = risk_aversion * cov.dot(w_mkt)
        
        # 2. Views (P, Q)
        if not view_dict:
            # No views -> Return Market Weights (or MVO on Pi)
            # If we trust equilibrium exactly, optimal is w_mkt.
            return w_mkt
            
        K = len(view_dict)
        P = np.zeros((K, len(tickers)))
        Q = np.zeros(K)
        
        for i, (ticker, ret) in enumerate(view_dict.items()):
            if ticker in tickers:
                idx = tickers.get_loc(ticker)
                P[i, idx] = 1
                Q[i] = ret
                
        # 3. Uncertainty / Omega
        # Simple scalar specification: tau * P * Sigma * P'
        tau = 0.025
        omega = np.dot(np.dot(P, (tau * cov)), P.T)
        if confidences:
            # Adjust diagonals based on confidence (higher conf -> lower variance)
            # This is a heuristic integration of confidence
            omega = omega * np.diag([(1 - c)/c if c > 0 else 1e6 for c in confidences])
            
        # 4. Posterior Estimates
        # BL Master Formula for E[R]
        # inv(tau*Sigma) is needed. Ideally compute via inverse of (inv + ...)
        
        sigma_inv = np.linalg.inv(cov)
        omega_inv = np.linalg.inv(omega)
        
        # Term 1: [(tau*Sigma)^-1 + P' Omega^-1 P]^-1
        # Simplify: M = (tau*cov)
        M_inv = np.linalg.inv(tau * cov)
        
        term1 = np.linalg.inv(M_inv + np.dot(np.dot(P.T, omega_inv), P))
        
        # Term 2: [(tau*Sigma)^-1 * Pi + P' Omega^-1 Q]
        term2 = np.dot(M_inv, pi) + np.dot(np.dot(P.T, omega_inv), Q)
        
        posterior_ret = np.dot(term1, term2)
        posterior_ret = pd.Series(posterior_ret, index=tickers)
        
        # Posterior Covariance (Optional for risk measure, usually MVO uses original Cov or updated)
        # S_post = cov + term1 (Posterior uncertainty)
        # For Optimization, we use Post Ret + Original Cov (common practice) or Post Cov
        
        # 5. Optimization (Max Sharpe on Posterior)
        # We can reuse standard MVO logic, or closed form if unconstrained.
        # Let's simple Mean-Variance with sum=1, long-only.
        # w* = (Sigma^-1 * mu) / (1' * Sigma^-1 * mu) for max return/risk
        
        # Using analytical solution for Tangency Portfolio (unconstrained long/short)
        # w = Sigma^-1 * mu
        # But we want Long Only usually.
        # Let's call a helper or use a simple solver. 
        # For V1 speed, let's use the closed form Tangency and clip? No, that's bad.
        # Let's use CVXPY for robustness if available, else fallback.
        
        try:
            import cvxpy as cp
            w = cp.Variable(len(tickers))
            ret = posterior_ret.values
            risk = cp.quad_form(w, cov.values)
            
            # Maximize Utility: w'mu - (gamma/2) * w'Sigma w  (Standard quadratic utility)
            # Or Maximize Return subject to Risk <= target.
            # Let's Maximize Utility matches BL assumptions better.
            obj = cp.Maximize(w @ ret - (risk_aversion/2) * risk)
            constraints = [cp.sum(w) == 1, w >= 0]
            
            prob = cp.Problem(obj, constraints)
            prob.solve()
            return pd.Series(w.value, index=tickers)
            
        except ImportError:
            print("CVXPY not found, falling back to heuristic clipping.")
            # Fallback: Inverse Variance weighted by returns?
            return w_mkt # Fallback

    def get_cvar_weights(self, returns: pd.DataFrame, alpha: float = 0.95) -> pd.Series:
        """
        Computes weights minimizing Conditional Value at Risk (CVaR/Expected Shortfall).
        """
        import cvxpy as cp
        T, N = returns.shape
        mu = returns.mean().values
        ret_matrix = returns.values
        
        # Scale returns for numerical stability
        scale = 1000.0
        ret_matrix_scaled = ret_matrix * scale
        mu_scaled = mu * scale
        
        # Variables
        w = cp.Variable(N)
        gamma = cp.Variable()
        z = cp.Variable(T)
        
        # Objective (Scaled)
        cvar_term = (1 / ((1 - alpha) * T)) * cp.sum(z)
        obj = cp.Minimize(gamma + cvar_term)
        
        constraints = [
            z >= 0,
            z >= -ret_matrix_scaled @ w - gamma,
            cp.sum(w) == 1,
            w >= 0
        ]
        
        prob = cp.Problem(obj, constraints)
        
        # Try primary solver, then fallback
        try:
            prob.solve()
        except:
            try:
                prob.solve(solver=cp.SCS)
            except:
                print("CVaR Optimization failed with all solvers.")
                return pd.Series(1/N, index=returns.columns)
        
        if prob.status not in ['optimal', 'optimal_inaccurate']:
            print(f"CVaR Optimization failed status: {prob.status}")
            return pd.Series(1/N, index=returns.columns)
            
        return pd.Series(w.value, index=returns.columns)

