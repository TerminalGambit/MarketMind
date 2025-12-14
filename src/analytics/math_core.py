import networkx as nx
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from pathlib import Path

class MathCore:
    """
    Analytics Component: Linear Algebra & Spectral Graph Theory.
    Performs Spectral Clustering to find hidden market regimes/sectors.
    """
    def __init__(self, gold_path: str = "market_mind/data/gold"):
        self.gold_path = Path(gold_path)

    def load_graph(self) -> nx.Graph:
        files = list(self.gold_path.glob("market_graph_*.gexf"))
        if not files: raise FileNotFoundError("No GraphML/GEXF graph found.")
        latest = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading graph from: {latest}")
        return nx.read_gexf(latest)

    def spectral_clustering(self, k: int = 3):
        """
        Performs Spectral Clustering:
        1. Compute Laplacian Matrix (L = D - A)
        2. Compute Eigenvalues/Eigenvectors of L
        3. Cluster the Eigenvectors (embedding space) using K-Means
        """
        G = self.load_graph()
        
        # We focus on the 'Company' nodes for clustering, 
        # but using the full graph topology (including concepts) might be interesting.
        # For strict spectral clustering on stocks, we often just use the correlation subgraph.
        
        # Let's extract the Stock-only subgraph for pure financial clustering
        companies = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'Company']
        subG = G.subgraph(companies)
        
        print(f"Performing Spectral Clustering on {len(companies)} companies...")
        
        # 1. Adjacency Matrix (Weighted by correlation)
        adj_matrix = nx.to_numpy_array(subG, weight='weight')
        
        # 2. Laplacian Matrix (Normalized)
        # L_sym = I - D^-1/2 * A * D^-1/2
        # Scikit-learn has efficient implementation, but manually for 'Linear Algebra' demo:
        
        degrees = np.sum(adj_matrix, axis=1)
        # Avoid division by zero
        d_inv_sqrt = np.power(degrees, -0.5, where=degrees!=0)
        D_inv_sqrt = np.diag(d_inv_sqrt)
        
        I = np.eye(len(companies))
        L_sym = I - D_inv_sqrt @ adj_matrix @ D_inv_sqrt
        
        print("Laplacian Matrix shape:", L_sym.shape)
        
        # 3. Eigen Decomposition
        # We need the first k eigenvectors corresponding to the smallest eigenvalues
        eigenvals, eigenvecs = np.linalg.eigh(L_sym)
        
        # Sort indices
        idx = eigenvals.argsort()[:k]
        embedding = eigenvecs[:, idx]
        
        # 4. K-Means in Embedding Space
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(embedding)
        
        # Results
        results = pd.DataFrame({
            'Ticker': companies,
            'Cluster': labels
        })
        
        print("\nSpectral Clusters Found:")
        for cluster_id in range(k):
            members = results[results['Cluster'] == cluster_id]['Ticker'].tolist()
            print(f"  Cluster {cluster_id}: {members}")
            
        return results

if __name__ == "__main__":
    math_core = MathCore()
    math_core.spectral_clustering(k=2) # 2 clusters for small demo
