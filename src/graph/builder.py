import networkx as nx
import pandas as pd
import json
import numpy as np
from pathlib import Path
from datetime import datetime

class GraphBuilder:
    """
    Gold Layer Component: Builds the Master Knowledge Graph.
    - Financial Nodes: Tickers
    - Concept Nodes: Business Concepts (e.g. 'AI')
    - Edges: Correlation (Statistical), Related_To (Semantic)
    """
    def __init__(self, 
                 silver_path: str = "market_mind/data/silver", 
                 gold_path: str = "market_mind/data/gold"):
        self.silver_path = Path(silver_path)
        self.gold_path = Path(gold_path)

    def load_returns(self) -> pd.DataFrame:
        files = list(self.silver_path.glob("market_returns_*.parquet"))
        if not files: return pd.DataFrame()
        latest = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading returns from: {latest}")
        return pd.read_parquet(latest)

    def load_semantic_edges(self) -> list:
        files = list(self.gold_path.glob("semantic_edges_*.jsonl"))
        if not files: return []
        latest = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading semantic edges from: {latest}")
        
        edges = []
        with open(latest, 'r') as f:
            for line in f:
                edges.append(json.loads(line))
        return edges

    def build_graph(self) -> nx.Graph:
        G = nx.Graph()
        
        # 1. Statistical Edges (Correlation)
        df_ret = self.load_returns()
        if not df_ret.empty:
            corr_matrix = df_ret.corr()
            tickers = corr_matrix.columns
            # Add nodes
            G.add_nodes_from(tickers, type="Company")
            
            print("Building statistical edges...")
            threshold = 0.5 # Correlation threshold
            count = 0
            for i in range(len(tickers)):
                for j in range(i+1, len(tickers)):
                    t1, t2 = tickers[i], tickers[j]
                    corr = corr_matrix.loc[t1, t2]
                    if abs(corr) > threshold:
                        G.add_edge(t1, t2, weight=abs(corr), type="correlation", correlation=float(corr))
                        count += 1
            print(f"  Added {count} correlation edges (>{threshold}).")

        # 2. Semantic Edges
        sem_edges = self.load_semantic_edges()
        if sem_edges:
            print("Building semantic edges...")
            for edge in sem_edges:
                src, target = edge['source'], edge['target']
                
                # Ensure nodes exist
                if not G.has_node(src): G.add_node(src, type="Company")
                if not G.has_node(target): G.add_node(target, type="Concept")
                
                G.add_edge(src, target, weight=edge['weight'], type="semantic")
            print(f"  Added {len(sem_edges)} semantic edges.")
            
            print(f"  Added {len(sem_edges)} semantic edges.")

        # 3. Sector/Industry Edges
        # Note: In a real scaled app, we'd cache this metadata. 
        # Here we fetch on fly or use a simple mapping for the V1 tickers if speed is needed.
        # For robustness, let's use a hardcoded map for our known V1 tickers to avoid API latency in this builder.
        sector_map = {
            "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "GOOGL": "Technology", "AMZN": "Consumer Cyclical",
            "META": "Technology", "TSLA": "Consumer Cyclical", "AMD": "Technology", "INTC": "Technology", "CRM": "Technology",
            "ADBE": "Technology", "CSCO": "Technology",
            "JPM": "Financial Services", "BAC": "Financial Services", "WFC": "Financial Services", "C": "Financial Services",
            "GS": "Financial Services", "MS": "Financial Services", "V": "Financial Services", "MA": "Financial Services",
            "AXP": "Financial Services", "BLK": "Financial Services",
            "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare", "MRK": "Healthcare", "ABBV": "Healthcare",
            "LLY": "Healthcare", "TMO": "Healthcare", "DHR": "Healthcare",
            "PG": "Consumer Defensive", "KO": "Consumer Defensive", "PEP": "Consumer Defensive", "COST": "Consumer Defensive",
            "WMT": "Consumer Defensive", "TGT": "Consumer Defensive", "HD": "Consumer Cyclical", "MCD": "Consumer Cyclical",
            "NKE": "Consumer Cyclical", "SBUX": "Consumer Cyclical",
            "DIS": "Communication Services", "NFLX": "Communication Services", "CMCSA": "Communication Services",
            "TMUS": "Communication Services", "VZ": "Communication Services", "T": "Communication Services",
            "XOM": "Energy", "CVX": "Energy", "BA": "Industrials", "CAT": "Industrials", "GE": "Industrials",
            "HON": "Industrials", "UPS": "Industrials", "UNP": "Industrials"
        }
        
        print("Building sector edges...")
        for node in list(G.nodes()):
            if node in sector_map:
                sector = sector_map[node]
                if not G.has_node(sector): G.add_node(sector, type="Sector")
                G.add_edge(node, sector, type="part_of", weight=1.0)
        
        return G

    def save_graph(self, G: nx.Graph):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.gold_path / f"market_graph_{timestamp}.gexf" 
        nx.write_gexf(G, path)
        print(f"Graph saved to {path} with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

if __name__ == "__main__":
    builder = GraphBuilder()
    G = builder.build_graph()
    builder.save_graph(G)
