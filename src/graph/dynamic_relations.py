import json
import networkx as nx
import ollama
from pathlib import Path
import pandas as pd
from datetime import datetime

class DynamicGraphBuilder:
    """
    Phase 5: Neural Graph Expansion.
    Uses LLM (Ollama) to extract semantic relationships from news 
    and dynamically update the Knowledge Graph topology.
    """
    def __init__(self, gold_path: str = "market_mind/data/gold", silver_path: str = "market_mind/data/silver", model: str = "ministral-3:8b"):
        self.gold_path = Path(gold_path)
        self.silver_path = Path(silver_path)
        self.model = model
        print(f"Initialized Dynamic Graph Builder with {self.model}")

    def load_graph(self):
        files = list(self.gold_path.glob("market_graph_*.gexf"))
        if not files:
            raise FileNotFoundError("No base graph found.")
        latest_graph = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading base graph: {latest_graph.name}")
        return nx.read_gexf(latest_graph), latest_graph

    def load_news(self):
        files = list(self.silver_path.glob("news_processed_*.jsonl"))
        if not files:
            print("No news found.")
            return []
        latest = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading news from: {latest.name}")
        
        data = []
        with open(latest, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def extract_relation(self, headline: str, ticker: str):
        """
        Ask LLM to find relationships involving the ticker.
        """
        # We give the ticker so the LLM knows who is the subject roughly, 
        # but the headline might mention another company.
        
        prompt = f"""
        Analyze this financial news headline: "{headline}"
        
        Does it mention a relationship between {ticker} and ANY OTHER company?
        Search for: Partnerships, lawsuits, supply chain, competition, mergers.
        
        If YES, output a valid JSON object describing the relationship.
        If NO, output null.
        
        Format:
        {{
            "target_ticker": "TICKER_SYMBOL",
            "relationship": "supplier" | "competitor" | "partner" | "legal",
            "description": "very brief quote"
        }}
        
        Example JSON: {{"target_ticker": "MSFT", "relationship": "partner", "description": "cloud deal"}}
        """
        
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            content = response['message']['content']
            
            # Cleaning
            content = content.replace("```json", "").replace("```", "").strip()
            if "null" in content.lower() and len(content) < 10:
                return None
            
            # Heuristic to find JSON start/end if there is chatter
            if "{" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                result = json.loads(json_str)
                # Validation
                if 'target_ticker' in result and result['target_ticker'] != ticker:
                    return result
            return None
        except Exception as e:
            # print(f"Extraction failed: {e}")
            return None

    def update_graph(self):
        G, _ = self.load_graph()
        news_items = self.load_news()
        
        print(f"Scanning {len(news_items)} headlines for dynamic edges...")
        
        new_edges = 0
        
        for item in news_items:
            ticker = item.get('ticker') or item.get('Ticker')
            headline = item.get('clean_text')
            
            if not ticker or not headline:
                continue
                
            relation = self.extract_relation(headline, ticker)
            
            if relation:
                target = relation['target_ticker'].upper()
                edge_type = relation['relationship']
                desc = relation['description']
                
                # Check if target exists in graph (we only link internal nodes for GNN usually, 
                # unless we want to add new nodes. For V1 scale up, let's stick to known universe 
                # or add if we are bold. Let's add ONLY if both exist to avoid pollution.)
                
                if False and target not in G.nodes():
                    # Optional: Add new node
                    G.add_node(target, type='Company', sector='Unknown')
                    print(f"  [New Node] {target}")

                if target in G.nodes():
                    print(f"  [Edge Found] {ticker} --[{edge_type}]--> {target} ({desc})")
                    
                    # Add/Update Edge
                    # MultiDiGraph? G is usually undirected or directed.
                    # Let's assume directed for "Supply" etc.
                    # But our base might be undirected correlation. 
                    # NetworkX handles mixed data in attributes.
                    
                    G.add_edge(ticker, target, type='semantic_dynamic', relation=edge_type, description=desc, weight=2.0)
                    new_edges += 1
        
        if new_edges > 0:
            print(f"Found {new_edges} new semantic edges!")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.gold_path / f"market_graph_dynamic_{timestamp}.gexf"
            nx.write_gexf(G, output_path)
            print(f"Saved dynamic graph to {output_path}")
        else:
            print("No new relationships found in the news.")

if __name__ == "__main__":
    builder = DynamicGraphBuilder()
    builder.update_graph()
