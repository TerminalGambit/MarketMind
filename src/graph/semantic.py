import ollama
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

class SemanticExtractor:
    """
    Gold Layer Component: Extracts semantic relationships using Ollama.
    - Input: Silver News JSONL
    - Output: Gold Graph Edges (Ticker -> Concept)
    """
    def __init__(self, silver_path: str = "market_mind/data/silver", gold_path: str = "market_mind/data/gold", model: str = "llama3.1"):
        self.silver_path = Path(silver_path)
        self.gold_path = Path(gold_path)
        self.gold_path.mkdir(parents=True, exist_ok=True)
        self.model = model

    def load_silver_news(self) -> List[Dict]:
        files = list(self.silver_path.glob("news_processed_*.jsonl"))
        if not files: return []
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading silver news from: {latest_file}")
        
        data = []
        with open(latest_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    def extract_concepts(self, news_items: List[Dict]) -> List[Dict]:
        """
        Uses LLM to extract key business concepts from headlines.
        """
        extracted_edges = []
        
        print(f"Extracting concepts from {len(news_items)} headers using {self.model}...")
        
        for item in news_items:
            ticker = item['ticker']
            text = item['clean_text']
            
            prompt = f"""
            Analyze this financial headline: "{text}"
            Identify the key business concept or topic (e.g., "Earnings Beat", "Antitrust", "AI Partnership", "FDA Approval").
            Return ONLY a valid JSON object with a single key 'concept'. Example: {{"concept": "Earnings Beat"}}
            """
            
            try:
                response = ollama.chat(model=self.model, messages=[
                    {'role': 'user', 'content': prompt},
                ])
                res_content = response['message']['content']
                
                # Cleanup potential markdown ticks if model adds them
                res_content = res_content.replace('```json', '').replace('```', '').strip()
                
                concept_json = json.loads(res_content)
                concept = concept_json.get('concept', 'General News')
                
                extracted_edges.append({
                    "source": ticker,
                    "target": concept,
                    "type": "related_to",
                    "source_text": text,
                    "weight": 1.0 # Semantic link default weight
                })
                print(f"  [{ticker}] -> [{concept}]")
                
            except Exception as e:
                print(f"  Error processing '{text}': {e}")
                
        return extracted_edges

    def save_edges(self, edges: List[Dict]):
        if not edges: return
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.gold_path / f"semantic_edges_{timestamp}.jsonl"
        
        with open(file_path, 'w') as f:
            for edge in edges:
                f.write(json.dumps(edge) + "\n")
        print(f"Saved {len(edges)} semantic edges to {file_path}")

    def run(self):
        news = self.load_silver_news()
        if not news:
            print("No news to process.")
            return
        
        # Limit for demo speed if many items
        if len(news) > 20:
            print("Limiting to 20 items for demo...")
            news = news[:20]
            
        edges = self.extract_concepts(news)
        self.save_edges(edges)

if __name__ == "__main__":
    extractor = SemanticExtractor()
    extractor.run()
