import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict

class TextProcessor:
    """
    Silver Layer Text Processor: Cleans and normalizes news headlines.
    - Removes duplicate whitespace/newlines
    - Basic tokenization prep (lowercasing, etc - though for LLMs we keep case usually)
    - Dedupes articles by title
    """
    def __init__(self, bronze_path: str = "market_mind/data/bronze", silver_path: str = "market_mind/data/silver"):
        self.bronze_path = Path(bronze_path)
        self.silver_path = Path(silver_path)
        self.silver_path.mkdir(parents=True, exist_ok=True)

    def load_latest_bronze_news(self) -> List[Dict]:
        """Loads latest news jsonl file."""
        files = list(self.bronze_path.glob("news_headlines_*.jsonl"))
        if not files:
            print("No news data found.")
            return []
            
        latest_file = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading raw news from: {latest_file}")
        
        data = []
        with open(latest_file, 'r') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except:
                    continue
        return data

    def clean_text(self, text: str) -> str:
        """Basic text cleaning."""
        if not text:
            return ""
        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process(self):
        raw_news = self.load_latest_bronze_news()
        if not raw_news:
            return

        processed_news = []
        seen_titles = set()

        for item in raw_news:
            title = self.clean_text(item.get('title', ''))
            
            # Deduplication
            if title in seen_titles or not title:
                continue
            seen_titles.add(title)
            
            # Enriched item
            processed_item = {
                "ticker": item.get('ticker'),
                "original_title": item.get('title'),
                "clean_text": title,
                "source_link": item.get('link'),
                "scraped_at": item.get('scraped_at'),
                "processed_at": datetime.now().isoformat()
            }
            # Simple heuristic tag for "Analysis" vs "News" -> useful for Graph Edge types later
            if "upgrade" in title.lower() or "price target" in title.lower():
                processed_item["type"] = "Analyst Rating"
            else:
                processed_item["type"] = "General News"
                
            processed_news.append(processed_item)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.silver_path / f"news_processed_{timestamp}.jsonl"
        
        with open(file_path, 'w') as f:
            for entry in processed_news:
                f.write(json.dumps(entry) + "\n")
                
        print(f"Saved {len(processed_news)} processed news items to {file_path}")
        if processed_news:
            print(f"Sample: {processed_news[0]['clean_text']}")

if __name__ == "__main__":
    tp = TextProcessor()
    tp.process()
