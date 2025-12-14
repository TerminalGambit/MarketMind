import json
import pandas as pd
from pathlib import Path
import ollama
from datetime import datetime

class SentimentEngine:
    """
    Phase 4: LLM-Based Sentiment Analysis.
    Uses local Ollama (ministral-3:8b) to reason about headlines.
    Returns: Alpha Signal (-1.0 to 1.0) and Reasoning.
    """
    def __init__(self, silver_path: str = "market_mind/data/silver", model: str = "ministral-3:8b"):
        self.silver_path = Path(silver_path)
        self.model = model
        print(f"Initialized LLM Sentiment Engine with {self.model}")
        
    def load_processed_news(self):
        files = list(self.silver_path.glob("news_processed_*.jsonl"))
        if not files:
            print("No processed news found. Run nlp.py first.")
            return []
            
        latest = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading news from: {latest}")
        
        data = []
        with open(latest, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)

    def analyze_headline(self, text: str) -> float:
        """
        Ask LLM to rate sentiment.
        """
        prompt = f"""
        Act as a senior financial analyst. 
        Evaluate the sentiment of this news headline regarding the company's stock price.
        Headline: "{text}"
        
        Output strictly a JSON object with a single key 'score' between -1.0 (Very Negative) and 1.0 (Very Positive).
        Example: {{"score": 0.5}}
        """
        
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            content = response['message']['content']
            
            # Simple parsing: find the JSON
            # Ideally use pydantic validation or structured output if supported, 
            # but for 8b model, JSON instruction usually works.
            # Let's clean markdown
            content = content.replace("```json", "").replace("```", "").strip()
            result = json.loads(content)
            return float(result.get('score', 0.0))
        except Exception as e:
            print(f"LLM Error on '{text}': {e}")
            return 0.0

    def analyze(self):
        print("Running LLM Sentiment Analysis (this may take time)...")
        df = self.load_processed_news()
        if df.empty:
            print("No data to analyze.")
            return

        # Limited debugging: run on first 5-10 items only if dataset is huge?
        # For now, let's run on all (assuming small dataset for demo)
        print(f"Analyzing {len(df)} headlines...")
        
        # Determine lowercase 'ticker' column from nlp.py
        ticker_col = 'ticker' if 'ticker' in df.columns else 'Ticker'
        
        scores = []
        for i, row in df.iterrows():
            score = self.analyze_headline(row['clean_text'])
            print(f"[{row[ticker_col]}] {row['clean_text'][:30]}... -> {score}")
            scores.append(score)
            
        df['sentiment_score'] = scores
        
        # Parse Dates (Mock logic)
        if 'scraped_at' in df.columns:
            today = datetime.now().strftime("%Y-%m-%d")
            df['Date'] = pd.to_datetime(df['scraped_at'], errors='coerce').dt.strftime("%Y-%m-%d")
            df['Date'] = df['Date'].fillna(today)
        else:
             df['Date'] = datetime.now().strftime("%Y-%m-%d")

        # Aggregate per Ticker per Date
        daily_sentiment = df.groupby([ticker_col, 'Date'])['sentiment_score'].mean().reset_index()
        daily_sentiment.rename(columns={'sentiment_score': 'sentiment'}, inplace=True)
        # Ensure Ticker is Capitalized
        daily_sentiment.rename(columns={ticker_col: 'Ticker'}, inplace=True)
        
        # Save Parquet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.silver_path / f"market_sentiment_{timestamp}.parquet"
        daily_sentiment.to_parquet(output_path)
        
        print(f"Sentiment Analysis Complete.")
        print(f"Saved signal to {output_path}")
        print(daily_sentiment.head())
        
        return daily_sentiment

if __name__ == "__main__":
    eng = SentimentEngine()
    eng.analyze()
