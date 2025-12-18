import pandas as pd
import numpy as np
import json
import ollama
from pathlib import Path
from datetime import datetime
from src.analytics.portfolio_optimizer import PortfolioOptimizer
from src.analytics.advanced_optimizer import AdvancedOptimizer

class AnalystAgent:
    """
    Gold Layer Component: The "Market Mind" Narrator.
    Uses Mistral to synthesize performance data and news into an analyst report.
    """
    def __init__(self, model: str = "llama3.1"):
        self.model = model
        self.data_path = Path("data")
        self.reports_path = self.data_path / "gold" / "reports"
        self.reports_path.mkdir(parents=True, exist_ok=True)

    def load_latest_news(self):
        silver_path = self.data_path / "silver"
        files = list(silver_path.glob("news_processed_*.jsonl"))
        if not files: return []
        latest = max(files, key=lambda f: f.stat().st_mtime)
        news = []
        with open(latest, 'r') as f:
            for line in f:
                news.append(json.loads(line))
        return news

    def get_market_context(self, news_items, target_date=None, window_days=5):
        """Filters news around a target date to provide context."""
        if not news_items: return "No major news headlines captured."
        
        # In a real system, we'd filter by date. For this demo, 
        # let's just take the top 10 cleaning titles.
        context = "\n".join([f"- {item['ticker']}: {item['clean_text']}" for item in news_items[:15]])
        return context

    def generate_report(self, strategy_name, returns, news_context):
        cum_ret = (1 + returns).cumprod()
        total_ret = (cum_ret.iloc[-1] - 1) * 100
        max_dd = ((returns + 1).cumprod() / (returns + 1).cumprod().cummax() - 1).min() * 100
        vol = returns.std() * np.sqrt(252) * 100
        
        prompt = f"""
        You are a Senior Quantitative Portfolio Analyst. 
        Write a brief, professional market commentary (2-3 paragraphs) for the '{strategy_name}' strategy.
        
        PERFORMANCE SUMMARY:
        - Total Period Return: {total_ret:.2f}%
        - Maximum Drawdown: {max_dd:.2f}%
        - Annualized Volatility: {vol:.2f}%
        
        RECENT MARKET HEADLINES:
        {news_context}
        
        GUIDELINES:
        - Explain how the semantic news context might have influenced the asset relationships.
        - Discuss why this specific strategy (e.g., HRP, MVO) was suitable or challenged in this environment.
        - Be insightful, not just descriptive. Mention the "Hybrid" approach of using GNNs and LLMs.
        - Format with Markdown headers.
        """
        
        print(f"Generating Analyst Report for {strategy_name} using {self.model}...")
        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            return response['message']['content']
        except Exception as e:
            return f"Error generating report: {e}"

    def run(self):
        # 1. Get backtest data (similar to run_benchmark)
        mvo_gen = PortfolioOptimizer()
        try:
            full_returns = mvo_gen.load_returns()
        except FileNotFoundError:
            print("No returns found.")
            return

        split_idx = int(len(full_returns) * 0.7)
        test_df = full_returns.iloc[split_idx:]
        
        # Generate HRP weights for the demo
        adv_opt = AdvancedOptimizer()
        hrp_weights = adv_opt.get_hrp_weights(full_returns.iloc[:split_idx])
        hrp_returns = test_df.dot(hrp_weights)
        
        # 2. Get Context
        news = self.load_latest_news()
        context = self.get_market_context(news)
        
        # 3. Narrate
        report = self.generate_report("Hierarchical Risk Parity (HRP)", hrp_returns, context)
        
        # 4. Save
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = self.reports_path / f"analyst_report_{timestamp}.md"
        with open(file_path, 'w') as f:
            f.write(report)
        
        print(f"✓ Analyst Report saved to: {file_path}")
        return file_path

if __name__ == "__main__":
    agent = AnalystAgent()
    agent.run()
