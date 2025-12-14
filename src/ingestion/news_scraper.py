from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict

class NewsScraper:
    """
    Scrapes financial news using Selenium and saves to Bronze layer.
    """
    def __init__(self, bronze_path: str = "market_mind/data/bronze"):
        self.bronze_path = Path(bronze_path)
        self.bronze_path.mkdir(parents=True, exist_ok=True)
        
        # Setup Headless Chrome
        chrome_options = Options()
        chrome_options.add_argument("--headless") 
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        # Anti-detection measures (basic)
        chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
        
        self.driver = webdriver.Chrome(options=chrome_options)

    def scrape_headlines(self, tickers: List[str]) -> List[Dict]:
        """
        Scrape headlines for tickers. 
        Note: This is a template. Real scraping requires robust selectors for specific sites.
        We will use a mock example or a simple site (like Yahoo Finance) for demonstration.
        """
        results = []
        try:
            for ticker in tickers:
                print(f"Scraping news for {ticker}...")
                # Using Yahoo Finance as a generic target example
                url = f"https://finance.yahoo.com/quote/{ticker}/news"
                self.driver.get(url)
                
                # Wait for news stream to load (generic selector, might need adjustment)
                # This is fragile and depends on site structure.
                try:
                    # Updated selectors based on browser inspection (2025-12-13)
                    print(f"Loading URL: {url}")
                    WebDriverWait(self.driver, 10).until(
                        EC.presence_of_element_located((By.CSS_SELECTOR, "div.stream-items, div[id^='fin-stream']"))
                    )
                    
                    articles = self.driver.find_elements(By.CSS_SELECTOR, "a.subtle-link")
                    print(f"Found {len(articles)} potential articles for {ticker}")
                    
                    for article in articles[:10]:
                        try:
                            title_elem = article.find_element(By.CSS_SELECTOR, "h3")
                            title = title_elem.text
                            link = article.get_attribute("href")
                            
                            if "/news/" in link or "/m/" in link:
                                print(f"  - Found: {title[:30]}...")
                                results.append({
                                    "ticker": ticker,
                                    "title": title,
                                    "link": link,
                                    "scraped_at": datetime.now().isoformat()
                                })
                        except Exception:
                            continue
                            
                except Exception as e:
                    print(f"Scraping failed for {ticker}: {e}")
                    print("Attempting to generate MOCK data for educational purposes...")
                    results.append({
                        "ticker": ticker,
                        "title": f"Analyst upgrades {ticker} to Buy after earnings beat",
                        "link": f"https://finance.yahoo.com/quote/{ticker}/mock_news",
                        "scraped_at": datetime.now().isoformat(),
                        "is_mock": True
                    })
                    results.append({
                        "ticker": ticker,
                        "title": f"{ticker} faces new antitrust investigation in EU",
                        "link": f"https://finance.yahoo.com/quote/{ticker}/mock_news_2",
                        "scraped_at": datetime.now().isoformat(),
                        "is_mock": True
                    })

            if results:
                self._save_to_bronze(results)
                
            return results

        finally:
            self.quit()

    def _save_to_bronze(self, data: List[Dict]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"news_headlines_{timestamp}.jsonl"
        file_path = self.bronze_path / filename
        
        with open(file_path, 'w') as f:
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        print(f"Saved {len(data)} news items to {file_path}")

    def quit(self):
        self.driver.quit()

if __name__ == "__main__":
    scraper = NewsScraper()
    # Testing with robust tickers
    scraper.scrape_headlines(["AAPL", "NVDA"]) 
