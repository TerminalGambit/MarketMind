import yfinance as yf
import pandas as pd
import numpy as np
import networkx as nx
import torch
import ollama
import sklearn
import seaborn as sns
import matplotlib.pyplot as plt
from selenium import webdriver

def checks():
    print("Checking imports...")
    print(f"Pandas version: {pd.__version__}")
    print(f"PyTorch version: {torch.__version__}")
    print("Imports successful.\n")

    print("Step 1: Testing Data Fetching (yfinance)...")
    ticker = "AAPL"
    try:
        data = yf.download(ticker, period="1mo", progress=False)
        if not data.empty:
            print(f"Successfully fetched {len(data)} rows for {ticker}")
            print(data.head(2))
        else:
            print("Fetched data was empty.")
    except Exception as e:
        print(f"Error fetching data: {e}")

    print("\nStep 2: Testing Graph Creation (networkx)...")
    G = nx.Graph()
    G.add_edge("AAPL", "Technology")
    G.add_edge("NVDA", "Technology")
    print(f"Graph created with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    
    print("\nStep 3: checking local environment")
    if torch.backends.mps.is_available():
        print("Apple Metal Performance Shaders (MPS) is available for PyTorch acceleration!")
    else:
        print("MPS not detected. Running on CPU.")

if __name__ == "__main__":
    checks()
