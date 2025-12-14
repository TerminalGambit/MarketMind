import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import networkx as nx
from pathlib import Path

class GraphConvLayer(nn.Module):
    """
    Custom Graph Convolution Layer (GCN) from scratch.
    H' = ReLU( D^-0.5 * A * D^-0.5 * H * W )
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, x, adj):
        # x: Node Features [Num_Nodes, In_Features]
        # adj: Normalized Adjacency Matrix [Num_Nodes, Num_Nodes]
        
        # 1. Message Passing (Aggregating neighbor information)
        # support = H * W
        support = self.linear(x) 
        
        # 2. Aggregation
        # output = A * support
        output = torch.mm(adj, support)
        
        return output

class MarketGNN(nn.Module):
    def __init__(self, num_nodes, in_features, hidden_dim, out_dim):
        super().__init__()
        self.gcn1 = GraphConvLayer(in_features, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x, adj):
        x = F.relu(self.gcn1(x, adj))
        x = self.dropout(x)
        x = self.gcn2(x, adj)
        # Regression output (predicting next day return)
        return x

class GNNTrainer:
    def __init__(self, gold_path: str = "market_mind/data/gold", silver_path: str = "market_mind/data/silver"):
        self.gold_path = Path(gold_path)
        self.silver_path = Path(silver_path)

    def load_graph(self):
        files = list(self.gold_path.glob("market_graph_*.gexf"))
        latest_graph = max(files, key=lambda f: f.stat().st_mtime)
        return nx.read_gexf(latest_graph)

    def load_data(self):
        # 1. Load Technical Features
        files = list(self.silver_path.glob("market_features_v1_*.parquet"))
        if not files:
            print("Features not found.")
            return None, None
            
        latest_feat = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Loading features from: {latest_feat}")
        df_feat = pd.read_parquet(latest_feat)
        
        # 2. Load Sentiment (Optional)
        sent_files = list(self.silver_path.glob("market_sentiment_*.parquet"))
        if sent_files:
            latest_sent = max(sent_files, key=lambda f: f.stat().st_mtime)
            print(f"Loading sentiment from: {latest_sent}")
            df_sent = pd.read_parquet(latest_sent)
            
            # Merge Sentiment into Features
            # Features index: Date. Cols: MultiIndex (Ticker, Metric) OR Long fmt?
            # features.py saves in Long format with Ticker column? 
            # Let's check features.py output. It was MultiIndex columns in the dataframe but saved as parquet.
            # Usually parquet saves as flat table.
            # prepare_tensors logic assumes Long format:
            # "df = df.reset_index()"
            
            # Let's assume load_data returns the raw dataframe.
            # We need to merge them.
            
            # Reset indexes to be safe
            if 'Date' not in df_feat.columns and df_feat.index.name == 'Date':
                df_feat = df_feat.reset_index()
            # Ensure Date is datetime or string consistent
            df_feat['Date'] = pd.to_datetime(df_feat['Date']).dt.strftime('%Y-%m-%d')
            
            df_sent['Date'] = pd.to_datetime(df_sent['Date']).dt.strftime('%Y-%m-%d')
            
            # Merge left (Features are primary)
            # Sentiment might be sparse. FillNA 0 (Neutral).
            df_merged = pd.merge(df_feat, df_sent, on=['Ticker', 'Date'], how='left')
            df_merged['sentiment'] = df_merged['sentiment'].fillna(0)
            
            return df_merged, True
        else:
            print("No sentiment data found.")
            return df_feat, False

    def prepare_tensors(self, split_date="2025-06-01"):
        G = self.load_graph()
        # Updated load_data now returns: df, has_sentiment
        # Note: I changed the signature of load_data above, need to catch that.
        # But wait, original load_data returned (df, is_enriched).
        # Modified load_data returns (df, has_sentiment).
        # Let's align.
        
        df, has_sentiment = self.load_data()
        if df is None:
            print("Data loading failed.")
            return [], None, [], [], [], 0

        is_enriched = True # Since we are loading V1 features
        
        # Get list of tickers from Graph that exist in Data
        # (Graph might have concepts, we only want stocks)
        graph_nodes = [n for n in G.nodes()]
        
        if is_enriched:
            # Enriched is Long format: Index=Date, Cols=Ticker, Close, RSI...
            # We need to Pivot to get [Date, Ticker, Feature]
            # Reset index to make Date a column
            # Ensure Date column exists if not in index
            if 'Date' not in df.columns:
                 df = df.reset_index()
            # Pivot
            # We want a 3D Tensor: [Time, Node, Feature]
            # Let's align nodes first.
            available_tickers = df['Ticker'].unique()
            # Interact with Graph nodes
            valid_tickers = [t for t in graph_nodes if t in available_tickers]
            valid_tickers.sort() # Ensure consistent order
            
            df = df[df['Ticker'].isin(valid_tickers)]
            
            # Pivot for features
            # Values: RSI, MACD, BB_Pct, Close (compute return from close)
            # Actually, let's just use what we have. 
            # We need Returns as Target.
            df = df.sort_values(['Ticker', 'Date'])
            df['Return'] = df.groupby('Ticker')['Close'].pct_change().fillna(0)
            
            # Pivot tables
            # Added 'sentiment' to metrics if available
            metrics = ['Return', 'RSI', 'MACD', 'BB_Pct']
            if has_sentiment and 'sentiment' in df.columns:
                metrics.append('sentiment')
                print("Including Sentiment Feature in GNN input.")
            
            pivot_dict = {}
            for m in metrics:
                pivot_dict[m] = df.pivot(index='Date', columns='Ticker', values=m).fillna(0)
            
            # Common Dates
            common_dates = pivot_dict['Return'].index
            
            # Adjacency
            # For V1, we use static correlation from the Graph
            # Or rebuild simple correlation from the Train set returns?
            # Let's use the static graph topology we built in Gold
            subG = G.subgraph(valid_tickers)
            adj = nx.to_numpy_array(subG, nodelist=valid_tickers)
            adj = torch.tensor(adj, dtype=torch.float32) + torch.eye(len(valid_tickers))
            # Normalize
            deg = adj.sum(dim=1)
            d_inv_sqrt = torch.diag(torch.pow(deg, -0.5))
            adj = torch.mm(torch.mm(d_inv_sqrt, adj), d_inv_sqrt)
            
            # Build Snapshots
            x_list = []
            y_list = []
            
            # Windowing
            # Input: Features at t. Output: Return at t+1.
            dates = common_dates
            
            for i in range(len(dates) - 1):
                # Features for valid_tickers at date i
                # Shape: [Num_Nodes, Num_Features]
                feats = []
                for m in metrics:
                    # Get row for date i, all tickers ordered
                    row = pivot_dict[m].loc[dates[i], valid_tickers].values
                    feats.append(row)
                
                # Stack features: [Num_Nodes, 4]
                x_t = np.stack(feats, axis=1) 
                
                # Target: Return at t+1
                y_t = pivot_dict['Return'].loc[dates[i+1], valid_tickers].values
                
                x_list.append(torch.tensor(x_t, dtype=torch.float32))
                y_list.append(torch.tensor(y_t, dtype=torch.float32).unsqueeze(1))
            
            # Train/Test Split
            split_idx = int(len(dates) * 0.8) # Simple 80/20 split based on index
            
            train_data = list(zip(x_list[:split_idx], y_list[:split_idx]))
            test_data = list(zip(x_list[split_idx:], y_list[split_idx:]))
            test_dates = dates[split_idx+1:] # +1 because y is t+1
            
            return valid_tickers, adj, train_data, test_data, test_dates, len(metrics) # Added test_dates return

        else:
            print("Legacy mode not implemented for V1.")
            return [], None, [], [], [], 0

    def train(self, epochs=50):
        print("\n=== Market-Mind V1 GNN Training ===")
        tickers, adj, train_data, test_data, test_dates, num_features = self.prepare_tensors()
        
        if not tickers:
            print("Data preparation failed.")
            return

        print(f"Graph: {len(tickers)} nodes. Features: {num_features}.")
        print(f"Training Samples: {len(train_data)}. Test Samples: {len(test_data)}")
        
        # Model
        model = MarketGNN(num_nodes=len(tickers), in_features=num_features, hidden_dim=16, out_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        
        # Training Loop
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, y in train_data:
                optimizer.zero_grad()
                out = model(x, adj)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Avg Loss {total_loss / len(train_data):.6f}")
                
        # Evaluation & Prediction Collection
        model.eval()
        test_loss = 0
        all_preds = []
        
        with torch.no_grad():
            for x, y in test_data:
                out = model(x, adj)
                loss = criterion(out, y)
                test_loss += loss.item()
                # shape: [Num_Nodes, 1] -> [Num_Nodes]
                all_preds.append(out.flatten().numpy())
        
        print(f"\nFinal Test MSE: {test_loss / len(test_data):.6f}")
        
        # Ensure predictions match dates
        # all_preds is len(test_data), test_dates should match
        if len(all_preds) == len(test_dates):
            print("Saving Backtest Predictions...")
            # Create DataFrame: Index=Date, Cols=Tickers
            pred_df = pd.DataFrame(all_preds, index=test_dates, columns=tickers)
            output_path = self.gold_path / "backtest_predictions.parquet"
            pred_df.to_parquet(output_path)
            print(f"Saved to {output_path}")
        else:
            print(f"Warning: Dim mismatch. Preds {len(all_preds)} vs Dates {len(test_dates)}")
        
        # Latest Prediction
        latest_x, _ = test_data[-1]
        latest_pred = model(latest_x, adj)
        
        print("\nLatest Forecast (Top 5 Movers):")
        preds = pd.Series(latest_pred.flatten().detach().numpy(), index=tickers)
        print(preds.sort_values(ascending=False).head(5))

if __name__ == "__main__":
    trainer = GNNTrainer()
    trainer.train()
