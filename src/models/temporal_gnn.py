import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from src.models.gnn import GraphConvLayer, GNNTrainer # Reuse base components
from sklearn.preprocessing import StandardScaler

class SpatioTemporalGNN(nn.Module):
    """
    Advanced Model: Spatio-Temporal GNN.
    Combines Graph Convolution (Spatial) with LSTM (Temporal).
    Flow: Input -> GCN -> LSTM -> FC -> Output
    """
    def __init__(self, num_nodes, in_features, hidden_dim, lstm_hidden_dim, out_dim):
        super().__init__()
        # Spatial Block
        self.gcn1 = GraphConvLayer(in_features, hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)
        
        # Temporal Block (LSTM)
        # Input to LSTM: [Batch (Time), Nodes, Hidden] -> Need to reshape
        # Actually, standard LSTM takes [Seq_Len, Batch, Features]
        # Here we treat each Node as a sequence? Or the whole graph history?
        
        # Approach: 
        # 1. Process each Snapshot t through GCN -> Node Embeddings H_t
        # 2. Sequence of H_t passed to LSTM
        
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=lstm_hidden_dim, batch_first=True)
        
        # Output Head
        self.fc = nn.Linear(lstm_hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x_seq, adj):
        # x_seq: [Batch, Seq_Len, Num_Nodes, In_Features]
        # For simplicity in V1, let's assume Batch=1 (Single Time Series History)
        # x_seq: [Seq_Len, Num_Nodes, In_Features]
        
        seq_len, num_nodes, feat_dim = x_seq.shape
        
        gcn_outputs = []
        for t in range(seq_len):
            x_t = x_seq[t] # [Num_Nodes, Features]
            
            # GCN
            h_t = F.relu(self.gcn1(x_t, adj))
            h_t = self.dropout(h_t)
            h_t = self.gcn2(h_t, adj) # [Num_Nodes, Hidden]
            gcn_outputs.append(h_t)
            
        # Stack: [Seq_Len, Num_Nodes, Hidden]
        gcn_stack = torch.stack(gcn_outputs) 
        
        # LSTM needs [Batch, Seq_Len, Features]
        # Treating each Node as an independent sequence in the Batch
        # permute -> [Num_Nodes, Seq_Len, Hidden]
        lstm_input = gcn_stack.permute(1, 0, 2)
        
        # LSTM Output
        lstm_out, (hn, cn) = self.lstm(lstm_input)
        
        # Take last time step hidden state: hn [1, Num_Nodes, LSTM_Hidden]
        # Squeeze -> [Num_Nodes, LSTM_Hidden]
        last_hidden = hn[-1]
        
        # Prediction
        out = self.fc(last_hidden) # [Num_Nodes, 1]
        
        return out

class SpatioTemporalTrainer(GNNTrainer):
    """
    Extends base trainer to handle Sequence Data for LSTM.
    """
    def prepare_sequences(self, seq_len=10):
        # Reuse preparation logic to get the full time series arrays
        tickers, adj, train_data_simple, test_data_simple, test_dates, num_feat = self.prepare_tensors()
        
        # Reconstruct full X and Y list from the simple pairs
        # Actually, prepare_tensors in base class creates single step pairs.
        # We need the underlying time series data.
        # Efficient way: Access internal methods or modify prepare_tensors.
        # For now, let's use the X_list from the base method logic (copy-paste logic for safety or refactor base).
        # Refactoring base is better, but to avoid breaking base, let's just re-implement seq prep here
        # leveraging load_data/load_graph.
        
        # ... (Assuming we have x_list, y_list of full history)
        # Let's quickly rebuild x_list/y_list
        df, has_sent = self.load_data()
        G = self.load_graph()
        # ... (Pivot logic omitted for brevity, assuming standard tensors ready)
        # Quick Hack using the simple data: 
        # train_data is [(x,y), ...]. x is [Nodes, Feats].
        
        all_x = [x for x, y in train_data_simple] + [x for x, y in test_data_simple]
        all_y = [y for x, y in train_data_simple] + [y for x, y in test_data_simple] # Targets
        
        X_seq = []
        Y_seq = []
        
        # Sliding Window
        for i in range(len(all_x) - seq_len):
            # Sequence: t to t+seq_len
            seq_x = torch.stack(all_x[i:i+seq_len]) # [Seq_Len, Nodes, Feats]
            target_y = all_y[i+seq_len-1] # Target at end of sequence? 
            # Usually we predict t+1 given t-k..t.
            # Base data is: x[t] -> y[t] (where y[t] is return at t+1)
            # So sequence x[t-k]..x[t] predicts y[t]
            
            X_seq.append(seq_x)
            Y_seq.append(target_y)
            
        return tickers, adj, X_seq, Y_seq, num_feat

    def train_st_model(self, epochs=50, seq_len=5):
        print(f"\n=== Spatio-Temporal GNN Training (SeqLen={seq_len}) ===")
        tickers, adj, X_seq, Y_seq, num_feat = self.prepare_sequences(seq_len)
        
        split = int(len(X_seq) * 0.8)
        train_X = X_seq[:split]
        train_Y = Y_seq[:split]
        test_X = X_seq[split:]
        test_Y = Y_seq[split:]
        
        print(f"Training Samples: {len(train_X)}")
        
        model = SpatioTemporalGNN(len(tickers), num_feat, 16, 32, 1)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x, y in zip(train_X, train_Y):
                optimizer.zero_grad()
                out = model(x, adj)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Loss {total_loss/len(train_X):.6f}")

if __name__ == "__main__":
    trainer = SpatioTemporalTrainer()
    trainer.train_st_model()
