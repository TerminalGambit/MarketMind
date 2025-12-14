from pyvis.network import Network
import networkx as nx
from pathlib import Path
import json
import sys
import os

# Ensure src is in path to import analytics
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from analytics.math_core import MathCore

class GraphVisualizer:
    """
    Visualization Component: Interactive Network Graph.
    Uses PyVis to generate a D3-like HTML visualization of the Market Mind.
    Integrated with MathCore for Spectral Clustering (Regime Detection).
    """
    def __init__(self, gold_path: str = None):
        if gold_path:
            self.gold_path = Path(gold_path)
        else:
            # Resolve relative to this script: src/visualization/../../data/gold
            self.gold_path = Path(__file__).resolve().parent.parent.parent / "data" / "gold"
            
        self.output_path = self.gold_path / "interactive_dashboard.html"
        # Pass the resolved absolute path to MathCore so it also finds the files
        self.math_core = MathCore(gold_path=str(self.gold_path))

    def load_latest_graph(self) -> nx.Graph:
        files = list(self.gold_path.glob("market_graph_*.gexf"))
        if not files: raise FileNotFoundError("No Graph found.")
        latest = max(files, key=lambda f: f.stat().st_mtime)
        print(f"Visualizing graph: {latest}")
        return nx.read_gexf(latest)

    def generate_viz(self):
        G_nx = self.load_latest_graph()
        
        # 1. Math Analysis: Get Clusters
        print("Running Spectral Clustering for visualization colors...")
        try:
            # We use k=5 for the 5 main sectors we put in Bronze
            clusters = self.math_core.spectral_clustering(k=5)
            # Create a dict: Ticker -> ClusterID
            cluster_map = dict(zip(clusters['Ticker'], clusters['Cluster']))
        except Exception as e:
            print(f"Clustering failed: {e}. Falling back to default colors.")
            cluster_map = {}

        # Initialize PyVis Network
        # cdn_resources='in_line' fixes the "blank screen" issue by embedding JS directly
        net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white", select_menu=True, cdn_resources="in_line")
        
        # Color Palette for Clusters (Neon/Cyberpunk theme)
        # 0: Cyber Blue, 1: Neon Green, 2: Hot Pink, 3: Bright Yellow, 4: Electric Purple
        cluster_colors = ["#00ccff", "#00ff88", "#ff00cc", "#ffff00", "#bf00ff"]
        
        print("Adding nodes...")
        for node, attr in G_nx.nodes(data=True):
            node_type = attr.get('type', 'Unknown')
            
            # Styling based on Type & Cluster
            if node_type == 'Company':
                if node in cluster_map:
                    cid = cluster_map[node]
                    color = cluster_colors[cid % len(cluster_colors)]
                    title = f"Ticker: {node}\nCluster: {cid}"
                else:
                    color = "#ffffff" # Fallback white
                    title = f"Ticker: {node}"
                size = 25
            elif node_type == 'Concept':
                color = "#bbbbbb" # Grey
                size = 15
                title = f"Concept: {node}"
            else:
                color = "gray"
                size = 10
                title = node
                
            net.add_node(node, label=node, title=title, color=color, size=size)
            
        print("Adding edges...")
        for u, v, attr in G_nx.edges(data=True):
            edge_type = attr.get('type', 'semantic')
            weight = attr.get('weight', 1.0)
            
            if edge_type == 'correlation':
                color = "#ff4444" # Red for correlation links
                # Scale width but cap it
                width = min(weight * 3, 5)
                title = f"Correlation: {weight:.2f}"
                dashed = False
            else:
                color = "#aaaaaa" # Semantic
                width = 1
                title = "Semantic Link"
                dashed = True
                
            net.add_edge(u, v, title=title, color=color, width=width, dashes=dashed)
            
        # Physics Options
        net.barnes_hut(gravity=-2000, central_gravity=0.3, spring_length=200, spring_strength=0.05, damping=0.09)
        # Enable UI controls for physics
        net.show_buttons(filter_=['physics'])
        
        # Save
        print(f"Saving interactive graph to {self.output_path}...")
        net.save_graph(str(self.output_path))
        print("Done! Open 'interactive_dashboard.html' to explore.")

if __name__ == "__main__":
    viz = GraphVisualizer()
    viz.generate_viz()
