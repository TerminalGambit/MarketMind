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
        
        # 1. Advanced Math: Community Detection (Native NetworkX)
        print("Running Community Detection (Native NetworkX)...")
        try:
            from networkx.algorithms import community as nx_comm
            # Louvain (Native in NetworkX 3.x+)
            communities = nx_comm.louvain_communities(G_nx, weight='weight', seed=42)
            # Convert list of sets to node:id mapping
            partition = {}
            for i, comm in enumerate(communities):
                for node in comm:
                    partition[node] = i
        except Exception as e:
            print(f"Native Louvain failed: {e}. Falling back to Greedy Modularity.")
            try:
                from networkx.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(G_nx))
                partition = {node: i for i, comm in enumerate(communities) for node in comm}
            except:
                partition = {node: 0 for node in G_nx.nodes()}

        # 2. Physics & UI Styling
        net = Network(height="900px", width="100%", bgcolor="#111111", font_color="white", select_menu=True, cdn_resources="in_line")
        
        # Extended Cyberpunk Color Palette
        cluster_colors = ["#00ccff", "#00ff88", "#ff00cc", "#ffff00", "#bf00ff", "#ff4400", "#00ffcc", "#ffcc00"]
        
        print("Adding nodes with weighted influence...")
        for node, attr in G_nx.nodes(data=True):
            node_type = attr.get('type', 'Unknown')
            degree = G_nx.degree(node)
            
            # Base size on degree (influence)
            size = 20 + (degree * 2)
            
            # Color by Community Partition
            community_id = partition.get(node, 0)
            color = cluster_colors[community_id % len(cluster_colors)]
            
            if node_type == 'Company':
                shape = 'dot'
                border_width = 2
                title = f"Ticker: {node}\nCommunity: {community_id}\nConnections: {degree}"
            elif node_type == 'Concept':
                shape = 'diamond'
                color = "#ffdd00" # High contrast for concepts
                size = 30
                border_width = 3
                title = f"Semantic Concept: {node}"
            elif node_type == 'Sector':
                shape = 'star'
                color = "#ffffff"
                size = 40
                border_width = 1
                title = f"Economic Sector: {node}"
            else:
                shape = 'ellipse'
                border_width = 1
                title = node
                
            net.add_node(node, label=node, title=title, color=color, size=size, shape=shape, borderWidth=border_width)
            
        print("Adding interactive edges...")
        for u, v, attr in G_nx.edges(data=True):
            edge_type = attr.get('type', 'semantic')
            weight = attr.get('weight', 1.0)
            
            if edge_type == 'correlation':
                color = "rgba(0, 204, 255, 0.4)" # Translucent Cyan
                width = weight * 4
                title = f"Correlation: {weight:.2f}"
                dashed = False
            elif edge_type == 'part_of':
                color = "rgba(255, 255, 255, 0.2)" # Subtle white
                width = 1
                dashed = True
            else:
                color = "rgba(255, 0, 204, 0.6)" # Neon Pink for semantic
                width = 2
                title = "Semantic Link"
                dashed = False
                
            net.add_edge(u, v, title=title, color=color, width=width, dashes=dashed)
            
        # Physics Enhancements for Stability and Movement
        net.set_options("""
        var options = {
          "nodes": {
            "font": { "size": 14, "face": "Inter" },
            "shadow": true
          },
          "edges": {
            "color": { "inherit": true },
            "smooth": { "type": "continuous", "roundness": 0.5 },
            "shadow": true
          },
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -30000,
              "centralGravity": 0.3,
              "springLength": 150,
              "springStrength": 0.05,
              "damping": 0.09,
              "avoidOverlap": 0.5
            },
            "minVelocity": 0.75
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "tooltipDelay": 100
          }
        }
        """)
        
        # Save
        print(f"Saving premium interactive graph to {self.output_path}...")
        net.save_graph(str(self.output_path))
        print("Visualization Complete!")

if __name__ == "__main__":
    viz = GraphVisualizer()
    viz.generate_viz()
