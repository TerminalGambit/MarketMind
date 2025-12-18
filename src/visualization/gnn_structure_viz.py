import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_gnn_viz(save_path):
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='#111111')
    ax.set_facecolor('#111111')
    
    # Define layers and positions
    # [x, y, width, height, color, label]
    layers = [
        # Input
        [1, 3, 2, 4, '#00ccff', 'Input Sequences\n(Returns + Sentiment)\n[T x N x F]'],
        # GCN
        [5, 3.5, 2, 3, '#00ff88', 'Spatial Block\n(Graph Conv)\nRelational Learning'],
        # LSTM
        [9, 3.5, 2, 3, '#bf00ff', 'Temporal Block\n(LSTM)\nSequential Learning'],
        # Output
        [13, 4, 1.5, 2, '#ff00cc', 'Output Head\n(Linear)\nAsset Predictions']
    ]
    
    # Draw Boxes
    for x, y, w, h, color, label in layers:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        
        # Border
        rect_border = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                             linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect_border)
        
        # Text
        ax.text(x + w/2, y + h/2, label, color='white', ha='center', va='center', 
                weight='bold', fontsize=11)

    # Draw Arrows
    arrow_props = dict(arrowstyle='->', lw=2.5, color='#aaaaaa', mutation_scale=20)
    
    # Input -> GCN
    ax.annotate('', xy=(5, 5), xytext=(3, 5), arrowprops=arrow_props)
    # GCN -> LSTM
    ax.annotate('', xy=(9, 5), xytext=(7, 5), arrowprops=arrow_props)
    # LSTM -> Output
    ax.annotate('', xy=(13, 5), xytext=(11, 5), arrowprops=arrow_props)

    # Add small "Graph" icon near GCN
    circle = patches.Circle((6, 3), 0.3, color='#00ff88', alpha=0.6)
    ax.add_patch(circle)
    ax.text(6, 2.5, "Graph Latent Space", color='#00ff88', ha='center', fontsize=9)

    # Title
    plt.title('Spatio-Temporal Graph Neural Network Architecture', 
              color='white', fontsize=18, fontweight='bold', pad=20)
    
    # Subtitle
    ax.text(7.5, 8, 'Market-Mind: Bridging Spatial Relationships & Temporal Trends', 
            color='#aaaaaa', ha='center', style='italic', fontsize=12)

    # Remove axes
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"✓ GNN Architecture Viz saved to {save_path}")

if __name__ == "__main__":
    out_dir = Path("visuals")
    out_dir.mkdir(parents=True, exist_ok=True)
    create_gnn_viz(out_dir / "gnn_architecture.png")
