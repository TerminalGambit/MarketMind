import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

def create_analyst_flow_viz(save_path):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor='#111111')
    ax.set_facecolor('#111111')
    
    # Define boxes [x, y, width, height, color, label]
    boxes = [
        # Data Inputs
        [1, 5.5, 3, 1.5, '#2E86AB', 'Backtest Metrics\n(Returns, DD, Vol)'],
        [1, 3, 3, 1.5, '#2E86AB', 'News Headlines\n(Silver JSONL)'],
        
        # Processor
        [6, 4, 3, 2, '#F18F01', 'Mistral-7B Agent\n(via Ollama)\nContextual Synthesis'],
        
        # Output
        [11, 4, 2.5, 2, '#6A994E', 'Analyst Report\n(Markdown)\nMarket Commentary']
    ]
    
    # Draw Boxes
    for x, y, w, h, color, label in boxes:
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                      linewidth=2, edgecolor=color, facecolor=color, alpha=0.3)
        ax.add_patch(rect)
        rect_border = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2", 
                                             linewidth=2, edgecolor=color, facecolor='none')
        ax.add_patch(rect_border)
        ax.text(x + w/2, y + h/2, label, color='white', ha='center', va='center', 
                weight='bold', fontsize=11)

    # Draw Arrows
    arrow_props = dict(arrowstyle='->', lw=2.5, color='#aaaaaa', mutation_scale=20)
    
    # Inputs -> Mistral
    ax.annotate('', xy=(6, 5), xytext=(4.2, 6.25), arrowprops=arrow_props)
    ax.annotate('', xy=(6, 5), xytext=(4.2, 3.75), arrowprops=arrow_props)
    
    # Mistral -> Output
    ax.annotate('', xy=(11, 5), xytext=(9.2, 5), arrowprops=arrow_props)

    # Add labels
    ax.text(2.5, 7.5, "DATA INPUTS", color='#2E86AB', ha='center', weight='bold', fontsize=13)
    ax.text(7.5, 6.5, "CORE AI LOGIC", color='#F18F01', ha='center', weight='bold', fontsize=13)
    ax.text(12.25, 6.5, "OUTPUT", color='#6A994E', ha='center', weight='bold', fontsize=13)

    # Title
    plt.title('Mistral AI Analyst: From Data to Narrative', 
              color='white', fontsize=18, fontweight='bold', pad=30)
    
    # Remove axes
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='#111111')
    plt.close()
    print(f"✓ Analyst Flow Viz saved to {save_path}")

if __name__ == "__main__":
    out_dir = Path("visuals")
    out_dir.mkdir(parents=True, exist_ok=True)
    create_analyst_flow_viz(out_dir / "analyst_flow_viz.png")
