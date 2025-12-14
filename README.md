# MarketMind

**MarketMind** is a financial analysis and educational project that leverages modern technologies like Graph Neural Networks (GNNs) and data visualization to explore market dynamics. It aims to bridge the gap between complex quantitative finance concepts and accessible educational content.

## 🚀 Features

### 📊 Advanced Analytics
- **Portfolio Optimization**: Implementation of Modern Portfolio Theory (MPT) and Efficient Frontier analysis.
- **Graph Neural Networks**: Using GNNs to model relationships between assets and predict market movements.
- **Sentiment Analysis**: Integrating news sentiment into financial models.
- **Backtesting**: Robust engine to validate trading strategies.

### 🎥 Educational Content
- **Manim Animations**: We use [Manim](https://www.manim.community/) to create high-quality, programmatic educational videos explaining concepts like Alpha, Beta, and Sigma.
- **Interactive Notebooks**: A series of Jupyter notebooks guiding through data engineering, econometrics, and machine learning.

## 🛠️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/TerminalGambit/MarketMind.git
   cd MarketMind
   ```

2. **Set up the Environment**
   This project uses Conda for dependency management.
   ```bash
   conda env create -f environment.yaml
   conda activate market_mind
   ```
   *Note: This environment includes `manim` for video generation and `pytorch` for the GNN models.*

## 📂 Project Structure

- `notebooks/`: Educational and experimental Jupyter notebooks.
- `src/`: Core source code.
    - `analytics/`: Financial math and stats modules.
    - `graph/`: Logic for building and managing the knowledge graph.
    - `models/`: Machine learning models (GNNs).
    - `presentation/`: Scripts for Manim animations.
- `data/`: Data storage (ignored by git).

## 🎬 Generating Videos

To generate the educational animations (e.g., Alpha/Beta/Sigma):

```bash
conda activate market_mind
manim -qp src/presentation/alpha_beta_sigma.py AlphaBetaSigmaScene
```
This will verify the installation and render the video to `media/videos/`.

## 🤝 Contributing

This project is currently under active development.
