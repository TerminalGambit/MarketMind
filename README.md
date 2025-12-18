# MarketMind 🧠📈

**An Integrated Quantitative Finance & Machine Learning Platform**

MarketMind is a comprehensive financial analysis platform that bridges theoretical finance with practical implementation. Built as both a pedagogical tool and a production-ready system, it synthesizes data engineering, machine learning, quantitative finance, and software engineering into a cohesive whole.

> 💡 **Pedagogical Mission**: This project demonstrates how concepts from data science, econometrics, deep learning, and quantitative finance come together to solve real-world financial problems.

---

## 🎯 What Makes MarketMind Unique?

### 1. **Complete Integration**
- **Data Engineering**: Bronze → Silver → Gold pipeline with Parquet storage
- **Machine Learning**: Graph Neural Networks (GNNs) for asset relationship modeling
- **Quantitative Finance**: 5 portfolio optimization algorithms (MVO, HRP, Black-Litterman, CVaR)
- **NLP & Sentiment**: LLM-based news analysis integrated into predictions
- **Visualization**: Manim animations + interactive notebooks

### 2. **Educational First**
- **12 Progressive Notebooks**: From data engineering to advanced optimization
- **Interactive CLI**: Browse and filter notebooks by topic
- **Visual Explanations**: Programmatic animations explaining Alpha, Beta, Sigma
- **Hands-on Learning**: Every concept backed by working code

### 3. **Production Quality**
- **Robust Testing**: Pytest suite with comprehensive coverage
- **Modular Architecture**: Clean separation of concerns
- **Configuration Management**: Centralized config for easy customization
- **Type Hints & Documentation**: Professional code standards

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/TerminalGambit/MarketMind.git
cd MarketMind

# Set up environment (includes PyTorch, Manim, CVXPY, etc.)
conda env create -f environment.yaml
conda activate market_mind
```

### Explore the Notebooks

```bash
# Option 1: Launch Jupyter
jupyter notebook

# Option 2: Use the interactive CLI
python tools/library_cli.py
```

### Run Portfolio Optimization

```python
from src.analytics.advanced_optimizer import AdvancedOptimizer
import pandas as pd

# Load market data
returns = pd.read_parquet("data/silver/market_returns_latest.parquet")

# Run Hierarchical Risk Parity
optimizer = AdvancedOptimizer()
hrp_weights = optimizer.get_hrp_weights(returns)
print(hrp_weights.sort_values(ascending=False))
```

---

## 📚 Learning Path

| Notebook | Topics | Difficulty |
|----------|--------|------------|
| **01** Data Engineering | Medallion Architecture, ETL, Parquet | Beginner |
| **02** Mathematical Foundations | Linear Algebra, Graph Theory | Beginner |
| **03** Econometrics & Alpha | Regression, Hypothesis Testing | Intermediate |
| **04** Graph Neural Networks | PyTorch, Message Passing | Advanced |
| **05** Modern Portfolio Theory | MPT, Efficient Frontier | Intermediate |
| **06** Alpha, Beta, Sigma | Risk-Adjusted Returns, CAPM | Intermediate |
| **07** Efficient Frontier | Optimization, Scipy | Intermediate |
| **08** Portfolio Construction | Strategy Integration | Intermediate |
| **09** Hierarchical Risk Parity | Clustering, Diversification | Advanced |
| **10** Black-Litterman Model | Bayesian Statistics | Advanced |
| **11** CVaR Optimization | Tail Risk, Convex Optimization | Advanced |
| **12** Portfolio Comparison | Benchmarking, Decision Framework | Intermediate |

---

## 🏗️ Architecture

```
MarketMind/
├── data/                    # Bronze → Silver → Gold layers
│   ├── bronze/             # Raw market & news data
│   ├── silver/             # Cleaned, validated data
│   └── gold/               # Analytics-ready aggregations
│
├── src/                     # Core source code
│   ├── ingestion/          # Data collection (yfinance, news scraping)
│   ├── processing/         # Cleaning, features, NLP, sentiment
│   ├── analytics/          # Portfolio optimization, backtesting, metrics
│   ├── models/             # GNN implementations (PyTorch)
│   ├── graph/              # Knowledge graph construction
│   ├── presentation/       # Manim animations
│   └── visualization/      # Interactive plots
│
├── notebooks/               # 12 educational notebooks
├── tests/                   # Pytest test suite
├── tools/                   # CLI utilities
└── research-papers/         # Academic references
```

---

## 🛠️ Technology Stack

**Core**: Python 3.10, PyTorch, NumPy, Pandas, SciPy  
**Optimization**: CVXPY, SciPy.optimize  
**Data**: yfinance, Selenium, Parquet  
**ML/NLP**: PyTorch, VADER, Ollama (Mistral)  
**Visualization**: Matplotlib, Seaborn, Manim, PyVis  
**Tools**: Pytest, Jupyter, Questionary  

---

## 🎓 Key Concepts Demonstrated

### Data Engineering
- Medallion architecture (Bronze/Silver/Gold)
- ETL pipelines with data quality checks
- Efficient storage with Parquet

### Machine Learning
- Graph Neural Networks for asset relationships
- Temporal modeling with attention mechanisms
- Feature engineering from market data

### Quantitative Finance
- Modern Portfolio Theory (Markowitz)
- Hierarchical Risk Parity (López de Prado)
- Black-Litterman (Bayesian optimization)
- CVaR (tail risk management)

### Software Engineering
- Modular, testable code
- Configuration management
- Type hints and documentation
- Continuous testing with pytest

---

## 📊 Portfolio Optimization Algorithms

| Algorithm | Focus | Best For |
|-----------|-------|----------|
| **Equal Weight** | Simplicity | Baseline comparison |
| **MVO (Max Sharpe)** | Risk-adjusted returns | High-quality return forecasts |
| **HRP** | Diversification | Large universes, stability |
| **Black-Litterman** | View incorporation | Institutional portfolios |
| **CVaR** | Tail risk | Downside protection |

---

## 🎬 Generate Educational Animations

```bash
# Create Alpha/Beta/Sigma visualization
manim -qp src/presentation/alpha_beta_sigma.py AlphaBetaSigmaScene

# Output: media/videos/alpha_beta_sigma.mp4
```

---

## 🧪 Run Tests

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_optimization.py

# Run with coverage
pytest --cov=src tests/
```

---

## 📖 Documentation

- **[PROJECT_OVERVIEW.md](PROJECT_OVERVIEW.md)**: Comprehensive project description
- **Notebooks**: Interactive tutorials with explanations
- **Docstrings**: Every function documented with examples
- **Type Hints**: Clear function signatures

---

## 🎯 Learning Outcomes

By exploring MarketMind, you'll learn:

1. **Data Engineering**: Building scalable data pipelines
2. **Mathematical Finance**: Translating theory into code
3. **Machine Learning**: Applying GNNs to financial problems
4. **Portfolio Optimization**: From classical to cutting-edge techniques
5. **Software Engineering**: Writing production-quality code
6. **Communication**: Presenting technical work clearly

---

## 🔬 Research Integration

MarketMind incorporates recent academic research:

- **HRP**: López de Prado (2016) - Diversified portfolios via clustering
- **GNNs**: Temporal graph networks for asset prediction
- **Black-Litterman**: Bayesian portfolio optimization
- **CVaR**: Coherent risk measures for tail risk

---

## 🚀 Future Directions

- Real-time trading integration
- Multi-asset class support (bonds, commodities, crypto)
- Reinforcement learning for dynamic allocation
- Web dashboard for portfolio monitoring
- Explainable AI for GNN predictions

---

## 🤝 Contributing

This project is currently under active development as a pedagogical tool. Feedback and suggestions are welcome!

---

## 📄 License

This project is for educational purposes.

---

## 🙏 Acknowledgments

Built with concepts from:
- Modern Portfolio Theory (Markowitz)
- Graph Neural Networks (Kipf & Welling)
- Hierarchical Risk Parity (López de Prado)
- Black-Litterman Model (Black & Litterman)
- Quantitative Risk Management (McNeil et al.)

---

**MarketMind**: Where finance meets machine learning, and theory meets practice. 🧠📈
