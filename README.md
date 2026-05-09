#  West Asia Conflict Predictive Model (2026)

> A real-time stochastic prediction engine that models the trajectory, attrition, and termination probability of the March 2026 Iran–USA–Israel conflict — using open-source intelligence, Lanchester's Laws, and Monte Carlo simulation.

---

##  Project Objective

This project transforms raw geopolitical events into a **quantified probability distribution of conflict outcomes**. It ingests live OSINT data streams, runs 10,000 stochastic simulations per update cycle, and outputs a continuously updating "Conflict Dashboard" with statistically rigorous forecasts.

**Built to demonstrate:** Data engineering pipelines, stochastic modeling, time-series analysis, and real-world complexity quantification — for roles in Quant Finance and Data Science.

---

##  Theoretical Framework

### Lanchester's Laws — The Physics of Fire
The base layer models kinetic attrition as a system of Ordinary Differential Equations (ODEs):

```
dA/dt = -k_B · B²     (Square Law — modern aimed combat)
dB/dt = -k_A · A²
```

- **Linear Law** → urban insurgency, area bombardment (unaimed fire)
- **Square Law** → long-range missile and drone exchanges (current conflict)

### Stochastic Shocks — Poisson Jump Processes
Critical events (arsenal strikes, leadership eliminations) are modeled as **Poisson Jumps** — not continuous variables. Each shock applies:
1. A discrete step-down in force inventory `N`
2. A temporary exponential decay in combat effectiveness coefficient `k`

### The 4-Layer Factor Taxonomy

| Layer | Description | Key Variables |
|---|---|---|
| **Kinetic** | Physical battlefield attrition | Loss-Exchange Ratio (LER), munition stockpiles, force density |
| **Economic** | Industrial & financial engine | Brent Crude, sanctions index, currency volatility |
| **Socio-Political** | Will to fight | Tooth-to-Tail Ratio (T3R), NLP sentiment, social media entropy |
| **Hybrid/Tech** | Digital & electronic warfare | Jamming success rate, cyber-attack frequency, tech-gap latency |

---

##  Data Sources (100% Free, 0 Budget)

| Source | What It Measures | API |
|---|---|---|
| **NASA FIRMS** | Thermal anomalies → missile strikes & artillery as kinetic proxy | REST API |
| **GDELT Project** | NLP sentiment across 100+ languages → diplomatic temperature | DOC 2.0 API |
| **Yahoo Finance** | Brent Crude, VIX, USD/ILS → economic panic index | `yfinance` |
| **ACLED** | GPS-tagged battle events, confirmed casualties | REST API |
| **Mesa (ABM)** | Synthetic data generation for censored/lagged periods | Python library |

---

##  Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   POLLING LOOP (15 min)                  │
│  NASA FIRMS → GDELT → yfinance → ACLED                  │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                FEATURE ENGINEERING                       │
│  Normalization → Lagging → PCA (20+ vars → 3–4 PCs)    │
│  Shock Detection (Poisson Jump identification)          │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│             SIMULATION ENGINE                            │
│  Lanchester ODE  +  Stochastic Shocks                   │
│  "Trump Pause" diplomatic switch logic                  │
│  Monte Carlo: 10,000 iterations → outcome PDF           │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  CONFLICT DASHBOARD                      │
│  Plotly Dash: Force curves, Sentiment, Heat map, PDF    │
│  Sensitivity Tornado Chart: What is the bottleneck?     │
└─────────────────────────────────────────────────────────┘
```

---

##  Repository Structure

```
west-asia-conflict-model-2026/
│
├── data/                        # Raw & processed data (gitignored)
│
├── notebooks/
│   └── EDA.ipynb                # Exploratory data analysis
│
├── src/
│   ├── __init__.py
│   ├── kinetic_pulse.py         # NASA FIRMS API module
│   ├── diplomatic_sentiment.py  # GDELT NLP sentiment module
│   ├── economic_signals.py      # yfinance economic indicators
│   ├── acled_events.py          # ACLED battle data module
│   ├── data_collector.py        # Master polling loop orchestrator
│   ├── feature_engineering.py   # Normalization, PCA, lagging
│   ├── lanchester_model.py      # ODE system + shock integration
│   ├── monte_carlo.py           # Stochastic simulation engine
│   └── abm_synthetic.py         # Mesa ABM for data augmentation
│
├── dashboard/
│   └── app.py                   # Plotly Dash dashboard
│
├── tests/
│   └── test_*.py                # Unit tests for each module
│
├── .github/
│   └── workflows/
│       └── data_update.yml      # GitHub Actions automation
│
├── results/
│   └── latest_output.json       # Auto-updated by CI/CD pipeline
│
├── .env                         # API keys (NEVER committed)
├── .gitignore
├── requirements.txt
└── README.md
```

---

##  Getting Started

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA (recommended for sentiment inference)
- Git

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/west-asia-conflict-model-2026.git
cd west-asia-conflict-model-2026

# 2. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Add your API keys
cp .env.example .env
# Edit .env with your NASA FIRMS and ACLED keys
```

### Running the Pipeline

```bash
# Run a single data collection cycle
python src/data_collector.py --once

# Start the continuous polling loop (15-min intervals)
python src/data_collector.py --loop

# Launch the dashboard
python dashboard/app.py
# Open http://localhost:8050
```

---

##  Key Outputs

### 1. Probability Density Function of Conflict Termination
The headline output: a full PDF showing the distribution of 10,000 simulated conflict end-dates, categorized by outcome type:

- Negotiated Ceasefire
- Iranian Capitulation
- Stalemate / Frozen Conflict
- Regional Escalation

### 2. Sensitivity Tornado Chart
Which factor is the **current bottleneck**? Ranks all variables by their marginal impact on the outcome PDF — answering whether oil prices or munition stockpiles are driving the trajectory right now.

### 3. Live Force Attrition Curves
Real-time ODE solutions showing projected force strength trajectories for both sides, with shock events overlaid as vertical markers.

### 4. Thermal Anomaly Heat Map
NASA FIRMS data plotted geospatially — a proxy for real-time kinetic intensity across the region.

---

##  Automation

This project uses **GitHub Actions** to run the data pipeline automatically:

- Polls all data sources on a scheduled interval
- Re-runs the Monte Carlo simulation
- Commits updated `results/latest_output.json` back to the repository
- Dashboard always reflects the latest state

See `.github/workflows/data_update.yml` for the workflow configuration.

---

##  Statistical Methods Used

| Method | Application |
|---|---|
| Ordinary Differential Equations (ODEs) | Lanchester attrition modeling |
| Poisson Jump Processes | Discrete shock events (arsenal strikes) |
| Monte Carlo Simulation (n=10,000) | Uncertainty quantification under fog of war |
| Principal Component Analysis (PCA) | Dimensionality reduction of 20+ variables |
| Gaussian KDE | Probability density estimation of outcomes |
| NLP Sentiment Analysis | GDELT diplomatic temperature quantification |
| Agent-Based Modeling (ABM) | Synthetic data generation for censored periods |
| Sensitivity Analysis | Variable importance ranking (tornado chart) |

---

##  Tech Stack

| Category | Tools |
|---|---|
| Core | Python 3.10, NumPy, SciPy, Pandas |
| ML / Stats | scikit-learn, HuggingFace Transformers |
| Simulation | Mesa (ABM), custom Monte Carlo engine |
| Data Sources | NASA FIRMS, GDELT, ACLED, yfinance |
| Visualization | Plotly, Dash |
| GPU Acceleration | PyTorch (CUDA), CuPy |
| Automation | GitHub Actions |
| Environment | VS Code, Jupyter |

---

##  Disclaimer

This project is an **academic and portfolio exercise** in applied statistics and data engineering. All conflict data is sourced from publicly available OSINT. Predictions are probabilistic model outputs — not intelligence assessments. This project does not advocate for any party in the conflict.

---

##  Author

**Anweshan Goswami**
MSc Statistics | Pondicherry University


---

## 📄 License

MIT License — see `LICENSE` for details.
