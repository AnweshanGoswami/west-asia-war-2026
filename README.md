West Asia War 2026 — Conflict Prediction Engine
> A real-time stochastic prediction engine modelling the trajectory, attrition, and termination probability of the 2026 Iran–USA–Israel conflict using open-source intelligence, Lanchester's Laws, and Monte Carlo simulation.
Built to demonstrate: Production-grade data engineering pipelines, stochastic ODE modelling, multi-source time-series fusion, and real-world geopolitical complexity quantification — for roles in Quant Finance and Data Science.
---
Project Objective
This project transforms raw geopolitical signals into a quantified probability distribution of conflict outcomes. It ingests live OSINT data streams, runs 10,000 stochastic simulations per update cycle, and outputs a continuously updating Conflict Dashboard with statistically rigorous forecasts.
---
Phase Progress
Phase	Description	Status
Phase 0	Project Setup	✅ Complete
Phase 1	Data Ingestion	✅ Complete
Phase 2	Feature Engineering	🔄 Active (Step 9 done)
Phase 3	Core Models (Lanchester ODE + Monte Carlo)	⏳ Pending
Phase 4	Dashboard	⏳ Pending
Phase 5	Backtesting & Uncertainty	⏳ Pending
Phase 6	Agent-Based Modelling	⏳ Pending
Phase 7	Automation & Polish	⏳ Pending
---
Theoretical Framework
Lanchester's Laws — The Physics of Fire
The base layer models kinetic attrition as a system of Ordinary Differential Equations:
```
dA/dt = -k_B · B²     (Square Law — modern aimed combat)
dB/dt = -k_A · A²
```
Square Law → long-range missile and drone exchanges (primary model)
Linear Law → urban insurgency, area bombardment (secondary regime)
Stochastic Shocks — Poisson Jump Processes
Critical events (arsenal strikes, infrastructure destruction) are modelled as Poisson Jumps, not continuous variables. Each shock applies:
A discrete step-down in force inventory `N`
A temporary exponential decay in combat effectiveness coefficient `k`
The Dual-Signal Veto System
A kinetic shock is declared only when two independent channels agree simultaneously:
`firms_frp_mean > 20` — physical thermal anomaly confirmed by NASA satellite
`gdelt_event_count > 0` — spatially anchored GDELT narrative event on the same date
FIRMS and GDELT were confirmed as statistically independent (r = −0.202), validating the necessity of cross-channel verification. A narrative report without thermal evidence, or a thermal spike without any GDELT event, does not trigger a shock.
GDELT 6-Day Narrative Lag
Cross-correlation analysis confirmed that GDELT conflict reporting trails physical thermal detections by exactly 6 days (peak CCF at lag +6). All GDELT event dates are corrected by −6 days during data merging to restore causal alignment before any spatial anchoring occurs.
---
Data Architecture
Sources
Source	Signal	API	Notes
NASA FIRMS	Thermal anomalies → kinetic proxy	REST	MODIS + VIIRS dual-satellite
GDELT V1	Daily historical kinetic events (CAMEO 18/19/20)	BigQuery	Replaces ACLED (paywall)
GDELT V2	15-minute realtime kinetic events	DOC 2.0 API	Live feed
GDELT DOC	DistilBERT NLP sentiment, GMM regime weights	DOC 2.0 API	100+ languages
Yahoo Finance	Brent Crude, VIX, USD/ILS, Gold	`yfinance`	S&P 500 display-only
> **Note:** ACLED was deprecated in Step 7 due to paywall restrictions. GDELT V1 (historical) and V2 (realtime) provide equivalent CAMEO-coded kinetic event coverage at zero cost.
4-Layer Factor Taxonomy
Layer	Variables
Kinetic	`firms_frp_mean`, `firms_brightness_mean`, `firms_anomaly_count`, `gdelt_event_count`, `gdelt_avg_goldstein`
Economic	`brent_crude_change`, `vix_change`, `usd_ils_change`, `gold_change`
Socio-Political	`distilbert_avg`, `hostile_weight`, `diplomatic_weight`, `bloc_divergence`, `military_diplomatic_gap`
Spatial	`gdelt_avg_anchor_dist` (diagnostic), BallTree 100km geocoding radius
Multi-Shard Storage Architecture
Raw data is distributed across 10+ source CSVs. All sharding, deduplication, and conflict resolution happens entirely in memory — raw files are never destructively merged. This prevents historical file corruption during incremental pipeline runs.
---
Pipeline Architecture
```
┌──────────────────────────────────────────────────────────────┐
│                    POLLING LOOP (15 min)                      │
│         src/data_collector.py orchestrates all modules        │
│   NASA FIRMS  →  GDELT V1/V2  →  yfinance  →  Sentiment     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  PHYSICAL LAYER COMPILER                      │
│  src/firms_compiler.py                                        │
│  • Coalesces MODIS + VIIRS Archive + VIIRS NRT                │
│  • unified_brightness: VIIRS preferred, MODIS fallback        │
│  • Spatial-temporal dedup (highest FRP wins on overlap)       │
│  • Hard filter: Feb 01 2026+ only                             │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                   DATA MERGER (Step 9)                        │
│  src/data_merger.py                                           │
│  • Master spine: daily Feb 01 2026 → present                  │
│  • BallTree spatial anchor: 100km Haversine radius            │
│  • 6-day GDELT lag correction applied pre-anchor              │
│  • Iran exception: IR events marked unverified, not dropped   │
│  • Sentiment stack: daily + outbreak_patch + realtime         │
│  • Weekend forward-fill on economic signals                   │
│  • Missing days flagged for ODE uncertainty propagation       │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING                          │
│  Step 10: MinMaxScaler fitted on training window only         │
│  Step 11: Lag columns (Oil +3d, Sentiment +7–14d)             │
│  Step 12: Dual-Signal Veto → kinetic_shock boolean            │
│  Step 13: PCA diagnostic (scree + loadings before commit)     │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  SIMULATION ENGINE                            │
│  Step 14: Lanchester Square Law ODE (scipy solve_ivp)         │
│  Step 15: Poisson Jump shocks → force step-downs              │
│  Step 16: "Trump Pause" diplomatic switch logic               │
│  Step 17: Monte Carlo — 10,000 iterations                     │
│  Step 18: Gaussian KDE → 4-outcome PDF                        │
└──────────────────────────┬───────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                  CONFLICT DASHBOARD                           │
│  dashboard/app.py — 8 Plotly Dash panels                      │
│  Force curves · Sentiment · Heat map · PDF · Tornado chart    │
└──────────────────────────────────────────────────────────────┘
```
---
Key Empirical Findings
Signal Independence Confirmed
GDELT narrative data and NASA FIRMS thermal detections are statistically independent (Pearson r = −0.202). This validates the Dual-Signal Veto architecture: the two channels provide genuinely orthogonal evidence, not redundant measurement of the same phenomenon.
GDELT 6-Day Narrative Lag
Peak cross-correlation function (CCF) analysis confirms global media reporting of kinetic events trails satellite thermal detection by 6 days. This lag is mechanically applied at Step 9 before spatial anchoring to ensure causality.
Spatial Anchor Distribution
BallTree nearest-neighbour matching in the West Asia theater produces tight geocoding residuals: mean 8.5km, median 2.0km, p75 2.8km (n=94 anchored days). The distribution confirms GDELT's gazetteer in this theater is more precise than expected, with genuine outliers (max 104km) corresponding to Iranian events where capital-city snapping is unavoidable.
Stationarity Results (ADF + KPSS)
Series	Result
Brent Crude	Trend-stationary (deterministic trend, not unit root)
Gold	Stationary after first differencing
USD/ILS	Stationary after first differencing
VIX	Stationary after first differencing
Absolute prices are dropped immediately after differencing. Only first-differences enter the model.
GPU Benchmarking (DistilBERT, RTX 3050 4GB)
Method	Throughput
Sequential loop	143.1 articles/sec ✅
PyTorch Dataset	99.5 articles/sec
Sequential loop outperforms Dataset batching on this GPU due to VRAM constraints. Sequential is the production method.
NASA Thermal Unification
`unified_brightness` → coalesces VIIRS 4.0µm (`bright_ti4`) and MODIS 4.0µm (`brightness`) fire channels
VIIRS preferred; MODIS as fallback
`bright_t31` / `bright_ti5` (11.0µm ambient channels) dropped — not used in kinetic detection
April FIRMS Satellite Blackout
NASA FIRMS telemetry absent on Apr 28–29 2026. These days are processed as `firms_data_missing = True` in the master timeline. The Dual-Signal Veto cannot fire on these dates and they receive elevated uncertainty in the Lanchester ODE.
---
Repository Structure
```
west-asia-war-2026/
│
├── data/                              # Raw & processed data (gitignored)
│   ├── firms_raw.csv                  # MODIS historical feed
│   ├── fire_archive_SV-C2_*.csv       # VIIRS archive
│   ├── fire_nrt_SV-C2_*.csv           # VIIRS NRT
│   ├── firms_compiled.csv             # Compiled physical layer (generated)
│   ├── gdelt_kinetic_raw.csv          # GDELT V1/V2 CAMEO 18/19/20 events
│   ├── gdelt_sentiment_daily.csv      # Daily NLP sentiment backbone
│   ├── outbreak_patch.csv             # Feb 28 – Mar 01 sentiment patch
│   ├── sentiment_realtime.csv         # Live sentiment shard
│   ├── economic_raw.csv               # Yahoo Finance historical
│   └── master_df.csv                  # Merged master timeline (generated)
│
├── models/                            # Fitted model objects (gitignored)
│   ├── scaler.pkl                     # MinMaxScaler (train window only)
│   └── pca.pkl                        # PCA transformer
│
├── src/
│   ├── kinetic_pulse.py               # NASA FIRMS API — NRT + Archive modes
│   ├── firms_compiler.py              # MODIS/VIIRS staging compiler
│   ├── diplomatic_sentiment.py        # GDELT DistilBERT + GMM sentiment
│   ├── gdelt_kinetic.py               # GDELT V1/V2 kinetic events
│   ├── economic_signals.py            # yfinance economic indicators
│   ├── data_collector.py              # 15-min polling orchestrator
│   ├── snapshot_manager.py            # JSON snapshots → dashboard replay
│   ├── historical_backfill.py         # Sequential Feb 01 reconstruction
│   ├── data_merger.py                 # Step 9 — master timeline builder
│   ├── normalizer.py                  # Step 10 — MinMaxScaler
│   ├── lag_engineer.py                # Step 11 — lag columns + CCF verification
│   ├── shock_detector.py              # Step 12 — Dual-Signal Veto
│   ├── pca_reducer.py                 # Step 13 — PCA diagnostic + transform
│   ├── lanchester_model.py            # Step 14 — Square Law ODEs
│   ├── monte_carlo.py                 # Step 17 — 10,000-iteration engine
│   └── abm_synthetic.py               # Mesa ABM synthetic data
│
├── dashboard/
│   └── app.py                         # Plotly Dash — 8 panels
│
├── tests/
│   └── test_*.py
│
├── .github/
│   └── workflows/
│       └── data_update.yml            # Daily GitHub Actions automation
│
├── results/
│   └── latest_output.json             # Auto-updated by CI/CD
│
├── .env                               # API keys (never committed)
├── .gitignore
├── requirements.txt
└── README.md
```
---
Getting Started
Prerequisites
Python 3.10+
NVIDIA GPU with CUDA (recommended for DistilBERT inference)
NASA FIRMS API key — register free
Git
Installation
```bash
git clone https://github.com/AnweshanGoswami/west-asia-war-2026.git
cd west-asia-war-2026

python -m venv venv
source venv/Scripts/activate      # Windows/Git Bash
# source venv/bin/activate         # Linux/Mac

pip install -r requirements.txt

cp .env.example .env
# Add NASA FIRMS key to .env
```
Running the Pipeline
```bash
# Step 1 — Compile physical layer (run once, or after new FIRMS data)
python src/firms_compiler.py

# Step 2 — Build master timeline
python src/data_merger.py

# Step 3 — Single data collection cycle
python src/data_collector.py --once

# Step 4 — Continuous 15-minute polling
python src/data_collector.py --loop

# Step 5 — Launch dashboard
python dashboard/app.py
# Open http://localhost:8050
```
---
Model Features
The following 16 variables enter the normalizer, lag engineer, and PCA. S&P 500 is excluded from all model inputs (structural break Feb 27 2026; retained as display-only drawdown).
```python
MODEL_FEATURES = [
    # Kinetic (physical layer)
    "firms_frp_mean",           # Mean fire radiative power
    "firms_brightness_mean",    # Mean unified thermal brightness
    "firms_anomaly_count",      # Daily thermal detection count

    # GDELT kinetic (narrative layer, 6-day lag applied)
    "gdelt_event_count",        # Spatially anchored CAMEO 18/19/20 events
    "gdelt_avg_goldstein",      # Mean Goldstein hostility scale
    "gdelt_total_mentions",     # Total media mentions
    "gdelt_avg_tone",           # Mean GDELT tone score

    # Economic
    "brent_crude_change",       # Brent Crude first difference
    "vix_change",               # VIX first difference
    "usd_ils_change",           # USD/ILS first difference
    "gold_change",              # Gold first difference

    # Sentiment (GMM-weighted DistilBERT)
    "distilbert_avg",           # Mean DistilBERT sentiment
    "hostile_weight",           # GMM hostile regime weight
    "diplomatic_weight",        # GMM diplomatic regime weight
    "bloc_divergence",          # Adversarial vs allied bloc divergence
    "military_diplomatic_gap",  # Military vs diplomatic sentiment gap
]
```
---
Key Outputs
1. Conflict Termination PDF
Distribution of 10,000 simulated end-dates across 4 outcome categories:
Negotiated Ceasefire
Iranian Capitulation
Stalemate / Frozen Conflict
Regional Escalation
2. Sensitivity Tornado Chart
Ranks all 16 model variables by marginal impact on outcome PDF — identifying the current strategic bottleneck in real time.
3. Live Force Attrition Curves
ODE solutions showing projected force strength trajectories with Poisson shock events overlaid as vertical markers.
4. Thermal Anomaly Heat Map
NASA FIRMS data plotted geospatially — real-time kinetic intensity proxy across the conflict theater.
5. Dashboard Replay Slider
Snapshot system (`snapshot_manager.py`) saves model state to JSON at each polling cycle, enabling full timeline scrubbing.
---
Statistical Methods
Method	Application
Ordinary Differential Equations	Lanchester Square Law attrition
Poisson Jump Processes	Discrete kinetic shock events
Monte Carlo Simulation (n=10,000)	Uncertainty quantification under fog of war
Gaussian Mixture Model (GMM)	Hostile / diplomatic sentiment regime detection
Principal Component Analysis	Dimensionality reduction (20+ vars → 3–4 PCs)
Gaussian KDE	Outcome probability density estimation
DistilBERT NLP	Multilingual diplomatic sentiment inference
Negative Binomial Distribution	Probabilistic casualty estimation from CAMEO codes
BallTree Haversine	Spatial cross-matching of GDELT events to FIRMS anomalies
ADF + KPSS Tests	Stationarity verification for all economic series
Cross-Correlation Function	GDELT narrative lag quantification
Agent-Based Modelling (Mesa)	Synthetic data for censored / lagged periods
---
Tech Stack
Category	Tools
Core	Python 3.10, NumPy, SciPy, Pandas
ML / NLP	scikit-learn, HuggingFace Transformers (DistilBERT)
Simulation	Mesa (ABM), custom Monte Carlo engine
Spatial	scikit-learn BallTree (Haversine metric)
Data Sources	NASA FIRMS, GDELT V1/V2, yfinance
Visualisation	Plotly, Dash
GPU Acceleration	PyTorch (CUDA), CuPy — RTX 3050 4GB tested
Automation	GitHub Actions
Environment	Python venv, VS Code
---
Disclaimer
This project is an academic and portfolio exercise in applied statistics and data engineering. All conflict data is sourced from publicly available OSINT. Predictions are probabilistic model outputs — not intelligence assessments. This project does not advocate for any party in the conflict.
---
Author
Anweshan Goswami
MSc Statistics · Pondicherry University
---
License
MIT License — see `LICENSE` for details.
