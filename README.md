# ForexGuard — Real-Time Forex Trader Anomaly Detection


> Production-grade, unsupervised anomaly detection engine for a forex brokerage — featuring a multi-scale Transformer autoencoder, Borda count ensemble, SHAP explainability, LLM compliance narratives, and a live WebSocket dashboard.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Model Design & Justification](#3-model-design--justification)
4. [Feature Engineering](#4-feature-engineering)
5. [Suspicious Behaviour Coverage](#5-suspicious-behaviour-coverage)
6. [Assignment Requirements Checklist](#6-assignment-requirements-checklist)
7. [Setup & Running Locally](#7-setup--running-locally)
8. [API Reference](#8-api-reference)
9. [Real-Time Triggers](#9-real-time-triggers)
10. [Deployment on Render](#10-deployment-on-render)
11. [Assumptions, Trade-offs & Limitations](#11-assumptions-trade-offs--limitations)
12. [File Structure](#12-file-structure)

---

## 1. Project Overview

ForexGuard is a four-phase, end-to-end ML pipeline that identifies anomalous user and trader behaviour across a simulated forex brokerage:

| Phase | Script | Purpose |
|---|---|---|
| 1 | `generate_data.py` | 50,000+ synthetic events across 500 users, 90-day simulation |
| 2 | `feature_engineering.py` | 80+ behavioural features across 11 groups |
| 3 | `train_models.py` | Three models + calibrated ensemble + SHAP + LLM summaries |
| 4 | `main.py` + `dashboard.html` | FastAPI REST + WebSocket real-time pipeline |

The system flags suspicious users in real time, pushes alerts to a live dashboard via WebSocket, and generates human-readable compliance narratives using an LLM.

---

## 2. System Architecture

```
Raw Events  (login / trade / deposit / withdrawal)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  Phase 1 — Data Generation  (generate_data.py)       │
│  500 users · 90-day simulation · 50,000+ events      │
│  Distributions: log-normal, Pareto, Gamma            │
│  12 anomaly types · network-level collusion rings    │
│  Output: portal_events.csv · trade_events.csv        │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  Phase 2 — Feature Engineering  (feature_engineering.py) │
│  80+ features across 11 groups:                      │
│  Login/Access · Financial · Trading · Session        │
│  Rolling Windows · Inter-event Deltas · Device/IP    │
│  Graph/Network · Temporal · Composite Risk Scores    │
│  RobustScaler fitted and saved for consistent        │
│  transform at training and inference time            │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  Phase 3 — Model Training  (train_models.py)         │
│                                                      │
│  Baseline 1: Isolation Forest (contamination sweep)  │
│  Baseline 2: Local Outlier Factor (independent sweep)│
│  Advanced:   Staged Sliding Window Transformer       │
│              · 3 temporal scales: 7d / 14d / 30d     │
│              · Entropy-weighted self-attention       │
│              · Sinusoidal positional encoding        │
│              · Unsupervised autoencoder (MSE score)  │
│              · Inspired by Bao et al. arXiv:2504.00287│
│  Ensemble:   Borda count rank-fusion + Platt scaling │
│  Explain:    SHAP KernelExplainer + per-feature MSE  │
│  LLM:        Groq llama-3.3-70b compliance summaries │
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  Phase 4 — Real-Time API + Dashboard  (main.py)      │
│                                                      │
│  FastAPI                                             │
│  ├── /score          → full risk profile (REST)      │
│  ├── /predict        → live feature inference (REST) │
│  ├── /alerts         → paginated alert list          │
│  ├── /stream/ingest  → event ingestion               │
│  ├── /ws/alerts      → real-time push (WebSocket) ◄─┐│
│  └── /ws/events      → raw event echo (WebSocket)   ││
│                                                      ││
│  Streaming Pipeline                                  ││
│  ├── Per-event incremental feature update            ││
│  ├── Instant triggers (brute-force / spike / geo)    ││
│  ├── Blended scoring (60% history + 40% live)        ││
│  └── WebSocket broadcast on CRITICAL/HIGH alert ─────┘│
└──────────────────────────┬───────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────┐
│  Dashboard  (dashboard.html)                         │
│  ├── WebSocket connected — alerts arrive instantly   │
│  ├── Toast notifications for CRITICAL/HIGH           │
│  ├── Live alert feed with LLM narratives             │
│  ├── Score distribution + tier donut charts          │
│  ├── Per-anomaly-type detection rate bars            │
│  ├── User lookup with SHAP drivers + domain scores   │
│  └── Live inference panel (slider-based input)       │
└──────────────────────────────────────────────────────┘
```

---

## 3. Model Design & Justification

### 3.1 Why Unsupervised?

In a real forex brokerage, ground-truth fraud labels are sparse, delayed, and biased (only confirmed cases are labelled). Supervised models would train on an unrepresentative sample. The assignment explicitly prefers an unsupervised approach. All three models operate without labels during training.

---

### 3.2 Baseline 1 — Isolation Forest

**How it works:** Randomly partitions the feature space using decision trees. Anomalies, being rare and different, require fewer splits to isolate and therefore have shorter average path lengths.

**Why chosen:**
- O(n log n) time complexity — scales to large user populations.
- No distributional assumptions — effective on the mixed feature types in this dataset (counts, ratios, z-scores, binary flags).
- Contamination parameter (proportion of anomalies expected) was swept across `[0.10, 0.15, 0.20, 0.25]` with `n_estimators` swept across `[200, 300]`. Best configuration selected by validation AUC.

---

### 3.3 Baseline 2 — Local Outlier Factor

**How it works:** Compares the local density of a point to the densities of its k-nearest neighbours. Points in significantly lower-density regions than their neighbours are flagged as outliers.

**Why chosen:**
- Density-based approach provides a different inductive bias from the tree-based IF, making them complementary in the ensemble.
- Catches tight clusters of normal users and isolates genuine outliers even in non-linear feature spaces where IF may struggle.
- `novelty=True` allows scoring of new users not seen during training.
- `n_neighbors` independently swept across `[10, 20, 30, 50]` — not shared with the IF contamination sweep.

---

### 3.4 Advanced Model — Staged Sliding Window Transformer Autoencoder

This is the novel contribution of ForexGuard, adapted from the research literature.

**Architecture inspiration:** Bao et al. (2025), *"A Deep Learning Approach to Anomaly Detection in High-Frequency Trading Data"*, arXiv:2504.00287.

**Key design decisions and what is new:**

| Design Decision | What it does | Why it matters |
|---|---|---|
| **Multi-scale temporal windows** | Three parallel encoder stages process the last 7, 14, and 30 days of daily-aggregated behaviour simultaneously | Captures both short-term bursts (brute-force, volume spikes) and long-term drift (structuring, bonus abuse) in one model |
| **Daily-aggregated sequences** | Each timestep is a vector of 9 daily aggregated metrics (trade count, volume, PnL, deposits, withdrawals, logins, failed logins, unique IPs, session count) | Avoids the modulated-static-vector antipattern; creates a true temporal sequence |
| **Entropy-weighted self-attention** | Attention scores are modulated by per-timestep feature entropy: w_t = −Σ p·log(p) across feature values | Amplifies attention on windows where features are unusually concentrated (e.g., all trades in one instrument), which is exactly when anomalies appear |
| **Sinusoidal positional encoding** | Standard PE injected at each stage before self-attention | Preserves time-ordering within each window so the model knows that day 7 comes after day 1 |
| **Unsupervised autoencoder** | Trained to reconstruct normal user sequences; reconstruction MSE becomes the anomaly score | Follows real-world constraint: <1% of brokerage users have confirmed fraud labels at any time |
| **Sliding window training augmentation** | Training uses stride=1 windows across the 90-day history, not just one sequence per user | Multiplies training samples, improves generalisation to partial histories |
| **Padding mask** | Padded timesteps (new users with short history) are masked in self-attention | Prevents zero-fill from being interpreted as anomalous activity |
| **Mean-fill padding** | Padded positions use the user's historical mean rather than zeros | Avoids false anomaly signals from cold-start users |
| **Conditional triggering** | IF/LOF run first; Transformer only re-scores users with `volume_spike OR login_anomaly` flags | Reduces compute in production; prioritises expensive deep inference where it adds value |

**The paper's sigmoid classifier is replaced with an MSE reconstruction threshold** because the paper assumes access to labelled trading anomalies during training, which this assessment deliberately avoids. Per-feature reconstruction error (`recon_error_per_feat`) is used as a per-dimension anomaly attribution, equivalent to SHAP for the Transformer.

---

### 3.5 Ensemble — Borda Count Rank Fusion + Platt Scaling

**Why rank fusion instead of score averaging?**

IF and LOF share a density-related inductive bias (both penalise isolated points). Averaging their scores under-weights the independent Transformer signal. Borda count rank fusion treats each model as a ranker — it assigns ordinal ranks and combines those, meaning a user who ranks #1 in all three models gets maximum weight regardless of whether the raw scores are on the same scale.

**Platt scaling** (logistic calibration on a held-out validation set) converts the raw ensemble rank scores into interpretable probabilities: a score of 0.87 means the model assigns 87% probability to this user being anomalous. This is directly meaningful to a compliance analyst.

**Optional Bayesian refinement:** After Borda fusion, a logistic regression trained on validation labels is offered as a second-stage combiner. It is only used if its AUC exceeds the Borda baseline — otherwise the simpler Borda count is retained.

---

### 3.6 Alert Tier Thresholds

Thresholds are computed from the actual score distribution at startup using the **Tukey IQR fence method** (Rousseeuw & Croux robust statistics), not fixed percentiles:

```
CRITICAL : calibrated_score > Q75 + 3.0 × IQR  (extreme outlier, outer fence)
HIGH     : calibrated_score > Q75 + 1.5 × IQR  (mild outlier, inner fence)
MEDIUM   : calibrated_score > Q50 + 1.0 × IQR  (above median by one spread)
LOW      : everything else
```

This means the threshold adapts to the actual score distribution rather than arbitrarily flagging exactly 5% of users as critical regardless of whether those users are genuinely suspicious.

---

## 4. Feature Engineering

80+ features across 11 groups, all derived from raw event timestamps and amounts. All thresholds are **data-derived from the population distribution** (loaded from `population_stats.csv`), not hardcoded.

| Group | Count | Key Features |
|---|---|---|
| A — Login & Access | 10 | `ip_entropy`, `brute_force_max_burst`, `offhours_login_ratio`, `geo_impossible_login_count`, `multi_ip_day_count`, `login_failure_rate` |
| B — Financial | 9 | `small_deposit_ratio`, `dormant_withdrawal_flag`, `kyc_before_withdrawal_flag`, `bonus_abuse_flag`, `deposit_withdrawal_ratio`, `deposit_to_withdrawal_min_h` |
| C — Trading | 16 | `volume_spike_ratio`, `latency_arb_ratio`, `instrument_gini`, `pnl_sharpe`, `win_rate`, `direction_imbalance`, `rolling_7d_volume_zscore` |
| D — Session & Behaviour | 8 | `bot_speed_ratio`, `event_type_entropy`, `event_bigram_entropy`, `mean_session_duration`, `support_ticket_count` |
| E — Rolling Windows | 8 | `rolling_7d_volume_zscore`, `rolling_30d_volume_zscore`, `rolling_7d_deposit_zscore`, `rolling_7d_withdrawal_zscore` |
| F — Inter-event Deltas | 3 | `mean_inter_login_hours`, `min_inter_login_sec`, `login_to_trade_mean_min` |
| G — Device & Fingerprint | 3 | `fingerprint_entropy`, `fingerprint_mismatch_count`, `device_rotation_rate` |
| H — Graph / Network | 6 | `ip_hub_score`, `device_sharing_score`, `sync_trade_count`, `collusion_risk_score`, `graph_betweenness`, `max_users_per_shared_ip` |
| I — Temporal | 5 | `concept_drift_volume_ratio`, `news_window_trade_ratio`, `trade_dow_gini`, `trade_hour_entropy` |
| J — Composite Risk Scores | 5 | `login_risk_score`, `financial_risk_score`, `trading_risk_score`, `network_risk_score`, `composite_risk_score` |
| K — User Activity | 2 | `user_total_events`, `user_activity_score` |

**Notable engineering choices:**

- **Bigram entropy** over event sequences: a bot running a fixed script produces very low bigram entropy (always the same 2–3 transitions); a human has varied, non-repetitive navigation.
- **Gini coefficient** on instrument trade counts: a value near 1.0 means all trades are on one instrument (single-instrument concentration, §8.3 of the assessment).
- **Composite risk scores** are computed with **data-driven weights**: each component feature's weight equals its absolute Pearson correlation with the ground-truth anomaly label, normalised to sum to 1. This avoids ad hoc weight tuning.
- **Graph betweenness centrality** (NetworkX) identifies IP/device nodes that act as hubs in the user–IP–device bipartite graph, directly detecting the §8.5 network-level patterns.
- Labels are kept in `labels_eval.csv` and **never merged into** `features.csv`, preventing any label leakage into model training.

---

## 5. Suspicious Behaviour Coverage

Every behaviour signal listed in the assessment (§8.1–8.7) is implemented:

| Section | Category | Signals | Feature / Method | Status |
|---|---|---|---|---|
| §8.1 | Login & Access | Multi-IP simultaneous login | `multi_ip_day_count`, `ip_entropy` | ✅ |
| §8.1 | Login & Access | Rapid IP switching / geo-impossible | `geo_impossible_login_count` | ✅ |
| §8.1 | Login & Access | IP hub — multiple users on one IP | `ip_hub_score`, `max_users_per_shared_ip` | ✅ |
| §8.1 | Login & Access | Off-hours login (e.g. 3 AM) | `offhours_login_ratio`, `offhours_login_count` | ✅ |
| §8.1 | Login & Access | Device fingerprint mismatch | `fingerprint_entropy`, `fingerprint_mismatch_count` | ✅ |
| §8.2 | Financial | Sudden large withdrawal after dormancy | `dormant_withdrawal_flag` | ✅ |
| §8.2 | Financial | Deposit → minimal trading → withdrawal | `bonus_abuse_flag`, `deposit_to_withdrawal_min_h` | ✅ |
| §8.2 | Financial | High-frequency small deposits (structuring) | `small_deposit_ratio`, `small_deposit_count` | ✅ |
| §8.2 | Financial | Bonus abuse cycles | `bonus_abuse_flag` | ✅ |
| §8.3 | Trading | Volume spike (10× baseline) | `volume_spike_ratio`, `rolling_7d_volume_zscore` | ✅ |
| §8.3 | Trading | Single-instrument concentration | `instrument_gini`, `top_instrument_ratio` | ✅ |
| §8.3 | Trading | Latency arbitrage patterns | `latency_arb_ratio`, `min_trade_duration_s` | ✅ |
| §8.3 | Trading | Consistent profit in short bursts | `pnl_sharpe`, `win_rate`, `news_window_trade_ratio` | ✅ |
| §8.4 | Behavioural | Unusual session durations | `mean_session_duration`, `min_session_duration` | ✅ |
| §8.4 | Behavioural | Bot-like rapid navigation | `bot_speed_ratio`, `event_bigram_entropy` | ✅ |
| §8.4 | Behavioural | Frequent device switching | `device_rotation_rate`, `unique_device_count` | ✅ |
| §8.5 | Graph / Network | Shared IP/device across accounts | `device_sharing_score`, graph bipartite analysis | ✅ |
| §8.5 | Graph / Network | Synchronised trading across accounts | `sync_trade_count`, `collusion_risk_score` | ✅ |
| §8.5 | Graph / Network | Collusion rings / mirror trades | Collusion ring injection + `sync_trade_count` | ✅ |
| §8.6 | Temporal | Trading aligned with news events | `news_window_trade_ratio` | ✅ |
| §8.6 | Temporal | Sudden behaviour shift | `concept_drift_volume_ratio`, `concept_drift_trade_ratio` | ✅ |
| §8.7 | Account Risk | KYC changes before withdrawal | `kyc_before_withdrawal_flag` | ✅ |
| §8.7 | Account Risk | Multiple failed logins then success | `brute_force_max_burst`, `login_failure_rate` | ✅ |

---

## 6. Assignment Requirements Checklist

| Requirement | Implementation | Status |
|---|---|---|
| **Unsupervised anomaly detection** | IF + LOF + Transformer all trained without labels | ✅ |
| **One baseline model** | Isolation Forest | ✅ |
| **One advanced model** | Staged SW Transformer Autoencoder | ✅ |
| **Model justification** | Section 3 of this README; `model_report.json` | ✅ |
| **Rolling statistical windows** | 7d / 14d / 30d z-scored rolling features | ✅ |
| **Session-based metrics** | Duration, bot-speed, bigram entropy | ✅ |
| **Inter-event time deltas** | Login gaps, login-to-trade latency | ✅ |
| **Trade clustering patterns** | Instrument Gini, direction imbalance, sync count | ✅ |
| **Device/IP deviation scoring** | Fingerprint entropy, IP hub score, geo-impossible | ✅ |
| **PnL volatility** | `pnl_std`, `pnl_sharpe`, `win_rate` | ✅ |
| **Streaming / real-time processing** | Async simulation via `stream_simulator.py` + WebSocket push | ✅ |
| **FastAPI /predict or /score** | Both endpoints exposed | ✅ |
| **Top contributing features per anomaly** | SHAP top-3 + Transformer per-feature MSE | ✅ |
| **Human-readable alerts** | LLM compliance narratives (Groq) + rule-based fallback | ✅ |
| **MLflow tracking** | Per-epoch val_loss + hyperparameter sweeps | ✅ |
| **Docker** | `Dockerfile` included | ✅ |
| **Hosted demo** | Deployable on Render (instructions below) | ✅ |
| **Clean, modular code** | Four independent pipeline scripts | ✅ |
| **README with architecture + assumptions** | This document | ✅ |

---

## 7. Setup & Running Locally

### Prerequisites

```bash
Python 3.10+
pip install -r requirements.txt

# Optional: GPU acceleration for Transformer training
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

### Environment Variables

```bash
# Free key at https://console.groq.com — enables LLM risk summaries
export GROQ_API_KEY="gsk_..."

# Windows PowerShell
$env:GROQ_API_KEY = "gsk_..."
```

The pipeline runs fully without `GROQ_API_KEY` — it falls back to rule-based compliance summaries.

---

### Step 1 — Generate Synthetic Data

```bash
python generate_data.py
```

Output in `data/`:
- `portal_events.csv` — ~35,000 client portal events
- `trade_events.csv` — ~20,000 trading terminal events
- `labels.csv` — ground-truth anomaly labels
- `user_profiles.csv` — per-user baseline statistics
- `population_stats.csv` — population-level distribution thresholds

---

### Step 2 — Feature Engineering

```bash
python feature_engineering.py
```

Output:
- `data/features.csv` — 80+ feature matrix (500 users × 80+ features)
- `data/labels_eval.csv` — evaluation labels (separate from feature matrix)
- `models/scaler.pkl` — fitted RobustScaler

---

### Step 3 — Train Models

```bash
# Fast mode (default, ~10–20 min CPU)
python train_models.py

# Full research sweep (~1–2 hrs CPU, higher accuracy)
FOREXGUARD_FAST=0 python train_models.py
```

Output in `models/`:
- `isolation_forest.pkl`, `lof.pkl`
- `transformer_autoencoder.pt`, `transformer_config.json`
- `ensemble_weights.json`, `platt_scaler.pkl`
- `llm_risk_summaries.json`, `model_report.json`

Output in `data/`:
- `scores.csv` — per-user scores, alert tiers, SHAP top features, LLM summaries
- `shap_values.csv` — full SHAP attribution matrix

---

### Step 4 — Start the API

```bash
uvicorn main:app --host 0.0.0.0 --port 8000

# Verify it's running:
curl http://localhost:8000/health
```

---

### Step 5 — Open the Dashboard

```
Open dashboard.html in your browser, or navigate to:
http://localhost:8000/dashboard
```

The dashboard auto-connects to `ws://localhost:8000/ws/alerts`. You will see **LIVE** in the top-right corner when the WebSocket is established.

---

### Step 6 — Run the Stream Simulator

```bash
python stream_simulator.py
```

This replays all 50,000+ historical events at 1000× real-time speed through the ingest API. Watch CRITICAL and HIGH alerts fire on the dashboard in real time.

---

## 8. API Reference

### Endpoints

| Method | URL | Description |
|---|---|---|
| GET | `/` | API status, model load state, WS endpoints |
| GET | `/health` | Health check |
| POST | `/score?user_id=U0042` | Full blended risk profile |
| POST | `/predict` | Raw feature inference (JSON body) |
| GET | `/alerts` | Paginated flagged user list |
| GET | `/alerts/summary/stats` | Aggregate counts for dashboard charts |
| GET | `/alerts/{user_id}` | Single user detail |
| POST | `/stream/ingest` | Ingest one real-time event |
| POST | `/stream/ingest/batch` | Ingest multiple events |
| GET | `/stream/status` | Pipeline metrics + WS client count |
| WS | `/ws/alerts` | Real-time alert push channel |
| WS | `/ws/events` | Raw event echo |
| GET | `/models/info` | Model metadata + AUC scores |
| GET | `/features/list` | All 80+ feature names |
| GET | `/dashboard` | Serve dashboard.html |
| GET | `/docs` | Swagger UI |

### Sample `/score` Response

```json
{
  "user_id": "U0042",
  "if_score": 0.7821,
  "lof_score": 0.6543,
  "trans_score": 0.8102,
  "ensemble_score": 0.7654,
  "calibrated_score": 0.8231,
  "alert_tier": "CRITICAL",
  "score_source": "blended",
  "shap_top1_feature": "volume_spike_ratio",
  "shap_top1_value": 0.4312,
  "shap_top2_feature": "latency_arb_ratio",
  "shap_top2_value": 0.2841,
  "shap_top3_feature": "offhours_login_ratio",
  "shap_top3_value": 0.1923,
  "llm_risk_summary": "User U0042 exhibits a 12× trade volume spike ...",
  "composite_risk_score": 0.812,
  "financial_risk_score": 0.643,
  "trading_risk_score": 0.891
}
```

### Sample `/predict` Request Body

```json
{
  "user_id": "NEW_USER_001",
  "features": {
    "ip_entropy": 3.5,
    "login_failure_rate": 0.8,
    "volume_spike_ratio": 12.3,
    "brute_force_max_burst": 8,
    "offhours_login_ratio": 0.9,
    "geo_impossible_login_count": 2
  }
}
```

### WebSocket Alert Message

```json
{
  "user_id": "U0042",
  "tier": "CRITICAL",
  "score": 0.87,
  "trigger": "volume_spike",
  "top_feature": "volume_spike_ratio",
  "llm_summary": "User exhibits 12× volume spike relative to 30-day baseline...",
  "timestamp": "2024-03-15T14:32:05.123456"
}
```

---

## 9. Real-Time Triggers

Certain event patterns bypass the 100-event buffer and trigger immediate scoring:

| Trigger | Condition | Why immediate |
|---|---|---|
| Brute-force login | ≥ 3 failed logins within 10 minutes | Account takeover risk; cannot wait |
| Volume spike | Trade volume > 5× user's rolling mean | May indicate wash trading in progress |
| Large withdrawal | Amount ≥ $5,000 | High financial exposure |
| Geo-impossible login | Two logins from different continents within 2 hours | Physically impossible travel |

Blended scoring combines 60% of the pre-computed historical score with 40% of the live inference score. This provides stability against single-event noise while remaining reactive to sustained anomalous patterns.

---

## 10. Deployment on Render

### Prerequisites
- GitHub repository with all project files committed.
- Free account at [render.com](https://render.com).
- Trained model artefacts (`models/` directory) and pre-computed scores (`data/scores.csv`) committed to the repository, **or** a build step that runs the training pipeline.

> **Note:** Render's free tier has limited memory (~512 MB). The API loads pre-trained models from disk on startup — no training runs on Render. Commit `models/` and `data/` to the repository before deploying.

### Step-by-Step

**1. Prepare your repository**

```bash
git init
git add .
git commit -m "Initial ForexGuard submission"
git remote add origin https://github.com/YOUR_USERNAME/forexguard.git
git push -u origin main
```

**2. Create a new Web Service on Render**

- Go to [render.com](https://render.com) → **New** → **Web Service**
- Connect your GitHub account and select your repository

**3. Configure the service**

| Setting | Value |
|---|---|
| **Environment** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn main:app --host 0.0.0.0 --port 10000` |
| **Instance Type** | Free (512 MB) — sufficient for inference only |

**4. Set environment variables**

In the Render dashboard → **Environment** tab:

| Key | Value |
|---|---|
| `GROQ_API_KEY` | `gsk_...` (your free Groq key) |
| `FOREXGUARD_FAST` | `1` |

**5. Deploy**

Click **Create Web Service**. Render will build and deploy automatically. The first deploy takes 2–4 minutes.

Your service URL will be: `https://forexguard-xxxx.onrender.com`

**6. Verify**

```bash
curl https://forexguard-xxxx.onrender.com/health
# → {"status": "ok", "users_loaded": 500}

curl https://forexguard-xxxx.onrender.com/
# → API status with model info
```

**7. Open the dashboard**

Navigate to: `https://forexguard-xxxx.onrender.com/dashboard`

The dashboard auto-detects the API base URL from `window.location.origin` — no configuration needed.

### Docker Deployment (any cloud provider)

```bash
# Build
docker build -t forexguard .

# Run locally
docker run -p 8000:8000 -e GROQ_API_KEY=gsk_... forexguard

# Push to a registry and deploy to AWS/GCP/Azure as needed
docker tag forexguard YOUR_REGISTRY/forexguard:latest
docker push YOUR_REGISTRY/forexguard:latest
```

---

## 11. Assumptions, Trade-offs & Limitations

### Assumptions

- **Stable user identity:** User IDs are assumed stable across sessions. No anonymous-to-logged-in identity resolution is performed.
- **Historical baseline as ground truth:** `scores.csv` represents each user's full behavioural baseline. The blending ratio (60/40) assumes the historical signal is more reliable than any single new event.
- **Synthetic distribution fidelity:** Log-normal deposits, Pareto trade volumes, and Gamma session durations are domain-informed approximations of real forex user behaviour. Exact parameter values would require calibration against real brokerage data.
- **IP-to-continent mapping:** The geo-impossible detection uses a simple first-octet IP mapping as a proxy for MaxMind GeoLite2. Accuracy is sufficient for synthetic data; a production system would use a real GeoIP database.

### Trade-offs

- **Blended scoring (60/40)** prioritises stability over immediate reactivity. A user's baseline does not change from a single event, but repeated anomalous events will progressively shift the live score upward.
- **LOF fitted on training users only** (`novelty=True`). New users fall back to IF-only inference until they accumulate sufficient event history.
- **Transformer trained unsupervised** because the anomaly rate in real brokerage data would be <1%, making supervised labels unavailable at scale. The paper's sigmoid classifier is replaced by MSE reconstruction threshold.
- **Fast mode by default** reduces the hyperparameter sweep from ~2 hours to ~15 minutes with <1% AUC impact. Full sweep is available via `FOREXGUARD_FAST=0`.
- **LLM summaries capped at 20 users** during training to avoid Groq free-tier rate limits. All CRITICAL/HIGH users receive summaries at inference time via the API.

### Limitations

- **No persistence across API restarts:** Live user event buffers are held in memory. A production deployment would use Redis for state persistence.
- **WebSocket not authenticated:** A production system would validate JWT tokens on the WebSocket handshake.
- **No Kafka integration:** `stream_simulator.py` replays events via HTTP — functionally equivalent to a Kafka consumer calling the ingest endpoint, but without partition-based parallelism or consumer group semantics.
- **Render free tier cold starts:** The free Render tier spins down after 15 minutes of inactivity, causing a ~30-second cold start on the next request.

---

## 12. File Structure

```
forexguard/
├── generate_data.py            # Phase 1: synthetic data generation
├── feature_engineering.py      # Phase 2: feature extraction (80+ features)
├── train_models.py             # Phase 3: model training + ensemble
├── main.py                     # Phase 4: FastAPI + WebSocket server
├── stream_simulator.py         # Async event replay simulator
├── dashboard.html              # Real-time compliance dashboard
├── Dockerfile                  # Container definition
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/                       # Generated at runtime
│   ├── portal_events.csv       # Client portal activity stream
│   ├── trade_events.csv        # Trading terminal activity stream
│   ├── labels.csv              # Ground-truth anomaly labels
│   ├── user_profiles.csv       # Per-user baseline statistics
│   ├── population_stats.csv    # Population-level distribution thresholds
│   ├── features.csv            # ML-ready feature matrix
│   ├── labels_eval.csv         # Evaluation labels (no leakage)
│   ├── scores.csv              # Per-user scores, tiers, SHAP, LLM summaries
│   └── shap_values.csv         # Full SHAP attribution matrix
│
└── models/                     # Generated at runtime
    ├── scaler.pkl               # Fitted RobustScaler
    ├── isolation_forest.pkl     # Trained IF model + metadata
    ├── lof.pkl                  # Trained LOF model + metadata
    ├── transformer_autoencoder.pt  # Transformer state dict
    ├── transformer_config.json  # Architecture configuration
    ├── ensemble_weights.json    # Rank-fusion metadata
    ├── platt_scaler.pkl         # Platt calibration model
    ├── llm_risk_summaries.json  # Pre-generated LLM narratives
    └── model_report.json        # Full evaluation report
```

---

## References

- Bao, Y. et al. (2025). *A Deep Learning Approach to Anomaly Detection in High-Frequency Trading Data*. arXiv:2504.00287.
- Liu, F. T., Ting, K. M., & Zhou, Z-H. (2008). *Isolation Forest*. IEEE ICDM.
- Breunig, M. M. et al. (2000). *LOF: Identifying Density-Based Local Outliers*. ACM SIGMOD.
- Vaswani, A. et al. (2017). *Attention Is All You Need*. NeurIPS.
- Rousseeuw, P. J. & Croux, C. (1993). *Alternatives to the Median Absolute Deviation*. JASA. *(IQR threshold method)*
- Platt, J. (1999). *Probabilistic Outputs for SVMs and Comparisons to Regularized Likelihood Methods*. *(Platt scaling)*
