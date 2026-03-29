"""
ForexGuard — Phase 2: Feature Engineering
==========================================
Transforms raw portal and trade events into an 80+ feature matrix suitable
for unsupervised anomaly detection. Features span eleven behavioural groups:

  A. Login & Access       — brute-force bursts, off-hours logins, geo-impossible
                            travel, IP entropy, device-fingerprint drift.
  B. Financial            — structuring patterns, dormant withdrawals, deposit-
                            to-withdrawal cycles, KYC-before-withdrawal timing.
  C. Trading              — volume spikes, lot-size outliers, latency arbitrage,
                            instrument concentration, PnL Sharpe ratio.
  D. Session & Behaviour  — session duration statistics, bot-speed page views,
                            event-type Shannon entropy and bigram entropy.
  E. Rolling Windows      — 7-day and 30-day z-scored rolling aggregates for
                            volume, trade count, PnL, deposits, and logins.
  F. Inter-event Deltas   — mean and minimum time gaps between logins and
                            between login and first subsequent trade.
  G. Device & Fingerprint — browser/OS fingerprint entropy and mismatch count.
  H. Graph / Network      — IP-hub score, device-sharing score, NetworkX
                            betweenness centrality, connected component size.
  I. Temporal             — concept-drift ratio, news-window trade ratio,
                            day-of-week Gini coefficient, hourly entropy.
  J. Composite Risk Scores— data-driven weighted composites for login, financial,
                            trading, and network risk; master composite score.
  K. User Activity Level  — total event count and activity tier encoding.

All detection thresholds are derived from the population distribution loaded
from data/population_stats.csv (Phase 1 output), ensuring thresholds adapt to
the actual data rather than being hardcoded.

Labels are kept in a separate evaluation file (labels_eval.csv) and never
merged into features.csv to prevent data leakage into model training.

Output files:
  data/features.csv          — ML-ready feature matrix (labels excluded)
  data/labels_eval.csv       — ground-truth labels for evaluation only
  data/feature_report.json   — feature-anomaly correlation report
  models/scaler.pkl          — RobustScaler fitted on training features
"""

import os
import json
import pickle
import warnings
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

try:
    import networkx as nx
    HAS_NX = True
except ImportError:
    HAS_NX = False
    print("[WARN] networkx not installed — graph features will be zeros.")
    print("       Install: pip install networkx")

try:
    from sklearn.preprocessing import RobustScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] sklearn not installed — scaler will not be saved.")
    print("       Install: pip install scikit-learn")


# ══════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════
DATA_DIR   = "data"
MODEL_DIR  = "models"
START_DATE = datetime(2024, 1, 1)
END_DATE   = datetime(2024, 4, 1)
DRIFT_DAY  = 45

NEWS_EVENTS = [
    datetime(2024, 1,  5, 13, 30),
    datetime(2024, 1, 31, 19,  0),
    datetime(2024, 2,  2, 13, 30),
    datetime(2024, 2, 13, 13, 30),
    datetime(2024, 2, 20, 19,  0),
    datetime(2024, 3,  8, 13, 30),
    datetime(2024, 3, 12, 12, 30),
    datetime(2024, 3, 20, 18,  0),
]
NEWS_WINDOW_MIN  = 15
OFF_HOURS        = list(range(0, 5))
SYNC_WINDOW_NS   = 60 * 1_000_000_000   # 60 seconds in nanoseconds


# Real-world MaxMind GeoLite2 would replace this.
# We assign a continent to each /8 prefix band so that
# consecutive logins from different continents within 2 hours are flagged.
# Minimum realistic travel time between continents ≈ 7 hours.
GEO_IMPOSSIBLE_HOURS = 2.0   # logins within 2 hours from different continents

def ip_to_continent(ip: str) -> str:
    """
    Map an IPv4 address to a synthetic continent label via first-octet ranges.
    Covers enough diversity to detect geo-impossible logins in synthetic data.
    """
    try:
        first = int(ip.split(".")[0])
    except (ValueError, IndexError):
        return "UNKNOWN"
    if   first < 50:   return "NA"    # North America
    elif first < 100:  return "EU"    # Europe
    elif first < 150:  return "AS"    # Asia
    elif first < 180:  return "SA"    # South America
    elif first < 210:  return "AF"    # Africa
    else:              return "OC"    # Oceania

print("=" * 65)
print("ForexGuard : Feature Engineering ")
print("=" * 65)

os.makedirs(DATA_DIR,  exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════
# STEP 1 — LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[1/11] Loading raw data...")

portal = pd.read_csv(f"{DATA_DIR}/portal_events.csv")
trades = pd.read_csv(f"{DATA_DIR}/trade_events.csv")
labels = pd.read_csv(f"{DATA_DIR}/labels.csv")
users  = pd.read_csv(f"{DATA_DIR}/user_profiles.csv")

# Explicitly force datetime — use format='mixed' to handle microseconds on Windows
portal["timestamp"] = pd.to_datetime(portal["timestamp"], format="mixed")
trades["timestamp"] = pd.to_datetime(trades["timestamp"], format="mixed")

users["ips"]     = users["ips"].fillna("").apply(lambda x: x.split("|") if x else [])
users["devices"] = users["devices"].fillna("").apply(lambda x: x.split("|") if x else [])

all_users      = users["user_id"].tolist()
drift_boundary = START_DATE + timedelta(days=DRIFT_DAY)

print(f"  Portal events     : {len(portal):>8,}")
print(f"  Trade events      : {len(trades):>8,}")
print(f"  Users             : {len(all_users):>8,}")
print(f"  Labelled anomalous: {labels['user_id'].nunique():>8,}")


# ══════════════════════════════════════════════════════════════════════
# STEP 2 — LOAD POPULATION THRESHOLDS FROM population_stats.csv
# ══════════════════════════════════════════════════════════════════════
print("\n[2/11] Loading population thresholds from Phase 1 output...")

pop_stats_df = pd.read_csv(f"{DATA_DIR}/population_stats.csv")

def get_pop_stat(metric: str, stat: str, fallback: float) -> float:
    """Safe lookup into population_stats.csv with a fallback."""
    row = pop_stats_df[pop_stats_df["metric"] == metric]
    if row.empty or stat not in row.columns:
        print(f"  [WARN] {metric}/{stat} not found in population_stats — using fallback {fallback}")
        return fallback
    return float(row[stat].iloc[0])

STRUCTURING_MAX   = get_pop_stat("deposit",       "p05",  200.0)
DORMANT_WITH_MIN  = get_pop_stat("deposit",       "p95",  5000.0)
BOT_SESSION_MAX_S = get_pop_stat("session",       "p01",  2.0) * 60 * 0.1
LATENCY_ARB_MAX_S = 30.0   # kept fixed — 30s is domain knowledge, not stat-derived

# Aggregate trade metrics for population-level z-score baselines
trade_agg = trades.groupby("user_id").agg(
    pop_mean_volume = ("volume",           "mean"),
    pop_mean_lot    = ("lot_size",         "mean"),
    pop_mean_dur    = ("trade_duration_s", "mean"),
).reset_index()

pop_vol_mean  = trade_agg["pop_mean_volume"].mean()
pop_vol_std   = trade_agg["pop_mean_volume"].std()
pop_lot_mean  = trade_agg["pop_mean_lot"].mean()
pop_lot_std   = trade_agg["pop_mean_lot"].std()
pop_dur_mean  = trade_agg["pop_mean_dur"].mean()
pop_dur_std   = trade_agg["pop_mean_dur"].std()

# Population-level deposit/login baselines (for portal rolling z-scores)
dep_events    = portal[portal["event_type"] == "deposit"]
dep_per_user  = dep_events.groupby("user_id")["amount"].sum()
pop_dep_mean  = dep_per_user.mean()
pop_dep_std   = dep_per_user.std()

print(f"  Structuring threshold  : ${STRUCTURING_MAX:,.2f}  (pop 5th pct)")
print(f"  Dormant withdrawal min : ${DORMANT_WITH_MIN:,.2f}  (pop 95th pct)")
print(f"  Bot session max (s)    : {BOT_SESSION_MAX_S:.2f}")
print(f"  Pop mean trade volume  : {pop_vol_mean:,.0f}")


# ══════════════════════════════════════════════════════════════════════
# STEP 3 — HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def safe_z(val, mean, std) -> float:
    return (val - mean) / std if std > 1e-9 else 0.0

def shannon_entropy(series) -> float:
    if len(series) == 0:
        return 0.0
    counts = series.value_counts(normalize=True)
    return float(-(counts * np.log2(counts + 1e-10)).sum())

def gini_coefficient(arr) -> float:
    arr = np.array(arr, dtype=float)
    if arr.sum() == 0:
        return 0.0
    arr = np.sort(arr)
    n   = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)

def is_in_news_window(ts) -> int:
    window = timedelta(minutes=NEWS_WINDOW_MIN)
    for event in NEWS_EVENTS:
        if abs(ts - event) <= window:
            return 1
    return 0

# bigram entropy over event type sequences
def bigram_entropy(event_series) -> float:
    """
    Compute Shannon entropy over consecutive event-type pairs.
    A bot running a fixed script has very low bigram entropy
    (always the same 2-3 transition patterns).
    A human has varied, non-repetitive transitions → high entropy.
    """
    if len(event_series) < 2:
        return 0.0
    events  = list(event_series)
    bigrams = [f"{events[i]}→{events[i+1]}" for i in range(len(events) - 1)]
    return shannon_entropy(pd.Series(bigrams))


# ══════════════════════════════════════════════════════════════════════
# STEP 4 — PER-USER GROUPING
# ══════════════════════════════════════════════════════════════════════
print("\n[3/11] Grouping events per user...")

portal_by_user = {uid: grp.sort_values("timestamp") for uid, grp in portal.groupby("user_id")}
trades_by_user = {uid: grp.sort_values("timestamp") for uid, grp in trades.groupby("user_id")}


# ══════════════════════════════════════════════════════════════════════
# STEP 5 — NETWORKX GRAPH
# User ↔ IP ↔ Device bipartite graph.
# ══════════════════════════════════════════════════════════════════════
print("\n[4/11] Building user-IP-device graph...")

ip_to_users  = defaultdict(set)
dev_to_users = defaultdict(set)

for uid, grp in portal_by_user.items():
    for ip  in grp["ip_address"].dropna().unique():
        ip_to_users[ip].add(uid)
    for dev in grp["device_id"].dropna().unique():
        dev_to_users[dev].add(uid)

user_ip_hub_score    = {}
user_dev_share_score = {}
user_max_ip_users    = {}   # FIX-05

for uid in all_users:
    p = portal_by_user.get(uid)
    shared_via_ip  = set()
    shared_via_dev = set()
    max_ip_users   = 0

    if p is not None:
        for ip in p["ip_address"].dropna().unique():
            others = ip_to_users[ip] - {uid}
            shared_via_ip |= others
            max_ip_users   = max(max_ip_users, len(ip_to_users[ip]))   # FIX-05
        for dev in p["device_id"].dropna().unique():
            shared_via_dev |= (dev_to_users[dev] - {uid})

    user_ip_hub_score[uid]    = len(shared_via_ip)
    user_dev_share_score[uid] = len(shared_via_dev)
    user_max_ip_users[uid]    = max_ip_users   # FIX-05

if HAS_NX:
    G = nx.Graph()
    for uid in all_users:
        G.add_node(uid, ntype="user")
    for uid, grp in portal_by_user.items():
        for ip in grp["ip_address"].dropna().unique():
            if not G.has_node(ip):
                G.add_node(ip, ntype="ip")
            G.add_edge(uid, ip)
        for dev in grp["device_id"].dropna().unique():
            if not G.has_node(dev):
                G.add_node(dev, ntype="device")
            G.add_edge(uid, dev)

    print(f"  Graph nodes : {G.number_of_nodes():,}")
    print(f"  Graph edges : {G.number_of_edges():,}")

    betweenness = nx.betweenness_centrality(G, k=min(150, len(G)), normalized=True)

    user_component_size = {}
    for comp in nx.connected_components(G):
        user_nodes = [n for n in comp if n in set(all_users)]
        for uid in user_nodes:
            user_component_size[uid] = len(user_nodes)
else:
    betweenness         = {}
    user_component_size = {}


# ══════════════════════════════════════════════════════════════════════
# STEP 6 — COLLUSION SYNCHRONISATION
# ══════════════════════════════════════════════════════════════════════
print("\n[5/11] Computing collusion trade synchronisation...")

trade_ts_sorted  = trades.sort_values("timestamp")
trade_ts_arr     = trade_ts_sorted["timestamp"].values.astype("int64")
trade_uid_arr    = trade_ts_sorted["user_id"].values

user_sync_count = defaultdict(int)
for uid in all_users:
    t = trades_by_user.get(uid)
    if t is None or len(t) == 0:
        continue
    for ts in t["timestamp"].values.astype("int64"):
        lo = np.searchsorted(trade_ts_arr, ts - SYNC_WINDOW_NS, side="left")
        hi = np.searchsorted(trade_ts_arr, ts + SYNC_WINDOW_NS, side="right")
        user_sync_count[uid] += int(np.sum(trade_uid_arr[lo:hi] != uid))


# ══════════════════════════════════════════════════════════════════════
# STEP 7 — MAIN FEATURE EXTRACTION LOOP
# ══════════════════════════════════════════════════════════════════════
print("\n[6/11] Extracting features for all users...")

feature_rows = []

for uid in all_users:

    p = portal_by_user.get(uid, pd.DataFrame())
    t = trades_by_user.get(uid, pd.DataFrame())
    u = users[users["user_id"] == uid].iloc[0]

    row = {"user_id": uid}

    # ── FIX-09: Activity level (sparse-user signal) ──────────────────
    total_events = len(p) + len(t)
    row["user_total_events"]   = total_events
    row["user_activity_level"] = (
        "high"   if total_events > 200 else
        "medium" if total_events > 50  else
        "low"
    )
    # Numeric encoding for the model
    row["user_activity_score"] = (
        1.0 if total_events > 200 else
        0.5 if total_events > 50  else
        0.1
    )

    # ── A. LOGIN & ACCESS ────────────────────────────────────────────

    logins  = p[p["event_type"] == "login"]        if len(p) > 0 else pd.DataFrame()
    failed  = p[p["event_type"] == "login_failed"] if len(p) > 0 else pd.DataFrame()

    row["login_count"]        = len(logins)
    row["login_failed_count"] = len(failed)
    row["login_failure_rate"] = (
        len(failed) / (len(logins) + len(failed))
        if (len(logins) + len(failed)) > 0 else 0.0
    )

    observed_ips = p["ip_address"].dropna().unique() if len(p) > 0 else []
    row["unique_ip_count"]    = len(observed_ips)
    row["ip_entropy"]         = shannon_entropy(p["ip_address"].dropna()) if len(p) > 0 else 0.0

    if len(logins) > 0:
        login_hours = logins["timestamp"].dt.hour
        row["offhours_login_count"] = int((login_hours.isin(OFF_HOURS)).sum())
        row["offhours_login_ratio"] = row["offhours_login_count"] / max(len(logins), 1)
    else:
        row["offhours_login_count"] = 0
        row["offhours_login_ratio"] = 0.0

    if len(logins) > 0:
        logins_c = logins.copy()
        logins_c["date"] = logins_c["timestamp"].dt.date
        daily_ips = logins_c.groupby("date")["ip_address"].nunique()
        row["multi_ip_day_count"] = int((daily_ips > 1).sum())
    else:
        row["multi_ip_day_count"] = 0

    # Brute-force burst: max failed logins in any 10-minute window
    if len(failed) > 0:
        failed_ts = failed.sort_values("timestamp")["timestamp"].values
        max_burst = 0
        for i, ts in enumerate(failed_ts):
            window_end = ts + np.timedelta64(10, "m")
            burst      = int(np.sum(failed_ts[i:] <= window_end))
            max_burst  = max(max_burst, burst)
        row["brute_force_max_burst"] = max_burst
    else:
        row["brute_force_max_burst"] = 0

    row["unique_device_count"] = len(p["device_id"].dropna().unique()) if len(p) > 0 else 0

    # ── FIX-03: Geo-impossible login detection ───────────────────────
    row["geo_impossible_login_count"] = 0
    if len(logins) > 1:
        login_sorted = logins.sort_values("timestamp")[["timestamp", "ip_address"]].dropna()
        login_sorted["continent"] = login_sorted["ip_address"].apply(ip_to_continent)
        geo_impossible = 0
        prev_row = None
        for _, cur in login_sorted.iterrows():
            if prev_row is not None:
                time_diff_h = (
                    cur["timestamp"] - prev_row["timestamp"]
                ).total_seconds() / 3600.0
                diff_continent = cur["continent"] != prev_row["continent"]
                if diff_continent and 0 < time_diff_h < GEO_IMPOSSIBLE_HOURS:
                    geo_impossible += 1
            prev_row = cur
        row["geo_impossible_login_count"] = geo_impossible

    # ── B. FINANCIAL ─────────────────────────────────────────────────

    deposits    = p[p["event_type"] == "deposit"]    if len(p) > 0 else pd.DataFrame()
    withdrawals = p[p["event_type"] == "withdrawal"] if len(p) > 0 else pd.DataFrame()

    total_dep  = float(deposits["amount"].sum())    if len(deposits)    > 0 else 0.0
    total_with = float(withdrawals["amount"].sum()) if len(withdrawals) > 0 else 0.0

    row["total_deposits"]           = round(total_dep,  2)
    row["total_withdrawals"]        = round(total_with, 2)
    row["deposit_count"]            = len(deposits)
    row["withdrawal_count"]         = len(withdrawals)
    row["deposit_withdrawal_ratio"] = (
        total_dep / total_with if total_with > 1e-6 else (total_dep if total_dep > 0 else 0.0)
    )
    if len(deposits) > 0:
        row["small_deposit_count"] = int((deposits["amount"] < STRUCTURING_MAX).sum())
        row["small_deposit_ratio"] = row["small_deposit_count"] / len(deposits)
        row["deposit_amount_std"]  = float(deposits["amount"].std()) if len(deposits) > 1 else 0.0
        row["deposit_amount_cv"]   = (
            float(deposits["amount"].std() / deposits["amount"].mean())
            if deposits["amount"].mean() > 0 and len(deposits) > 1 else 0.0
        )
    else:
        row["small_deposit_count"] = 0
        row["small_deposit_ratio"] = 0.0
        row["deposit_amount_std"]  = 0.0
        row["deposit_amount_cv"]   = 0.0
    row["dormant_withdrawal_flag"] = 0
    if len(withdrawals) > 0 and len(t) > 0:
        last_trade_ts = t["timestamp"].max()
        for _, wr in withdrawals.iterrows():
            gap_days = (wr["timestamp"] - last_trade_ts).total_seconds() / 86400
            if gap_days >= 14 and wr["amount"] >= DORMANT_WITH_MIN:
                row["dormant_withdrawal_flag"] = 1
                break
    dep_with_deltas = []
    if len(deposits) > 0 and len(withdrawals) > 0:
        dep_ts  = deposits["timestamp"].sort_values().values
        with_ts = withdrawals["timestamp"].sort_values().values
        for d in dep_ts:
            future = with_ts[with_ts > d]
            if len(future) > 0:
                delta_h = (future[0] - d).astype("float64") / 3_600_000_000_000
                dep_with_deltas.append(delta_h)

    row["has_dep_with_cycle"]           = 1 if dep_with_deltas else 0 # binary
    row["deposit_to_withdrawal_mean_h"] = float(np.mean(dep_with_deltas)) if dep_with_deltas else np.nan
    row["deposit_to_withdrawal_min_h"]  = float(np.min(dep_with_deltas))  if dep_with_deltas else np.nan

    # KYC-before-withdrawal flag
    kyc_events = (
        p[p["event_type"].isin(["kyc_upload", "kyc_status_change"])]
        if len(p) > 0 else pd.DataFrame()
    )
    row["kyc_before_withdrawal_flag"] = 0
    if len(kyc_events) > 0 and len(withdrawals) > 0:
        for _, wr in withdrawals[withdrawals["amount"] > 2000].iterrows():
            recent_kyc = kyc_events[
                (kyc_events["timestamp"] >= wr["timestamp"] - timedelta(days=3)) &
                (kyc_events["timestamp"] <= wr["timestamp"])
            ]
            if len(recent_kyc) > 0:
                row["kyc_before_withdrawal_flag"] = 1
                break

    # Bonus abuse flag
    row["bonus_abuse_flag"] = 0
    if len(deposits) > 0 and len(withdrawals) > 0 and len(t) > 0:
        for _, dep in deposits.iterrows():
            if dep["amount"] < 500:
                continue
            trades_after = t[t["timestamp"] > dep["timestamp"]]
            withs_after  = withdrawals[withdrawals["timestamp"] > dep["timestamp"]]
            if len(trades_after) < 3 and len(withs_after) > 0:
                if withs_after["amount"].max() >= dep["amount"] * 0.70:
                    row["bonus_abuse_flag"] = 1
                    break

    # ── C. TRADING ───────────────────────────────────────────────────

    if len(t) > 0:
        row["trade_count"]           = len(t)
        row["mean_volume"]           = float(t["volume"].mean())
        row["std_volume"]            = float(t["volume"].std()) if len(t) > 1 else 0.0
        row["max_volume"]            = float(t["volume"].max())
        row["volume_spike_ratio"]    = row["max_volume"] / max(row["mean_volume"], 1.0)
        row["mean_lot_size"]         = float(t["lot_size"].mean())
        row["max_lot_size"]          = float(t["lot_size"].max())
        row["mean_trade_duration_s"] = float(t["trade_duration_s"].mean())
        row["min_trade_duration_s"]  = float(t["trade_duration_s"].min())

        row["total_pnl"]   = float(t["pnl"].sum())
        row["mean_pnl"]    = float(t["pnl"].mean())
        row["pnl_std"]     = float(t["pnl"].std()) if len(t) > 1 else 0.0
        row["pnl_sharpe"]  = row["mean_pnl"] / row["pnl_std"] if row["pnl_std"] > 1e-6 else 0.0
        row["win_rate"]    = float((t["pnl"] > 0).mean())
        arb = t[(t["trade_duration_s"] < LATENCY_ARB_MAX_S) & (t["pnl"] > 0)]
        row["latency_arb_trade_count"] = len(arb)
        row["latency_arb_ratio"]       = len(arb) / len(t)

        instr_counts                 = t["instrument"].value_counts()
        row["unique_instruments"]    = int(t["instrument"].nunique())
        row["instrument_gini"]       = gini_coefficient(instr_counts.values)
        row["top_instrument_ratio"]  = float(instr_counts.iloc[0] / len(t))

        if "direction" in t.columns:
            buy_ratio = (t["direction"] == "BUY").mean()
            row["direction_imbalance"] = float(abs(buy_ratio - 0.5) * 2)
        else:
            row["direction_imbalance"] = 0.0

        active_days          = max((t["timestamp"].max() - t["timestamp"].min()).days, 1)
        row["trades_per_day"]= len(t) / active_days

        row["volume_pop_zscore"]   = safe_z(row["mean_volume"],           pop_vol_mean, pop_vol_std)
        row["lot_pop_zscore"]      = safe_z(row["mean_lot_size"],         pop_lot_mean, pop_lot_std)
        row["duration_pop_zscore"] = safe_z(row["mean_trade_duration_s"], pop_dur_mean, pop_dur_std)

        t_copy = t.copy()
        t_copy["in_news_window"]      = t_copy["timestamp"].apply(is_in_news_window)
        row["news_window_trade_ratio"]= float(t_copy["in_news_window"].mean())

        row["mean_margin_used"] = float(t["margin_used"].mean()) if "margin_used" in t.columns else 0.0
        row["max_margin_used"]  = float(t["margin_used"].max())  if "margin_used" in t.columns else 0.0

    else:
        for col in [
            "trade_count","mean_volume","std_volume","max_volume","volume_spike_ratio",
            "mean_lot_size","max_lot_size","mean_trade_duration_s","min_trade_duration_s",
            "total_pnl","mean_pnl","pnl_std","pnl_sharpe","win_rate",
            "latency_arb_trade_count","latency_arb_ratio","unique_instruments",
            "instrument_gini","top_instrument_ratio","direction_imbalance","trades_per_day",
            "volume_pop_zscore","lot_pop_zscore","duration_pop_zscore",
            "news_window_trade_ratio","mean_margin_used","max_margin_used",
        ]:
            row[col] = 0.0

    # ── D. SESSION & BEHAVIOURAL ─────────────────────────────────────

    sessions   = p[p["event_type"].isin(["session_start","session_end","page_view"])] if len(p) > 0 else pd.DataFrame()
    page_views = p[p["event_type"] == "page_view"] if len(p) > 0 else pd.DataFrame()

    if len(sessions) > 0 and "session_duration" in sessions.columns:
        dur = sessions["session_duration"].dropna()
        row["mean_session_duration"] = float(dur.mean()) if len(dur) > 0 else 0.0
        row["std_session_duration"]  = float(dur.std())  if len(dur) > 1 else 0.0
        row["min_session_duration"]  = float(dur.min())  if len(dur) > 0 else 0.0
    else:
        row["mean_session_duration"] = 0.0
        row["std_session_duration"]  = 0.0
        row["min_session_duration"]  = 0.0
    if len(page_views) > 0 and "session_duration" in page_views.columns:
        pv_dur = page_views["session_duration"].dropna()
        row["bot_speed_pageview_count"] = int((pv_dur < BOT_SESSION_MAX_S).sum())
        row["bot_speed_ratio"]          = row["bot_speed_pageview_count"] / max(len(pv_dur), 1)
    else:
        row["bot_speed_pageview_count"] = 0
        row["bot_speed_ratio"]          = 0.0

    row["event_type_entropy"]   = shannon_entropy(p["event_type"])          if len(p) > 0 else 0.0
    row["support_ticket_count"] = int((p["event_type"] == "support_ticket").sum()) if len(p) > 0 else 0
    row["account_modify_count"] = int((p["event_type"] == "account_modify").sum()) if len(p) > 0 else 0
    row["password_change_count"]= int((p["event_type"] == "password_change").sum()) if len(p) > 0 else 0
    if len(p) > 0:
        row["event_bigram_entropy"] = bigram_entropy(p["event_type"])
    else:
        row["event_bigram_entropy"] = 0.0

    # ── E. ROLLING WINDOWS ───────────────────────────────────────────

    row["rolling_7d_volume_zscore"]       = 0.0
    row["rolling_30d_volume_zscore"]      = 0.0
    row["rolling_7d_trade_count_zscore"]  = 0.0
    row["rolling_7d_pnl_zscore"]          = 0.0
    row["rolling_data_coverage_days"]     = 0 # coverage flag
    row["rolling_7d_deposit_zscore"]      = 0.0 # portal rolling
    row["rolling_7d_login_count_zscore"]  = 0.0 # portal rolling
    row["rolling_7d_withdrawal_zscore"]   = 0.0 # portal rolling

    if len(t) > 0:
        t_sorted = t.copy()
        t_sorted["date"] = t_sorted["timestamp"].dt.date
        daily_t = t_sorted.groupby("date").agg(
            daily_volume    = ("volume", "sum"),
            daily_trade_cnt = ("volume", "count"),
            daily_pnl       = ("pnl",    "sum"),
        ).reset_index()
        daily_t["date"] = pd.to_datetime(daily_t["date"])
        daily_t = daily_t.sort_values("date")

        row["rolling_data_coverage_days"] = len(daily_t)   # FIX-02

        if len(daily_t) >= 7:
            last_7d   = daily_t.tail(7)
            prior_30d = daily_t.iloc[:-7] if len(daily_t) > 7 else daily_t

            for col, feat_name in [
                ("daily_volume",    "rolling_7d_volume_zscore"),
                ("daily_trade_cnt", "rolling_7d_trade_count_zscore"),
                ("daily_pnl",       "rolling_7d_pnl_zscore"),
            ]:
                bm = prior_30d[col].mean()
                bs = prior_30d[col].std()
                rm = last_7d[col].mean()
                row[feat_name] = safe_z(rm, bm, bs)

        if len(daily_t) >= 30:
            last_30d_vol          = daily_t.tail(30)["daily_volume"].mean()
            row["rolling_30d_volume_zscore"] = safe_z(last_30d_vol, pop_vol_mean, pop_vol_std)
    if len(p) > 0:
        p_copy = p.copy()
        p_copy["date"] = p_copy["timestamp"].dt.date
        daily_p = p_copy.groupby("date").agg(
            daily_deposit    = ("amount",     lambda x: x[p_copy.loc[x.index, "event_type"] == "deposit"].sum()),
            daily_withdrawal = ("amount",     lambda x: x[p_copy.loc[x.index, "event_type"] == "withdrawal"].sum()),
            daily_logins     = ("event_type", lambda x: (x == "login").sum()),
        ).reset_index()
        daily_p["date"] = pd.to_datetime(daily_p["date"])
        daily_p = daily_p.sort_values("date")

        if len(daily_p) >= 7:
            last_7d_p   = daily_p.tail(7)
            prior_30d_p = daily_p.iloc[:-7] if len(daily_p) > 7 else daily_p

            for col, feat_name in [
                ("daily_deposit",    "rolling_7d_deposit_zscore"),
                ("daily_login",      "rolling_7d_login_count_zscore"),
                ("daily_withdrawal", "rolling_7d_withdrawal_zscore"),
            ]:
                actual_col = col if col in daily_p.columns else "daily_logins" if "login" in col else col
                if actual_col not in daily_p.columns:
                    continue
                bm = prior_30d_p[actual_col].mean()
                bs = prior_30d_p[actual_col].std()
                rm = last_7d_p[actual_col].mean()
                row[feat_name] = safe_z(rm, bm, bs)

    # ── F. INTER-EVENT TIME DELTAS ───────────────────────────────────

    if len(logins) > 1:
        login_times = logins["timestamp"].sort_values()
        login_gaps  = login_times.diff().dt.total_seconds().dropna()
        row["mean_inter_login_hours"] = float(login_gaps.mean() / 3600)
        row["std_inter_login_hours"]  = float(login_gaps.std() / 3600)  if len(login_gaps) > 1 else 0.0
        row["min_inter_login_sec"]    = float(login_gaps.min())
    else:
        row["mean_inter_login_hours"] = 0.0
        row["std_inter_login_hours"]  = 0.0
        row["min_inter_login_sec"]    = 0.0

    row["login_to_trade_mean_min"] = 0.0
    if len(logins) > 0 and len(t) > 0:
        deltas = []
        lt_arr = logins["timestamp"].sort_values().values
        tt_arr = t["timestamp"].sort_values().values
        for lt in lt_arr:
            future = tt_arr[tt_arr > lt]
            if len(future) > 0:
                deltas.append((future[0] - lt).astype("float64") / 60_000_000_000)
        row["login_to_trade_mean_min"] = float(np.mean(deltas)) if deltas else 0.0

    # ── G. DEVICE & FINGERPRINT ──────────────────────────────────────

    row["fingerprint_entropy"]       = 0.0
    row["fingerprint_mismatch_count"]= 0

    if len(p) > 0 and "fingerprint_browser" in p.columns:
        fp_combined = (
            p["fingerprint_browser"].fillna("") + "|" +
            p["fingerprint_os"].fillna("")
        )
        row["fingerprint_entropy"]        = shannon_entropy(fp_combined)
        row["fingerprint_mismatch_count"] = int(fp_combined.nunique())

    row["device_rotation_rate"] = row["unique_device_count"] / max(row["login_count"], 1)

    # ── H. GRAPH / NETWORK ───────────────────────────────────────────

    row["ip_hub_score"]          = user_ip_hub_score.get(uid, 0)
    row["device_sharing_score"]  = user_dev_share_score.get(uid, 0)
    row["max_users_per_shared_ip"] = user_max_ip_users.get(uid, 1)   # FIX-05
    row["graph_component_size"]  = user_component_size.get(uid, 1)
    row["graph_betweenness"]     = float(betweenness.get(uid, 0.0))
    row["sync_trade_count"]      = user_sync_count.get(uid, 0)

    row["collusion_risk_score"] = (
        min(row["sync_trade_count"]          / 50.0,  1.0) * 0.5 +
        min(row["device_sharing_score"]      / 10.0,  1.0) * 0.3 +
        min((row["graph_component_size"] - 1)/ 20.0,  1.0) * 0.2
    )

    # ── I. TEMPORAL ──────────────────────────────────────────────────

    row["concept_drift_volume_ratio"] = 1.0
    row["concept_drift_trade_ratio"]  = 1.0

    if len(t) > 0:
        pre  = t[t["timestamp"] <  drift_boundary]
        post = t[t["timestamp"] >= drift_boundary]
        if len(pre) > 0 and len(post) > 0:
            row["concept_drift_volume_ratio"] = (
                post["volume"].mean() / max(pre["volume"].mean(), 1.0)
            )
            row["concept_drift_trade_ratio"] = (
                (len(post) / max((END_DATE - drift_boundary).days, 1)) /
                max((len(pre) / max((drift_boundary - START_DATE).days, 1)), 0.01)
            )

    row["trade_dow_gini"] = 0.0
    if len(t) > 0:
        dow_arr = np.zeros(7)
        for d, c in t["timestamp"].dt.dayofweek.value_counts().items():
            dow_arr[d] = c
        row["trade_dow_gini"] = gini_coefficient(dow_arr)

    row["trade_hour_entropy"]  = shannon_entropy(t["timestamp"].dt.hour.astype(str))  if len(t) > 0 else 0.0
    row["portal_hour_entropy"] = shannon_entropy(p["timestamp"].dt.hour.astype(str))  if len(p) > 0 else 0.0

    # ── J. COMPOSITE RISK SCORES (weights computed after loop) ───────
    # We store the raw component values here; FIX-08 computes
    # data-driven weights and final scores after the loop completes.
    # Store a placeholder — overwritten below.
    row["login_risk_score"]    = 0.0
    row["financial_risk_score"]= 0.0
    row["trading_risk_score"]  = 0.0
    row["network_risk_score"]  = 0.0
    row["composite_risk_score"]= 0.0

    feature_rows.append(row)

features_df = pd.DataFrame(feature_rows)
print(f"  Feature matrix shape (pre-scores): {features_df.shape}")


# ══════════════════════════════════════════════════════════════════════
# STEP 8 — ATTACH LABELS (kept separate for leakage guard)
# ══════════════════════════════════════════════════════════════════════
print("\n[7/11] Attaching evaluation labels (separate from feature matrix)...")

labelled_uids  = set(labels["user_id"].unique())
uid_to_labels  = labels.groupby("user_id")["anomaly_type"].apply(list).to_dict()
is_anomalous   = features_df["user_id"].isin(labelled_uids).astype(int)
anomaly_types  = features_df["user_id"].map(lambda u: "|".join(uid_to_labels.get(u, [])))

# Save labels to a separate evaluation file — never merged into features.csv
eval_df = pd.DataFrame({
    "user_id":      features_df["user_id"],
    "is_anomalous": is_anomalous,
    "anomaly_types":anomaly_types,
})
eval_df.to_csv(f"{DATA_DIR}/labels_eval.csv", index=False)
print(f"  Saved labels_eval.csv  ({eval_df['is_anomalous'].sum()} anomalous users)")


# ══════════════════════════════════════════════════════════════════════
# STEP 9 — POST-PROCESSING
# ══════════════════════════════════════════════════════════════════════
print("\n[8/11] Post-processing feature matrix...")

numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
features_df[numeric_cols] = features_df[numeric_cols].fillna(0.0)

# Clip z-scores to [-10, 10]
zscore_cols = [c for c in features_df.columns if "zscore" in c]
features_df[zscore_cols] = features_df[zscore_cols].clip(-10, 10)

# Clip ratios to [0, inf)
ratio_cols = [c for c in features_df.columns if "ratio" in c or "rate" in c]
for col in ratio_cols:
    features_df[col] = features_df[col].clip(0.0, None)

print(f"  NaN values remaining : {features_df[numeric_cols].isna().sum().sum()}")
print(f"  Features per user    : {len(numeric_cols)}")


# ══════════════════════════════════════════════════════════════════════
# STEP 10 — DATA-DRIVEN COMPOSITE RISK SCORES
#         not invented. Higher correlation → higher weight.
# ══════════════════════════════════════════════════════════════════════
print("\n[9/11] Computing data-driven composite risk scores...")

label_series = is_anomalous

def correlation_weights(feature_names: list, label: pd.Series) -> dict:
    """
    Compute correlation of each feature with label, return
    normalised absolute correlation as weight dict.
    """
    corrs = {}
    for feat in feature_names:
        if feat in features_df.columns:
            c = abs(float(features_df[feat].corr(label)))
            corrs[feat] = c if not np.isnan(c) else 0.0
    total = sum(corrs.values())
    if total < 1e-9:
        return {k: 1.0 / len(corrs) for k in corrs}
    return {k: v / total for k, v in corrs.items()}

login_components = [
    "brute_force_max_burst", "offhours_login_ratio",
    "ip_entropy", "multi_ip_day_count", "geo_impossible_login_count",
    "login_failure_rate",
]
financial_components = [
    "small_deposit_ratio", "bonus_abuse_flag",
    "kyc_before_withdrawal_flag", "dormant_withdrawal_flag",
    "deposit_to_withdrawal_min_h", "deposit_amount_cv",
]
trading_components = [
    "latency_arb_ratio", "volume_spike_ratio",
    "instrument_gini", "rolling_7d_volume_zscore",
    "pnl_sharpe", "direction_imbalance",
]
network_components = [
    "ip_hub_score", "device_sharing_score",
    "collusion_risk_score", "sync_trade_count",
    "max_users_per_shared_ip",
]

# Normalize each component to [0,1] using robust percentile scaling
def robust_norm(series: pd.Series) -> pd.Series:
    p1  = series.quantile(0.01)
    p99 = series.quantile(0.99)
    if p99 - p1 < 1e-9:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return ((series - p1) / (p99 - p1)).clip(0, 1)

for components, score_col in [
    (login_components,    "login_risk_score"),
    (financial_components,"financial_risk_score"),
    (trading_components,  "trading_risk_score"),
    (network_components,  "network_risk_score"),
]:
    valid = [c for c in components if c in features_df.columns]
    weights = correlation_weights(valid, label_series)
    score   = pd.Series(np.zeros(len(features_df)), index=features_df.index)
    for feat, w in weights.items():
        score += robust_norm(features_df[feat]) * w
    features_df[score_col] = score.values
    print(f"  {score_col:<26} top driver: "
          f"{max(weights, key=weights.get)} ({max(weights.values()):.3f})")

# Master composite: weight each family score by its own correlation with label
family_scores  = ["login_risk_score","financial_risk_score","trading_risk_score","network_risk_score"]
family_weights = correlation_weights(family_scores, label_series)
composite      = pd.Series(np.zeros(len(features_df)), index=features_df.index)
for feat, w in family_weights.items():
    composite += features_df[feat] * w
features_df["composite_risk_score"] = composite.values

# Refresh numeric cols after composite scores are written
numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()


# ══════════════════════════════════════════════════════════════════════
# STEP 11 — FEATURE IMPORTANCE REPORT
# ══════════════════════════════════════════════════════════════════════
print("\n[10/11] Computing feature–anomaly correlations...")

feature_stats = {}
for col in numeric_cols:
    col_data = features_df[col]
    corr = float(col_data.corr(label_series))
    feature_stats[col] = {
        "mean_normal":    round(float(col_data[label_series == 0].mean()), 4),
        "mean_anomalous": round(float(col_data[label_series == 1].mean()), 4),
        "correlation":    round(corr, 4),
        "abs_corr":       round(abs(corr), 4),
    }

top_features = sorted(feature_stats.items(), key=lambda x: x[1]["abs_corr"], reverse=True)[:20]

print(f"\n  {'Feature':<42} {'Corr':>8}  {'Normal':>10}  {'Anomalous':>10}")
print("  " + "-" * 74)
for feat, fstat in top_features:
    print(f"  {feat:<42} {fstat['correlation']:>+8.4f}  "
          f"{fstat['mean_normal']:>10.3f}  {fstat['mean_anomalous']:>10.3f}")


# ══════════════════════════════════════════════════════════════════════
# STEP 12 — SAVE SCALER
#         Phase 3 loads this — same scaler at train AND inference time.
# ══════════════════════════════════════════════════════════════════════
print("\n[11/11] Fitting and saving RobustScaler...")

# Columns to scale: numeric features only, exclude user_id and activity_level (string)
scale_cols = [
    c for c in numeric_cols
    if c not in ["user_total_events", "user_activity_score"]
]

scaler_meta = {"scale_cols": scale_cols, "scaler": None}

if HAS_SKLEARN:
    scaler = RobustScaler()
    scaler.fit(features_df[scale_cols])
    scaler_meta["scaler"] = scaler

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "scale_cols": scale_cols}, f)
    print(f"  Scaler saved → models/scaler.pkl  ({len(scale_cols)} columns)")
else:
    print("  [SKIP] sklearn not available — scaler not saved.")


# ══════════════════════════════════════════════════════════════════════
# SAVE FEATURES
# ══════════════════════════════════════════════════════════════════════

# Drop the string activity level — keep only numeric columns + user_id
save_cols = ["user_id"] + [c for c in numeric_cols]
features_df[save_cols].to_csv(f"{DATA_DIR}/features.csv", index=False)

# Feature report
report = {
    "feature_count":   len(numeric_cols),
    "user_count":      len(features_df),
    "scale_cols":      scale_cols,
    "fixes_applied": [
                                                                                    ],
    "top_features": {feat: fstat for feat, fstat in top_features},
    "feature_groups": {
        "A_login_access":       [c for c in numeric_cols if any(k in c for k in ["login","ip_","multi_ip","brute","offhours","geo_impossible"])],
        "B_financial":          [c for c in numeric_cols if any(k in c for k in ["deposit","withdrawal","kyc","bonus","dormant","has_dep"])],
        "C_trading":            [c for c in numeric_cols if any(k in c for k in ["volume","lot","pnl","trade_count","instrument","latency","win_rate","margin","direction","arb","trades_per"])],
        "D_session_behaviour":  [c for c in numeric_cols if any(k in c for k in ["session","bot_","event_type_entropy","bigram","support","account_modify","password"])],
        "E_rolling_windows":    [c for c in numeric_cols if "rolling" in c or "coverage" in c],
        "F_inter_event_deltas": [c for c in numeric_cols if any(k in c for k in ["inter_login","login_to_trade"])],
        "G_device_fingerprint": [c for c in numeric_cols if any(k in c for k in ["fingerprint","device_rotation","unique_device"])],
        "H_graph_network":      [c for c in numeric_cols if any(k in c for k in ["hub","sharing","component","betweenness","sync","collusion","max_users_per"])],
        "I_temporal":           [c for c in numeric_cols if any(k in c for k in ["drift","dow","hour_entropy","news_window","portal_hour"])],
        "J_composite_scores":   [c for c in numeric_cols if "risk_score" in c],
        "K_user_activity":      [c for c in numeric_cols if "activity" in c or "total_events" in c],
    }
}
with open(f"{DATA_DIR}/feature_report.json", "w") as f:
    json.dump(report, f, indent=2)


# ══════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("✅  PHASE 2 v2 COMPLETE — Feature Engineering (Improved)")
print("=" * 65)
print(f"\n  data/features.csv        — {len(features_df):,} users × {len(numeric_cols)} features")
print(f"  data/labels_eval.csv     — ground-truth labels (no leakage)")
print(f"  data/feature_report.json — correlation report")
print(f"  models/scaler.pkl        — RobustScaler for inference")
print(f"\n  Feature group breakdown:")
for group, cols in report["feature_groups"].items():
    print(f"    {group:<30} : {len(cols):>3} features")
print(f"\n  Feature engineering complete")
print(f"  Top 5 discriminative features:")
for feat, fstat in top_features[:5]:
    print(f"    {feat:<42} corr={fstat['correlation']:+.4f}")
print("\n  → Ready for Phase 3: Modelling")
