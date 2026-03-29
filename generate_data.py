"""
ForexGuard — Phase 1: Synthetic Data Generator
================================================
Generates a realistic 90-day simulation of 500 forex brokerage users,
producing ~50,000+ events across client portal and trading terminal activity.

Key design principles:
  - Statistics-driven thresholds derived from actual population distributions,
    not hardcoded magic numbers.
  - Realistic event distributions (log-normal volumes, Pareto deposits, Gamma
    session durations) to approximate real forex user behaviour.
  - Market-hours weighted timestamps reflecting London/NY session patterns.
  - Blended anomalies: suspicious events are mixed into normal behaviour at
    configurable rates (default 20%), making detection non-trivial.
  - Network-level effects: collusion rings and IP-hub groups share behaviour
    across accounts, enabling graph-based detection.
  - Concept drift: a subset of normal users shifts behaviour mid-simulation.

Output files:
  data/portal_events.csv       — client portal activity stream
  data/trade_events.csv        — trading terminal activity stream
  data/labels.csv              — ground-truth anomaly labels (for evaluation)
  data/user_profiles.csv       — per-user baseline statistics
  data/population_stats.csv    — population-level distribution thresholds
"""

import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker
from scipy import stats  # noqa: F401 — available for downstream use

fake = Faker()
random.seed(42)
np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_USERS    = 500
SIM_DAYS   = 90
START_DATE = datetime(2024, 1, 1)
END_DATE   = START_DATE + timedelta(days=SIM_DAYS)

INSTRUMENTS = [
    "EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD",
    "USD/CHF", "AUD/USD", "NZD/USD", "USD/CAD",
]

PORTAL_EVENT_TYPES = [
    "login", "logout", "deposit", "withdrawal",
    "kyc_upload", "kyc_status_change", "support_ticket",
    "account_modify", "password_change", "doc_upload",
    "page_view", "session_start", "session_end", "login_failed",
]

# Per-hour weights reflecting real forex market sessions (UTC):
#   Peak: London open (08–09), NY+London overlap (13–17), Asian session (00–02)
MARKET_HOUR_WEIGHTS = {
    0: 0.35, 1: 0.30, 2: 0.30, 3: 0.10, 4: 0.10, 5: 0.15,
    6: 0.40, 7: 0.60, 8: 0.90, 9: 1.00, 10: 0.95, 11: 0.85,
    12: 0.80, 13: 1.00, 14: 1.00, 15: 0.98, 16: 0.90, 17: 0.70,
    18: 0.50, 19: 0.40, 20: 0.35, 21: 0.30, 22: 0.30, 23: 0.30,
}
_hour_weights_arr = np.array(
    [MARKET_HOUR_WEIGHTS[h] for h in range(24)]
)
_hour_weights_arr = _hour_weights_arr / _hour_weights_arr.sum()

DEVICES    = [f"device_{i}" for i in range(1, 1200)]
BROWSERS   = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
OS_LIST    = ["Windows", "MacOS", "Linux", "Android", "iOS"]
SCREEN_RES = ["1920x1080", "2560x1440", "1366x768", "1280x720", "3840x2160"]
TIMEZONES  = [
    "UTC+0", "UTC+1", "UTC+2", "UTC+3", "UTC+5:30",
    "UTC-5", "UTC-6", "UTC-8", "UTC+8", "UTC+9",
]


# ---------------------------------------------------------------------------
# Step 1 — Build realistic user baseline profiles
# ---------------------------------------------------------------------------
# Each user is assigned baseline behavioural parameters drawn from
# domain-appropriate statistical distributions rather than uniform random.
# This ensures the population resembles a real brokerage: a few power
# users with very high volumes and many small retail traders.

print("=" * 60)
print("STEP 1: Building user profiles with realistic distributions")
print("=" * 60)


def sample_deposit_amount() -> float:
    """
    Log-normal deposit amount. Median ≈ $1,100; tail extends to $50k+.
    Parameters chosen to match typical retail forex deposit behaviour.
    """
    return max(50.0, np.random.lognormal(mean=7.0, sigma=1.1))


def sample_trade_volume() -> float:
    """
    Pareto-distributed trade volume reflecting the 80/20 principle:
    ~20% of traders account for ~80% of volume (power-law tail).
    """
    return max(500.0, np.random.pareto(a=1.8) * 15_000)


def sample_lot_size() -> float:
    """Gamma-distributed lot size; most users trade 0.01–2.0 lots."""
    return max(0.01, np.random.gamma(shape=1.5, scale=0.8))


def sample_session_duration() -> float:
    """
    Gamma-distributed session duration in minutes.
    Most sessions last 5–30 minutes; power users stay 2+ hours.
    """
    return max(1.0, np.random.gamma(shape=2.5, scale=12.0))


def sample_trades_per_day() -> float:
    """Poisson trade frequency; average active user places ~2 trades/day."""
    return max(0.1, np.random.poisson(lam=2.0))


def make_device_fingerprint() -> dict:
    """Generate a synthetic browser/OS fingerprint for a user session."""
    return {
        "browser":    random.choice(BROWSERS),
        "os":         random.choice(OS_LIST),
        "screen_res": random.choice(SCREEN_RES),
        "timezone":   random.choice(TIMEZONES),
    }


users = []
for uid in range(1, N_USERS + 1):
    users.append({
        "user_id":                  f"U{uid:04d}",
        "country":                  fake.country_code(),
        "registered_at":            START_DATE - timedelta(days=random.randint(30, 730)),
        "kyc_verified":             random.random() > 0.10,
        "typical_deposit_amt":      round(sample_deposit_amount(), 2),
        "typical_trade_volume":     round(sample_trade_volume(), 2),
        "typical_lot_size":         round(sample_lot_size(), 2),
        "typical_session_duration": round(sample_session_duration(), 2),
        "trades_per_day":           round(sample_trades_per_day(), 2),
        "preferred_instrument":     random.choice(INSTRUMENTS),
        "ips":                      [fake.ipv4() for _ in range(random.randint(1, 3))],
        "devices":                  random.sample(DEVICES, k=random.randint(1, 3)),
        "fingerprint":              make_device_fingerprint(),
        "is_anomalous":             False,
        "anomaly_labels":           [],
    })

users_df = pd.DataFrame(users)


# ---------------------------------------------------------------------------
# Step 2 — Derive population-level thresholds from the actual distribution
# ---------------------------------------------------------------------------
# All detection thresholds are computed from the simulated population's
# statistical properties rather than being hardcoded. This ensures the
# thresholds scale correctly if population parameters change.

print("\nSTEP 2: Computing population-level thresholds from distributions")

pop_stats: dict = {}
for col, label in [
    ("typical_deposit_amt",      "deposit"),
    ("typical_trade_volume",     "volume"),
    ("typical_lot_size",         "lot_size"),
    ("typical_session_duration", "session"),
    ("trades_per_day",           "trades_per_day"),
]:
    arr = users_df[col].values
    pop_stats[label] = {
        "mean": float(np.mean(arr)),
        "std":  float(np.std(arr)),
        "p01":  float(np.percentile(arr, 1)),
        "p05":  float(np.percentile(arr, 5)),
        "p50":  float(np.percentile(arr, 50)),
        "p95":  float(np.percentile(arr, 95)),
        "p99":  float(np.percentile(arr, 99)),
        "p999": float(np.percentile(arr, 99.9)),
    }

THRESH = {
    # Volume spike: whichever is more conservative — 4-sigma or 99.9th percentile
    "volume_spike_multiplier": max(
        (pop_stats["volume"]["mean"] + 4 * pop_stats["volume"]["std"])
        / pop_stats["volume"]["mean"],
        pop_stats["volume"]["p999"] / pop_stats["volume"]["mean"],
    ),
    # Structuring: deposits at or below the 5th percentile signal fragmentation
    "structuring_max_deposit":  pop_stats["deposit"]["p05"],
    # Dormant withdrawal: amounts above the 95th percentile after inactivity
    "dormant_withdrawal_min":   pop_stats["deposit"]["p95"],
    # Bot session: page-view duration below 10% of the 1st-percentile session
    "bot_session_max_sec":      pop_stats["session"]["p01"] * 60 * 0.1,
    # Latency arbitrage: any trade held under 30 seconds is suspicious
    "latency_arb_max_sec":      30,
    # Lot spike: 4-sigma or 99.9th percentile of lot size distribution
    "lot_spike_multiplier": max(
        (pop_stats["lot_size"]["mean"] + 4 * pop_stats["lot_size"]["std"])
        / max(pop_stats["lot_size"]["mean"], 0.01),
        pop_stats["lot_size"]["p999"] / max(pop_stats["lot_size"]["mean"], 0.01),
    ),
}

print(f"  Volume spike threshold : {THRESH['volume_spike_multiplier']:.1f}× user baseline")
print(f"  Structuring max deposit: ${THRESH['structuring_max_deposit']:.2f}")
print(f"  Dormant withdrawal min : ${THRESH['dormant_withdrawal_min']:.2f}")
print(f"  Bot session max (sec)  : {THRESH['bot_session_max_sec']:.2f}s")
print(f"  Lot spike threshold    : {THRESH['lot_spike_multiplier']:.1f}× user baseline")

pop_stats_df = pd.DataFrame(
    [{"metric": k, **v} for k, v in pop_stats.items()]
)


# ---------------------------------------------------------------------------
# Step 3 — Assign anomaly labels including network-level groups
# ---------------------------------------------------------------------------

print("\nSTEP 3: Assigning anomaly labels")

all_uids = list(users_df["user_id"])
random.shuffle(all_uids)

anomaly_assignments = {
    "ip_hopper":             all_uids[0:10],
    "structuring":           all_uids[10:20],
    "dormant_withdrawal":    all_uids[20:28],
    "volume_spike":          all_uids[28:36],
    "bot_session":           all_uids[36:42],
    "deposit_and_flee":      all_uids[42:50],
    "brute_force_login":     all_uids[50:58],
    "kyc_before_withdrawal": all_uids[58:65],
    "latency_arbitrage":     all_uids[65:72],
    "device_mismatch":       all_uids[72:80],
    "collusion_ring":        all_uids[80:95],   # three rings of five users each
    "ip_hub":                all_uids[95:115],  # two hubs of ten users each
}

user_anomaly_map: dict = {}
for label, uids in anomaly_assignments.items():
    for uid in uids:
        user_anomaly_map.setdefault(uid, []).append(label)

# Build collusion rings — members share near-identical trade sequences
collusion_rings = [all_uids[80:85], all_uids[85:90], all_uids[90:95]]

# IP hubs — members share a single IP address across accounts
ip_hubs = [
    {"ip": fake.ipv4(), "members": all_uids[95:105]},
    {"ip": fake.ipv4(), "members": all_uids[105:115]},
]
for hub in ip_hubs:
    for uid in hub["members"]:
        idx = users_df.index[users_df["user_id"] == uid][0]
        users_df.at[idx, "ips"] = [hub["ip"]] + [fake.ipv4()]

# Concept drift: 20 normal users shift behaviour at day 45 of the simulation
drift_users = random.sample(
    [u for u in all_uids if u not in user_anomaly_map], k=20
)
drift_day = 45

users_df["is_anomalous"]  = users_df["user_id"].isin(user_anomaly_map)
users_df["anomaly_labels"] = users_df["user_id"].map(
    lambda uid: user_anomaly_map.get(uid, [])
)

print(f"  Total anomalous users: {len(user_anomaly_map)}")
for label, uids in anomaly_assignments.items():
    print(f"    {label:<28} → {len(uids)} users")


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def market_weighted_timestamp(
    start: datetime = START_DATE,
    end: datetime   = END_DATE,
    force_night: bool = False,
) -> datetime:
    """
    Return a random timestamp weighted by real forex market session volumes.

    Args:
        start: Simulation window start.
        end: Simulation window end.
        force_night: If True, restricts the hour to 01–04 UTC (off-hours
                     activity used for suspicious login patterns).
    """
    delta_days = (end - start).days
    base_date  = start + timedelta(days=random.randint(0, delta_days - 1))
    hour = (
        random.randint(1, 4) if force_night
        else int(np.random.choice(24, p=_hour_weights_arr))
    )
    return base_date.replace(
        hour=hour,
        minute=random.randint(0, 59),
        second=random.randint(0, 59),
    )


def blend(anomalous_val, normal_val, anomaly_rate: float = 0.20):
    """
    Mix an anomalous value into otherwise normal behaviour.

    Rather than making every event of an anomalous user suspicious (which
    would be trivially detectable), this function injects the anomalous
    value with the given probability and falls back to the normal value
    the rest of the time. This reflects how real fraudsters operate.

    Args:
        anomalous_val: The suspicious value to inject.
        normal_val: The baseline normal value.
        anomaly_rate: Probability of returning the anomalous value.
    """
    return anomalous_val if random.random() < anomaly_rate else normal_val


def get_user(uid: str) -> pd.Series:
    return users_df[users_df["user_id"] == uid].iloc[0]


# ---------------------------------------------------------------------------
# Step 4 — Generate portal events
# ---------------------------------------------------------------------------

print("\nSTEP 4: Generating portal events...")

portal_rows = []
drift_boundary = START_DATE + timedelta(days=drift_day)

for _, user in users_df.iterrows():
    uid    = user["user_id"]
    labels = user["anomaly_labels"]
    is_drift_user = uid in drift_users

    # Active users receive more events proportional to their trading frequency
    n_events = max(20, int(np.random.poisson(lam=60 + user["trades_per_day"] * 10)))

    for i in range(n_events):
        ts = market_weighted_timestamp()

        event_type = random.choices(
            PORTAL_EVENT_TYPES,
            weights=[14, 10, 7, 4, 2, 2, 2, 2, 2, 2, 22, 10, 10, 3],
            k=1,
        )[0]

        ip          = random.choice(user["ips"])
        device      = random.choice(user["devices"])
        amount      = None
        session_dur = None
        extra_flag  = None
        fingerprint = dict(user["fingerprint"])

        # Concept drift: post-drift, normal users begin using new IPs
        if is_drift_user and ts > drift_boundary and event_type == "login":
            if random.random() < 0.4:
                ip = fake.ipv4()

        # IP hopper: 25% of logins originate from an unrecognised IP
        if "ip_hopper" in labels and event_type == "login":
            suspicious_ip = fake.ipv4()
            ip = blend(suspicious_ip, random.choice(user["ips"]), anomaly_rate=0.25)
            if ip == suspicious_ip:
                extra_flag = "ip_hopper"

        # Structuring: 30% of deposits are suspiciously small (below 5th percentile)
        if event_type == "deposit":
            normal_amt = round(abs(np.random.lognormal(
                mean=np.log(max(user["typical_deposit_amt"], 1)), sigma=0.3
            )), 2)
            if "structuring" in labels:
                structured_amt = round(random.uniform(
                    THRESH["structuring_max_deposit"] * 0.3,
                    THRESH["structuring_max_deposit"],
                ), 2)
                amount = blend(structured_amt, normal_amt, anomaly_rate=0.30)
                if amount == structured_amt:
                    extra_flag = "structuring"
            else:
                amount = normal_amt

        # Dormant withdrawal: large withdrawals concentrated in the final 7 days
        if event_type == "withdrawal":
            normal_amt = round(abs(np.random.lognormal(
                mean=np.log(max(user["typical_deposit_amt"] * 0.8, 1)), sigma=0.3
            )), 2)
            if "dormant_withdrawal" in labels:
                ts = market_weighted_timestamp(
                    start=END_DATE - timedelta(days=7), end=END_DATE
                )
                large_amt = round(random.uniform(
                    THRESH["dormant_withdrawal_min"],
                    THRESH["dormant_withdrawal_min"] * 3.0,
                ), 2)
                amount = blend(large_amt, normal_amt, anomaly_rate=0.70)
                if amount == large_amt:
                    extra_flag = "dormant_withdrawal"
            else:
                amount = normal_amt

        # Bot session: 20% of page views have sub-human navigation speed
        if event_type == "page_view":
            normal_dur = round(abs(np.random.gamma(shape=2, scale=2.0)), 2)
            if "bot_session" in labels:
                bot_dur = round(random.uniform(0.05, THRESH["bot_session_max_sec"]), 2)
                session_dur = blend(bot_dur, normal_dur * 60, anomaly_rate=0.20)
                if session_dur == bot_dur:
                    extra_flag = "bot_session"
            else:
                session_dur = round(normal_dur * 60, 2)

        if event_type in ("session_start", "session_end"):
            session_dur = round(abs(np.random.gamma(
                shape=2.5, scale=user["typical_session_duration"]
            )), 2)

        # Brute-force login: burst of failures followed by a successful login
        if "brute_force_login" in labels and random.random() < 0.15:
            n_failures = random.randint(8, 20)
            for _ in range(n_failures):
                portal_rows.append({
                    "user_id":             uid,
                    "timestamp":           ts + timedelta(seconds=random.uniform(1, 30)),
                    "event_type":          "login_failed",
                    "ip_address":          fake.ipv4(),
                    "device_id":           device,
                    "amount":              None,
                    "session_duration":    None,
                    "country":             user["country"],
                    "kyc_verified":        user["kyc_verified"],
                    "fingerprint_browser": fingerprint["browser"],
                    "fingerprint_os":      fingerprint["os"],
                    "anomaly_flag":        "brute_force_login",
                })
            portal_rows.append({
                "user_id":             uid,
                "timestamp":           ts + timedelta(seconds=random.uniform(31, 60)),
                "event_type":          "login",
                "ip_address":          fake.ipv4(),
                "device_id":           device,
                "amount":              None,
                "session_duration":    None,
                "country":             user["country"],
                "kyc_verified":        user["kyc_verified"],
                "fingerprint_browser": fingerprint["browser"],
                "fingerprint_os":      fingerprint["os"],
                "anomaly_flag":        "brute_force_login",
            })
            continue

        # KYC-before-withdrawal: account modification shortly before a large withdrawal
        if "kyc_before_withdrawal" in labels and event_type == "kyc_status_change":
            if random.random() < 0.5:
                future_ts = ts + timedelta(days=random.uniform(1, 3))
                portal_rows.append({
                    "user_id":             uid,
                    "timestamp":           future_ts,
                    "event_type":          "withdrawal",
                    "ip_address":          ip,
                    "device_id":           device,
                    "amount":              round(random.uniform(
                        THRESH["dormant_withdrawal_min"],
                        THRESH["dormant_withdrawal_min"] * 2,
                    ), 2),
                    "session_duration":    None,
                    "country":             user["country"],
                    "kyc_verified":        user["kyc_verified"],
                    "fingerprint_browser": fingerprint["browser"],
                    "fingerprint_os":      fingerprint["os"],
                    "anomaly_flag":        "kyc_before_withdrawal",
                })
                extra_flag = "kyc_before_withdrawal"

        # Device mismatch: 20% of logins use a mismatched browser/OS fingerprint
        if "device_mismatch" in labels and event_type == "login":
            if blend(True, False, anomaly_rate=0.20):
                fingerprint = make_device_fingerprint()
                extra_flag  = "device_mismatch"

        portal_rows.append({
            "user_id":             uid,
            "timestamp":           ts,
            "event_type":          event_type,
            "ip_address":          ip,
            "device_id":           device,
            "amount":              amount,
            "session_duration":    session_dur,
            "country":             user["country"],
            "kyc_verified":        user["kyc_verified"],
            "fingerprint_browser": fingerprint["browser"],
            "fingerprint_os":      fingerprint["os"],
            "anomaly_flag":        extra_flag,
        })

# Deposit-and-flee sequences: large deposit followed quickly by near-full withdrawal
for uid in anomaly_assignments["deposit_and_flee"]:
    user = get_user(uid)
    base_ts  = market_weighted_timestamp(
        start=START_DATE + timedelta(days=10),
        end=END_DATE - timedelta(days=3),
    )
    dep_amt  = round(random.uniform(8_000, 25_000), 2)
    with_amt = round(dep_amt * random.uniform(0.75, 0.95), 2)

    for event_type, amount, extra_flag in [
        ("deposit",    dep_amt,  "deposit_and_flee"),
        ("withdrawal", with_amt, "deposit_and_flee"),
    ]:
        portal_rows.append({
            "user_id":             uid,
            "timestamp":           base_ts if event_type == "deposit"
                                   else base_ts + timedelta(hours=random.uniform(6, 22)),
            "event_type":          event_type,
            "ip_address":          user["ips"][0],
            "device_id":           user["devices"][0],
            "amount":              amount,
            "session_duration":    None,
            "country":             user["country"],
            "kyc_verified":        user["kyc_verified"],
            "fingerprint_browser": user["fingerprint"]["browser"],
            "fingerprint_os":      user["fingerprint"]["os"],
            "anomaly_flag":        extra_flag,
        })

portal_df = pd.DataFrame(portal_rows).sort_values("timestamp").reset_index(drop=True)
portal_df["event_id"] = ["PE" + str(i).zfill(6) for i in range(len(portal_df))]
print(f"  Portal events generated: {len(portal_df):,}")


# ---------------------------------------------------------------------------
# Step 5 — Generate trade events
# ---------------------------------------------------------------------------

print("\nSTEP 5: Generating trade events...")

trade_rows = []


def gen_collusion_sequence(n_trades: int = 15) -> list:
    """
    Build a master trade sequence for a collusion ring.
    All ring members execute these trades within a 45-second jitter window,
    simulating coordinated mirror trading.
    """
    sequences = []
    base_ts = market_weighted_timestamp()
    for _ in range(n_trades):
        base_ts = base_ts + timedelta(minutes=random.randint(30, 240))
        sequences.append({
            "base_ts":   base_ts,
            "instrument": random.choice(INSTRUMENTS),
            "lot_size":  round(abs(np.random.gamma(shape=1.5, scale=0.5)), 2),
            "volume":    round(abs(np.random.lognormal(mean=10, sigma=0.5)), 2),
            "direction": random.choice(["BUY", "SELL"]),
        })
    return sequences


ring_sequences = {
    tuple(ring): gen_collusion_sequence() for ring in collusion_rings
}
user_ring_sequence = {
    uid: seq
    for ring, seq in ring_sequences.items()
    for uid in ring
}

for _, user in users_df.iterrows():
    uid    = user["user_id"]
    labels = user["anomaly_labels"]
    n_trades = max(5, int(np.random.poisson(lam=user["trades_per_day"] * SIM_DAYS)))
    is_drift_user = uid in drift_users

    # Inject collusion ring trades: near-simultaneous execution with jitter
    if "collusion_ring" in labels and uid in user_ring_sequence:
        for master in user_ring_sequence[uid]:
            trade_rows.append({
                "user_id":          uid,
                "timestamp":        master["base_ts"] + timedelta(seconds=random.uniform(0, 45)),
                "instrument":       master["instrument"],
                "lot_size":         master["lot_size"],
                "volume":           master["volume"],
                "pnl":              round(random.gauss(0, master["volume"] * 0.003), 2),
                "margin_used":      round(master["volume"] * random.uniform(0.01, 0.03), 2),
                "trade_duration_s": random.randint(300, 3600),
                "direction":        master["direction"],
                "anomaly_flag":     "collusion_ring",
            })

    for _ in range(n_trades):
        ts     = market_weighted_timestamp()
        volume = round(abs(np.random.lognormal(
            mean=np.log(max(user["typical_trade_volume"], 1)), sigma=0.25
        )), 2)
        lot_size = round(abs(np.random.gamma(
            shape=max(user["typical_lot_size"], 0.01), scale=0.8
        )), 2)
        pnl      = round(np.random.normal(0, volume * 0.004), 2)
        margin   = round(volume * random.uniform(0.01, 0.05), 2)
        dur_sec  = int(np.random.gamma(shape=2, scale=1800))
        instrument = random.choice(INSTRUMENTS)
        extra_flag = None

        # Concept drift: post-drift normal users trade more aggressively
        if is_drift_user and ts > drift_boundary:
            volume   = round(volume   * random.uniform(1.5, 2.5), 2)
            lot_size = round(lot_size * random.uniform(1.3, 2.0), 2)

        # Volume spike: 20% of trades exceed the population-derived multiplier
        if "volume_spike" in labels:
            spike_volume = round(
                user["typical_trade_volume"] * random.uniform(
                    THRESH["volume_spike_multiplier"],
                    THRESH["volume_spike_multiplier"] * 2,
                ), 2,
            )
            spike_lot = round(
                user["typical_lot_size"] * random.uniform(
                    THRESH["lot_spike_multiplier"],
                    THRESH["lot_spike_multiplier"] * 1.5,
                ), 2,
            )
            volume   = blend(spike_volume, volume,   anomaly_rate=0.20)
            lot_size = blend(spike_lot,   lot_size,  anomaly_rate=0.20)
            if volume == spike_volume:
                extra_flag = "volume_spike"

        # Latency arbitrage: 30% of trades are very short with positive PnL
        if "latency_arbitrage" in labels:
            arb_dur = random.randint(1, THRESH["latency_arb_max_sec"])
            arb_pnl = round(random.uniform(10, 500), 2)
            dur_sec = blend(arb_dur, dur_sec, anomaly_rate=0.30)
            pnl     = blend(arb_pnl, pnl,     anomaly_rate=0.30)
            if dur_sec == arb_dur:
                extra_flag = "latency_arbitrage"

        # Deposit-and-flee: minimal trades on a single instrument
        if "deposit_and_flee" in labels:
            instrument = user["preferred_instrument"]
            pnl        = round(random.uniform(-30, 30), 2)
            n_trades   = min(n_trades, 5)
            extra_flag = "deposit_and_flee"

        trade_rows.append({
            "user_id":          uid,
            "timestamp":        ts,
            "instrument":       instrument,
            "lot_size":         max(0.01, lot_size),
            "volume":           max(100.0, volume),
            "pnl":              pnl,
            "margin_used":      margin,
            "trade_duration_s": max(1, dur_sec),
            "direction":        random.choice(["BUY", "SELL"]),
            "anomaly_flag":     extra_flag,
        })

trade_df = pd.DataFrame(trade_rows).sort_values("timestamp").reset_index(drop=True)
trade_df["trade_id"] = ["TR" + str(i).zfill(6) for i in range(len(trade_df))]
print(f"  Trade events generated: {len(trade_df):,}")


# ---------------------------------------------------------------------------
# Step 6 — Compile labels and write all output files
# ---------------------------------------------------------------------------

print("\nSTEP 6: Writing output files...")

labels_rows = [
    {"user_id": uid, "anomaly_type": label}
    for uid, anomaly_list in user_anomaly_map.items()
    for label in anomaly_list
] + [
    {"user_id": uid, "anomaly_type": "concept_drift"}
    for uid in drift_users
]
labels_df = pd.DataFrame(labels_rows)

# Serialise list columns to pipe-delimited strings for CSV compatibility
users_save = users_df.copy()
users_save["ips"]           = users_save["ips"].apply("|".join)
users_save["devices"]       = users_save["devices"].apply("|".join)
users_save["anomaly_labels"] = users_save["anomaly_labels"].apply(
    lambda x: "|".join(x) if x else ""
)
users_save["fingerprint"] = users_save["fingerprint"].apply(str)

os.makedirs("data", exist_ok=True)
portal_df.to_csv("data/portal_events.csv",       index=False)
trade_df.to_csv("data/trade_events.csv",          index=False)
labels_df.to_csv("data/labels.csv",               index=False)
users_save.to_csv("data/user_profiles.csv",        index=False)
pop_stats_df.to_csv("data/population_stats.csv",  index=False)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

total = len(portal_df) + len(trade_df)
print("\n" + "=" * 60)
print("DATA GENERATION COMPLETE")
print("=" * 60)
print(f"  Portal events   : {len(portal_df):>8,}")
print(f"  Trade events    : {len(trade_df):>8,}")
print(f"  Total events    : {total:>8,}")
print(f"  Total users     : {N_USERS:>8,}")
print(f"  Anomalous users : {len(user_anomaly_map):>8,}")
print(f"  Drift users     : {len(drift_users):>8,}")
print(f"\nAnomaly type breakdown:")
for label, uids in anomaly_assignments.items():
    print(f"  {label:<28} → {len(uids):>3} users")
print(f"\nPopulation-derived thresholds:")
for k, v in THRESH.items():
    print(f"  {k:<30} : {v:.4f}")
print("\nFiles saved:")
for fname in ["portal_events.csv", "trade_events.csv", "labels.csv",
              "user_profiles.csv", "population_stats.csv"]:
    print(f"  data/{fname}")
