"""
ForexGuard — Phase 4: Real-Time FastAPI + WebSocket Pipeline
=============================================================
Production-grade inference API exposing the trained ensemble model
with real-time event ingestion, WebSocket alert streaming, and an
interactive compliance dashboard.

Key capabilities:

  REST endpoints:
    POST /score          — full blended risk profile (60% historical + 40% live)
    POST /predict        — raw feature inference for arbitrary feature dicts
    GET  /alerts         — paginated list of flagged users with tier filtering
    POST /stream/ingest  — ingest a single real-time event
    GET  /stream/status  — pipeline metrics and recent alert list

  WebSocket endpoints:
    /ws/alerts  — real-time push channel for CRITICAL/HIGH alerts
    /ws/events  — raw event echo for monitoring and debugging

  Real-time scoring pipeline:
    Per-event incremental feature update maintains a rolling feature
    snapshot for each user. Events are buffered (configurable window size)
    and the full model stack is re-run periodically. Certain event patterns
    trigger immediate re-scoring regardless of buffer fill level:
      - Three or more failed logins within 10 minutes (brute-force)
      - Trade volume exceeding 5× the user's rolling mean (volume spike)
      - Withdrawal of $5,000 or more (large withdrawal)
      - Two logins from different continents within 2 hours (geo-impossible)

  Blended scoring:
    Historical scores from scores.csv (pre-computed in Phase 3) are blended
    with live inference scores using a 60/40 ratio, giving stability against
    single-event noise while remaining reactive to sustained anomalous patterns.

  LLM summaries:
    Pre-generated Groq narratives are served from cache. Live scoring
    generates rule-based summaries when the cache has no entry.

  Deployment:
    Configured for Render and HuggingFace Spaces via CORS allow_origins=["*"]
    and auto-detection of the API base URL in the dashboard.
"""


import os
import json
import math
import pickle
import asyncio
import logging
import warnings
from datetime import datetime, timedelta
from collections import defaultdict, deque
from typing import Optional, List, Dict, Any, Set

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("forexguard")

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from groq import Groq
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    HAS_GROQ     = bool(GROQ_API_KEY)
    groq_client  = Groq(api_key=GROQ_API_KEY) if HAS_GROQ else None
except ImportError:
    HAS_GROQ    = False
    groq_client = None

DATA_DIR        = "data"
MODEL_DIR       = "models"
STREAM_WINDOW   = 100          # events before live re-inference (high = faster simulation)
ALERT_THRESHOLD = 0.50
GROQ_MODEL      = "llama-3.3-70b-versatile"

# How many minutes must pass before the same user can generate another alert
ALERT_COOLDOWN_MINUTES = 30

# Cache live scoring results — re-score only after this many NEW events since last score
RESCORE_MIN_NEW_EVENTS = 50    # don't re-run ML until 50 more events arrive per user

# Geo-impossible detection (same as feature_engineering.py)
GEO_IMPOSSIBLE_HOURS = 2.0

def ip_to_continent(ip: str) -> str:
    try:
        first = int(ip.split(".")[0])
    except (ValueError, IndexError):
        return "UNKNOWN"
    if   first < 50:  return "NA"
    elif first < 100: return "EU"
    elif first < 150: return "AS"
    elif first < 180: return "SA"
    elif first < 210: return "AF"
    else:             return "OC"

# ── Pydantic models ────────────────────────────────────────────────────────────

class EventPayload(BaseModel):
    user_id:    str              = Field(..., example="U0042")
    event_type: str              = Field(..., example="login")
    timestamp:  str              = Field(..., example="2024-03-15T14:32:00")
    ip_address: Optional[str]   = None
    device_id:  Optional[str]   = None
    amount:     Optional[float] = None
    volume:     Optional[float] = None
    pnl:        Optional[float] = None
    instrument: Optional[str]   = None
    metadata:   Optional[Dict[str, Any]] = None

class RawFeaturePayload(BaseModel):
    user_id:  Optional[str]    = Field(None, example="UNKNOWN_001")
    features: Dict[str, float] = Field(..., example={"ip_entropy": 3.5, "login_failure_rate": 0.8})

class ScoreResponse(BaseModel):
    user_id:              str
    if_score:             float
    lof_score:            float
    trans_score:          float
    ensemble_score:       float
    calibrated_score:     float
    alert_tier:           str
    ensemble_flagged:     int
    window_7d_score:      float
    recent_spike_flag:    int
    shap_top1_feature:    Optional[str]
    shap_top1_value:      Optional[float]
    shap_top2_feature:    Optional[str]
    shap_top2_value:      Optional[float]
    shap_top3_feature:    Optional[str]
    shap_top3_value:      Optional[float]
    trans_top1_feature:   Optional[str]
    trans_top1_err:       Optional[float]
    llm_risk_summary:     Optional[str]
    composite_risk_score: Optional[float]
    financial_risk_score: Optional[float]
    trading_risk_score:   Optional[float]
    network_risk_score:   Optional[float]
    scored_at:            str
    score_source:         str = "precomputed"

class PredictResponse(BaseModel):
    user_id:        str
    if_score:       float
    lof_score:      float
    ensemble_score: float
    alert_tier:     str
    flagged:        bool
    top_features:   List[Dict[str, Any]]
    llm_summary:    Optional[str]
    scored_at:      str

class AlertSummary(BaseModel):
    total_users:       int
    flagged_users:     int
    critical_count:    int
    high_count:        int
    medium_count:      int
    low_count:         int
    recent_spikes:     int
    top_anomaly_types: List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ══════════════════════════════════════════════════════════════════════════════

class ModelRegistry:
    def __init__(self):
        self.if_bundle        = None
        self.lof_bundle       = None
        self.trans_config     = None
        self.trans_model      = None
        self.scaler           = None
        self.scale_cols       = None
        self.n_features       = None
        self.platt            = None
        self.ensemble_weights = None
        self.model_report     = None
        self.llm_summaries    = {}
        self.scores_df        = None
        self.scores_lookup    = {}
        self.loaded           = False
        self.load_errors      = []

    def load(self):
        logger.info("Loading model artifacts...")

        for name, path, attr in [
            ("scaler",     f"{MODEL_DIR}/scaler.pkl",                  "scaler_bundle"),
            ("if",         f"{MODEL_DIR}/isolation_forest.pkl",        "if_bundle"),
            ("lof",        f"{MODEL_DIR}/lof.pkl",                     "lof_bundle"),
            ("platt",      f"{MODEL_DIR}/platt_scaler.pkl",            "platt"),
            ("ensemble_w", f"{MODEL_DIR}/ensemble_weights.json",       "ensemble_weights"),
            ("report",     f"{MODEL_DIR}/model_report.json",           "model_report"),
            ("llm_sum",    f"{MODEL_DIR}/llm_risk_summaries.json",     "llm_summaries"),
        ]:
            try:
                with open(path, "rb" if path.endswith(".pkl") else "r") as f:
                    obj = pickle.load(f) if path.endswith(".pkl") else json.load(f)
                setattr(self, attr, obj)
                logger.info(f"  ✓ {name}")
            except Exception as e:
                self.load_errors.append(f"{name}: {e}")
                logger.warning(f"  ✗ {name}: {e}")

        # Unpack scaler bundle
        if hasattr(self, "scaler_bundle") and self.scaler_bundle:
            self.scaler     = self.scaler_bundle.get("scaler")
            self.scale_cols = self.scaler_bundle.get("scale_cols", [])

        # Detect IF feature count
        if self.if_bundle and "model" in self.if_bundle:
            try:
                self.n_features = self.if_bundle["model"].n_features_in_
                logger.info(f"  IF expects {self.n_features} features")
            except Exception:
                self.n_features = len(self.scale_cols) if self.scale_cols else 95

        # Load scores lookup
        try:
            self.scores_df = pd.read_csv(f"{DATA_DIR}/scores.csv")
            self.scores_lookup = {
                str(row["user_id"]): row.to_dict()
                for _, row in self.scores_df.iterrows()
            }
            logger.info(f"  ✓ scores.csv ({len(self.scores_df)} rows)")
        except Exception as e:
            self.load_errors.append(f"scores.csv: {e}")

        self.loaded = True
        logger.info(f"Registry ready | LLM: {'Groq ✓' if HAS_GROQ else 'rule-based'} | "
                    f"Users: {len(self.scores_lookup)}")


registry = ModelRegistry()

# Alert tier thresholds — computed at startup from actual score distribution using IQR method.
# Tukey fences: CRITICAL = Q75+3×IQR, HIGH = Q75+1.5×IQR, MEDIUM = Q50+1×IQR.

THRESH_CRITICAL = 0.80   # fallback — overwritten at startup
THRESH_HIGH     = 0.60   # fallback
THRESH_MEDIUM   = 0.45   # fallback

def compute_thresholds_from_scores(scores_csv_path: str) -> dict:
    """
    Compute alert thresholds using z-score + IQR (Tukey fence) method.
    More statistically principled than fixed percentiles:
      - CRITICAL = Q75 + 3.0*IQR  (extreme outlier, Tukey outer fence)
      - HIGH     = Q75 + 1.5*IQR  (mild outlier, Tukey inner fence)
      - MEDIUM   = Q50 + 1.0*IQR  (above median by one spread unit)
    Floors ensure thresholds are never trivially low.
    """
    import os
    try:
        df = pd.read_csv(scores_csv_path)
        col = "calibrated_score" if "calibrated_score" in df.columns else "ensemble_score"
        scores = df[col].dropna().values.astype(float)
        if len(scores) < 10:
            raise ValueError("Too few scores")

        q25  = float(np.percentile(scores, 25))
        q50  = float(np.percentile(scores, 50))
        q75  = float(np.percentile(scores, 75))
        iqr  = q75 - q25

        # Tukey fences
        t_critical = q75 + 3.0 * iqr   # extreme outlier
        t_high     = q75 + 1.5 * iqr   # mild outlier
        t_medium   = q50 + 1.0 * iqr   # above median + 1 IQR

        # Clamp to [0, 1] and apply sanity floors
        t_critical = float(np.clip(t_critical, 0.70, 1.0))
        t_high     = float(np.clip(t_high,     0.50, t_critical - 0.05))
        t_medium   = float(np.clip(t_medium,   0.30, t_high     - 0.05))

        # Also compute z-scores to cross-validate
        mu  = float(scores.mean())
        std = float(scores.std()) or 1e-6
        z_critical = (t_critical - mu) / std
        z_high     = (t_high     - mu) / std

        thresholds = {
            "CRITICAL":     round(t_critical, 4),
            "HIGH":         round(t_high,     4),
            "MEDIUM":       round(t_medium,   4),
            "method":       "IQR_tukey_fence",
            "basis":        col,
            "q25":          round(q25,  4),
            "q50":          round(q50,  4),
            "q75":          round(q75,  4),
            "iqr":          round(iqr,  4),
            "z_critical":   round(z_critical, 2),
            "z_high":       round(z_high,     2),
            "score_mean":   round(mu,   4),
            "score_std":    round(std,  4),
            "n_scores":     int(len(scores)),
            "n_critical":   int((scores >= t_critical).sum()),
            "n_high":       int(((scores >= t_high) & (scores < t_critical)).sum()),
            "n_medium":     int(((scores >= t_medium) & (scores < t_high)).sum()),
            "n_low":        int((scores < t_medium).sum()),
        }

        os.makedirs("models", exist_ok=True)
        with open(os.path.join("models", "alert_thresholds.json"), "w") as f:
            json.dump(thresholds, f, indent=2)

        logger.info(
            f"Thresholds (IQR method) — "
            f"CRITICAL≥{t_critical:.3f} (z={z_critical:.1f}σ, n={thresholds['n_critical']}), "
            f"HIGH≥{t_high:.3f} (z={z_high:.1f}σ, n={thresholds['n_high']}), "
            f"MEDIUM≥{t_medium:.3f} (n={thresholds['n_medium']})"
        )
        return thresholds

    except Exception as e:
        logger.warning(f"Threshold computation failed: {e} — using fallbacks")
        return {"CRITICAL": 0.80, "HIGH": 0.60, "MEDIUM": 0.45,
                "method": "fallback", "n_scores": 0}

def tier_from_score(score: float) -> str:
    """Assign alert tier using data-derived thresholds."""
    if score >= THRESH_CRITICAL: return "CRITICAL"
    if score >= THRESH_HIGH:     return "HIGH"
    if score >= THRESH_MEDIUM:   return "MEDIUM"
    return "LOW"


# ══════════════════════════════════════════════════════════════════════════════
# WEBSOCKET CONNECTION MANAGER  (RT-05)
# ══════════════════════════════════════════════════════════════════════════════

class ConnectionManager:
    """Manages all active WebSocket connections and broadcasts to them."""

    def __init__(self):
        self.alert_connections: Set[WebSocket]  = set()
        self.event_connections: Set[WebSocket]  = set()

    async def connect_alerts(self, ws: WebSocket):
        await ws.accept()
        self.alert_connections.add(ws)
        logger.info(f"WS alert client connected. Total: {len(self.alert_connections)}")

    async def connect_events(self, ws: WebSocket):
        await ws.accept()
        self.event_connections.add(ws)

    def disconnect(self, ws: WebSocket):
        self.alert_connections.discard(ws)
        self.event_connections.discard(ws)

    async def broadcast_alert(self, payload: dict):
        """Push a scored alert to all connected dashboard clients immediately."""
        if not self.alert_connections:
            return
        msg = json.dumps(payload, default=str)
        dead = set()
        for ws in list(self.alert_connections):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.alert_connections -= dead

    async def broadcast_event(self, payload: dict):
        if not self.event_connections:
            return
        msg = json.dumps(payload, default=str)
        dead = set()
        for ws in list(self.event_connections):
            try:
                await ws.send_text(msg)
            except Exception:
                dead.add(ws)
        self.event_connections -= dead


ws_manager = ConnectionManager()


# ══════════════════════════════════════════════════════════════════════════════
# REAL-TIME STREAMING PIPELINE  (RT-01 through RT-04)
# ══════════════════════════════════════════════════════════════════════════════

class StreamingPipeline:
    """
    Per-user event buffer with incremental feature tracking.
    On each relevant event:
      1. Update running feature snapshot (O(1) per field)
      2. Check instant-trigger conditions
      3. If triggered or buffer >= STREAM_WINDOW → rescore
      4. Push result to WebSocket manager
    """

    def __init__(self):
        self.user_buffers:   Dict[str, deque]  = defaultdict(lambda: deque(maxlen=500))
        self.user_features:  Dict[str, dict]   = {}   # running feature snapshots
        self.user_scores:    Dict[str, dict]   = {}
        self.user_last_ip:   Dict[str, tuple]  = {}   # (ip, continent, timestamp)
        self.event_count:    int               = 0
        self.alert_queue:    deque             = deque(maxlen=1000)
        self.last_processed: Optional[str]     = None
        self._loop:          Optional[asyncio.AbstractEventLoop] = None
        # FIX: track when each user last generated an alert to prevent spam
        self.user_last_alerted: Dict[str, datetime] = {}
        # FIX: track which known high-risk users have had their initial alert fired
        self.initial_alert_fired: set = set()
        # Performance: track event count at last score per user to avoid re-running
        # the full ML pipeline on every window tick
        self.user_events_at_last_score: Dict[str, int] = {}

    def _get_loop(self):
        # Use the global main loop set at startup
        global _main_loop
        return _main_loop

    # ── Public entry point ─────────────────────────────────────────────────────

    def ingest(self, event: EventPayload) -> dict:
        uid = event.user_id
        ts_str = event.timestamp
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        except Exception:
            ts = datetime.utcnow()

        raw_ev = {
            "event_type": event.event_type,
            "timestamp":  ts,
            "ip_address": event.ip_address,
            "device_id":  event.device_id,
            "amount":     event.amount,
            "volume":     event.volume,
            "pnl":        event.pnl,
            "instrument": event.instrument,
        }
        self.user_buffers[uid].append(raw_ev)
        self.event_count   += 1
        self.last_processed = datetime.utcnow().isoformat()

        #update running feature snapshot incrementally
        self._update_features(uid, raw_ev, ts)

        #check instant-trigger conditions
        instant = self._check_instant_triggers(uid, raw_ev)
        triggered = instant

        # For known high-risk users, trigger ONCE on their first event in this session.
        # FIX: was triggering on EVERY event for CRITICAL/HIGH users (uid not in user_scores
        # resets every restart but uid gets added after first score, yet the old code
        # used uid not in self.user_scores which is correct — the real bug was that
        # window-based scoring at STREAM_WINDOW=10 kept re-triggering alerts).
        if not triggered and uid in registry.scores_lookup:
            row = registry.scores_lookup[uid]
            pre_tier = str(row.get("alert_tier", "LOW"))
            if pre_tier in ("CRITICAL", "HIGH") and uid not in self.initial_alert_fired:
                triggered = True
                instant   = True

        # Window-based scoring when buffer fills (but only rescore if enough new events)
        buf_size = len(self.user_buffers[uid])
        if not triggered and buf_size >= STREAM_WINDOW:
            last_scored_at = self.user_events_at_last_score.get(uid, 0)
            events_since   = self.event_count - last_scored_at
            if events_since >= RESCORE_MIN_NEW_EVENTS:
                triggered = True

        score_result = None
        if triggered:
            self.user_events_at_last_score[uid] = self.event_count  # record when we scored
            score_result = self._score_user(uid)
            if score_result:
                self.user_scores[uid] = score_result

                # Mark initial alert as fired for this user
                if uid not in self.initial_alert_fired and uid in registry.scores_lookup:
                    self.initial_alert_fired.add(uid)

                # FIX: persist LLM summary back to registry so it isn't regenerated
                # on every subsequent rescore (expensive + causes duplicate Groq calls)
                if score_result.get("llm_summary") and uid not in registry.llm_summaries:
                    registry.llm_summaries[uid] = score_result["llm_summary"]

                # FIX: cooldown check — only push alert if this user hasn't alerted
                # recently (prevents the same flagged user flooding the dashboard
                # as every one of their events passes through the stream)
                now = datetime.utcnow()
                last_alerted = self.user_last_alerted.get(uid)
                cooldown_ok  = (
                    last_alerted is None or
                    (now - last_alerted).total_seconds() > ALERT_COOLDOWN_MINUTES * 60
                )

                if score_result.get("alert_tier") in ("CRITICAL", "HIGH") and cooldown_ok:
                    self.user_last_alerted[uid] = now
                    alert_entry = {
                        "user_id":   uid,
                        "tier":      score_result["alert_tier"],
                        "score":     score_result["ensemble_score"],
                        "source":    score_result.get("source", "live"),
                        "trigger":   "instant" if instant else "window",
                        "timestamp": self.last_processed,
                        "top_feature": score_result.get("top_feature", ""),
                        "llm_summary": score_result.get("llm_summary", ""),
                    }
                    self.alert_queue.append(alert_entry)
                    #push to WebSocket (non-blocking, thread-safe)
                    loop = self._get_loop()
                    if loop and loop.is_running():
                        asyncio.run_coroutine_threadsafe(
                            ws_manager.broadcast_alert(alert_entry), loop
                        )

        #push raw event to event WS channel (non-blocking)
        loop = self._get_loop()
        if loop and loop.is_running():
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast_event({
                    "type": "event", "user_id": uid,
                    "event_type": event.event_type,
                    "timestamp": ts_str,
                    "buffer_size": len(self.user_buffers[uid]),
                }), loop
            )

        # For known CRITICAL/HIGH users, always show their pre-computed tier
        # even if not yet scored in this session
        display_tier = self.user_scores.get(uid, {}).get("alert_tier", "UNKNOWN")
        if display_tier == "UNKNOWN" and uid in registry.scores_lookup:
            display_tier = str(registry.scores_lookup[uid].get("alert_tier","UNKNOWN"))

        return {
            "user_id":         uid,
            "buffer_size":     len(self.user_buffers[uid]),
            "total_events":    self.event_count,
            "re_scored":       triggered,
            "instant_trigger": instant,
            "current_tier":    display_tier,
            "current_score":   self.user_scores.get(uid, {}).get("ensemble_score", 0.0),
        }

    # ── Incremental feature updates (RT-02) ────────────────────────────────────

    def _update_features(self, uid: str, ev: dict, ts: datetime):
        f = self.user_features.setdefault(uid, {
            "login_count": 0, "login_failed_count": 0, "deposit_count": 0,
            "withdrawal_count": 0, "total_deposits": 0.0, "total_withdrawals": 0.0,
            "trade_count": 0, "total_volume": 0.0, "max_volume": 0.0,
            "total_pnl": 0.0, "unique_ips": set(), "unique_devices": set(),
            "volume_history": deque(maxlen=100), "pnl_history": deque(maxlen=100),
            "login_ts_history": deque(maxlen=20),
            "off_hours_logins": 0, "small_deposits": 0,
            "failed_burst_window": deque(maxlen=20),  # last 20 failed login timestamps
            "geo_impossible_count": 0,
        })

        etype = ev["event_type"]
        ip    = ev.get("ip_address")
        ts_h  = ts.hour

        if etype == "login":
            f["login_count"] += 1
            f["login_ts_history"].append(ts)
            if ts_h in range(0, 5):
                f["off_hours_logins"] += 1
            if ip:
                f["unique_ips"].add(ip)
                # Geo-impossible detection
                cont = ip_to_continent(ip)
                last = self.user_last_ip.get(uid)
                if last:
                    last_ip, last_cont, last_ts = last
                    hours_diff = abs((ts - last_ts).total_seconds()) / 3600
                    if cont != last_cont and 0 < hours_diff < GEO_IMPOSSIBLE_HOURS:
                        f["geo_impossible_count"] += 1
                self.user_last_ip[uid] = (ip, cont, ts)

        elif etype == "login_failed":
            f["login_failed_count"] += 1
            f["failed_burst_window"].append(ts)

        elif etype == "deposit":
            amt = ev.get("amount") or 0.0
            f["deposit_count"] += 1
            f["total_deposits"] += amt
            if amt < 200:  # structuring signal
                f["small_deposits"] += 1

        elif etype == "withdrawal":
            amt = ev.get("amount") or 0.0
            f["withdrawal_count"] += 1
            f["total_withdrawals"] += amt

        elif etype == "trade":
            vol = ev.get("volume") or 0.0
            pnl = ev.get("pnl") or 0.0
            f["trade_count"] += 1
            f["total_volume"] += vol
            f["max_volume"] = max(f["max_volume"], vol)
            f["volume_history"].append(vol)
            f["pnl_history"].append(pnl)
            f["total_pnl"] += pnl

        if ev.get("device_id"):
            f["unique_devices"].add(ev["device_id"])

    # ── Instant trigger conditions (RT-03) ──────────────────────────────────────

    def _check_instant_triggers(self, uid: str, ev: dict) -> bool:
        f = self.user_features.get(uid, {})
        etype = ev["event_type"]

        # 3 consecutive failed logins
        if etype == "login_failed" and f.get("login_failed_count", 0) >= 3:
            burst = list(f.get("failed_burst_window", []))
            if len(burst) >= 3:
                window_s = (burst[-1] - burst[-3]).total_seconds()
                if window_s < 600:  # 3 failures within 10 min
                    return True

        # Volume spike > 5× user mean
        if etype == "trade":
            vol_history = list(f.get("volume_history", []))
            if len(vol_history) >= 5:
                mean_vol = np.mean(vol_history[:-1])
                cur_vol  = ev.get("volume") or 0.0
                if mean_vol > 0 and cur_vol / mean_vol > 5.0:
                    return True

        # Large withdrawal
        if etype == "withdrawal":
            amt = ev.get("amount") or 0.0
            if amt >= 5000:
                return True

        # Geo-impossible login
        if etype == "login" and f.get("geo_impossible_count", 0) > 0:
            return True

        return False

    # ── Scoring engine (RT-04) ──────────────────────────────────────────────────

    def _score_user(self, uid: str) -> Optional[dict]:
        """
        For known users: augment precomputed score with live feature deltas.
        For unknown users: full live inference from buffer features.
        """
        f = self.user_features.get(uid, {})

        # ── Feature vector construction ────────────────────────────────────────
        if registry.scale_cols is None:
            return None

        n_base  = len(registry.scale_cols)
        n_feats = registry.n_features or n_base
        feat_vec = np.zeros((1, n_base))
        col_map  = {c: i for i, c in enumerate(registry.scale_cols)}

        def sf(name, val):
            if name in col_map:
                feat_vec[0, col_map[name]] = float(val)

        # Populate from running feature snapshot
        n_logins = max(f.get("login_count", 0) + f.get("login_failed_count", 0), 1)
        sf("login_count",           f.get("login_count", 0))
        sf("login_failed_count",    f.get("login_failed_count", 0))
        sf("login_failure_rate",    f.get("login_failed_count", 0) / n_logins)
        sf("unique_ip_count",       len(f.get("unique_ips", set())))
        sf("unique_device_count",   len(f.get("unique_devices", set())))
        sf("offhours_login_count",  f.get("off_hours_logins", 0))
        sf("offhours_login_ratio",  f.get("off_hours_logins", 0) / max(f.get("login_count", 1), 1))
        sf("brute_force_max_burst", f.get("login_failed_count", 0))
        sf("geo_impossible_login_count", f.get("geo_impossible_count", 0))
        sf("deposit_count",         f.get("deposit_count", 0))
        sf("withdrawal_count",      f.get("withdrawal_count", 0))
        sf("total_deposits",        f.get("total_deposits", 0.0))
        sf("total_withdrawals",     f.get("total_withdrawals", 0.0))
        sf("small_deposit_count",   f.get("small_deposits", 0))
        sf("small_deposit_ratio",   f.get("small_deposits", 0) / max(f.get("deposit_count", 1), 1))
        sf("trade_count",           f.get("trade_count", 0))
        sf("mean_volume",           f.get("total_volume", 0) / max(f.get("trade_count", 1), 1))
        sf("max_volume",            f.get("max_volume", 0.0))
        sf("total_pnl",             f.get("total_pnl", 0.0))

        vol_hist = list(f.get("volume_history", []))
        if len(vol_hist) >= 2:
            mean_v = np.mean(vol_hist)
            sf("std_volume",        float(np.std(vol_hist)))
            sf("volume_spike_ratio", f.get("max_volume", 0) / max(mean_v, 1.0))
            sf("pnl_std",           float(np.std(list(f.get("pnl_history", [])))) if len(f.get("pnl_history", [])) > 1 else 0.0)
            mean_pnl = np.mean(list(f.get("pnl_history", [])))
            pnl_std  = float(np.std(list(f.get("pnl_history", [])))) or 1e-6
            sf("pnl_sharpe",        mean_pnl / pnl_std)

        dep_with_ratio = f.get("total_deposits", 0) / max(f.get("total_withdrawals", 1e-6), 1e-6)
        sf("deposit_withdrawal_ratio", dep_with_ratio)

        # Scale and pad
        try:
            scaled = registry.scaler.transform(feat_vec)
        except Exception:
            scaled = feat_vec
        if n_feats > n_base:
            scaled = np.hstack([scaled, np.zeros((1, n_feats - n_base))])

        # Run IF
        if_score = 0.0
        if registry.if_bundle:
            try:
                raw = registry.if_bundle["model"].decision_function(scaled)[0]
                if_score = float(np.clip(-raw / 0.5, 0, 1))
            except Exception:
                pass

        # Run LOF
        lof_score = 0.0
        if registry.lof_bundle:
            try:
                n_lof = registry.lof_bundle["model"].n_features_in_
                lof_in = scaled[:, :n_lof] if scaled.shape[1] >= n_lof \
                         else np.hstack([scaled, np.zeros((1, n_lof - scaled.shape[1]))])
                raw_l = registry.lof_bundle["model"].decision_function(lof_in)[0]
                lof_score = float(np.clip(-raw_l / 0.5, 0, 1))
            except Exception:
                pass

        # Live ensemble from buffer events (will be low early on)
        live_ensemble = (if_score + lof_score) / 2.0
        source        = "live_inference"
        top_feature   = ""
        llm_cached    = None

        if uid in registry.scores_lookup:
            row         = registry.scores_lookup[uid]
            # Use pre-computed batch score as the authoritative base
            # (computed on full 90-day feature history — far more reliable
            #  than a feature vector from 1-30 live events)
            pre_score   = float(row.get("ensemble_score", 0.0))
            pre_tier    = str(row.get("alert_tier", "LOW"))
            top_feature = str(row.get("shap_top1_feature", ""))
            llm_cached  = registry.llm_summaries.get(str(uid))

            # Boost ensemble if live signal is suspicious
            # Live score can only raise the score, never lower it
            # (a 1-event buffer shouldn't clear a historically risky user)
            live_boost  = max(0.0, live_ensemble - 0.1)  # only count if meaningfully high
            ensemble    = min(1.0, pre_score + 0.3 * live_boost)
            source      = "blended"

            # Use pre-computed tier directly — it was computed on full data
            # Only upgrade tier if live signal is very strong (instant trigger)
            tier = pre_tier
            if live_ensemble > 0.70 and pre_tier in ("LOW","MEDIUM"):
                tier = "HIGH"   # live signal strong enough to upgrade
        else:
            # Unknown user: use live score only
            ensemble = live_ensemble
            tier = tier_from_score(ensemble)

        calibrated = ensemble  # already calibrated via pre-computed score

        # Top feature from live data (highest absolute scaled value)
        x_flat   = scaled[0]
        top_idx  = int(np.argmax(np.abs(x_flat)))
        live_top = registry.scale_cols[top_idx] if top_idx < len(registry.scale_cols) else "unknown"

        # LLM summary for CRITICAL/HIGH — use cached first, generate only if missing
        llm_summary = llm_cached  # from llm_risk_summaries.json (training time)
        if llm_summary is None and uid in registry.llm_summaries:
            llm_summary = registry.llm_summaries[uid]  # from live session cache
        if tier in ("CRITICAL", "HIGH") and llm_summary is None:
            llm_summary = _generate_llm_summary(uid, {
                "alert_tier": tier, "ensemble_score": calibrated,
                "if_score": if_score, "lof_score": lof_score,
                "shap_top1_feature": live_top,
                "shap_top1_value":   float(x_flat[top_idx]) if top_idx < len(x_flat) else 0.0,
                "shap_top2_feature": "",
                "shap_top2_value":   0.0,
            })
            # Cache so we don't call Groq again for this user this session
            if llm_summary:
                registry.llm_summaries[uid] = llm_summary

        return {
            "user_id":       uid,
            "if_score":      round(if_score, 4),
            "lof_score":     round(lof_score, 4),
            "live_ensemble": round(live_ensemble, 4),
            "ensemble_score":round(ensemble, 4),
            "calibrated_score": round(calibrated, 4),
            "alert_tier":    tier,
            "source":        source,
            "top_feature":   top_feature or live_top,
            "llm_summary":   llm_summary,
            "scored_at":     datetime.utcnow().isoformat(),
        }

    def get_status(self) -> dict:
        return {
            "active":              True,
            "total_events":        self.event_count,
            "active_users":        len(self.user_buffers),
            "scored_users":        len(self.user_scores),
            "alert_queue_size":    len(self.alert_queue),
            "ws_alert_clients":    len(ws_manager.alert_connections),
            "ws_event_clients":    len(ws_manager.event_connections),
            "last_processed":      self.last_processed,
            "recent_alerts":       list(self.alert_queue)[-10:],
        }


stream = StreamingPipeline()

# Global event loop reference — set at FastAPI startup
# This lets the synchronous ingest() schedule async WS broadcasts
_main_loop: Optional[asyncio.AbstractEventLoop] = None


# ══════════════════════════════════════════════════════════════════════════════
# LLM HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _generate_llm_summary(user_id: str, row: dict) -> str:
    def rule_based():
        tier  = row.get("alert_tier", "UNKNOWN")
        score = float(row.get("ensemble_score", row.get("calibrated_score", 0)))
        feats = [row.get(f"shap_top{i}_feature", "") for i in range(1, 4)]
        sigs  = ", ".join(f.replace("_", " ") for f in feats if f and f != "nan")
        actions = {
            "CRITICAL": "Immediate account suspension and compliance review recommended.",
            "HIGH":     "Escalate to senior compliance officer within 24 hours.",
            "MEDIUM":   "Add to enhanced monitoring watchlist.",
            "LOW":      "Continue passive monitoring.",
        }
        return (f"User {user_id} flagged at {tier} risk (score: {score:.3f}). "
                f"Primary signals: {sigs or 'statistical deviation'}. "
                f"{actions.get(tier, 'Monitor.')}")

    if not HAS_GROQ or groq_client is None:
        return rule_based()

    prompt = f"""You are a compliance analyst at a forex brokerage.
Write a concise 3-sentence risk alert for this flagged user.

User: {user_id} | Tier: {row.get('alert_tier')} | Score: {row.get('ensemble_score', row.get('calibrated_score',0)):.3f}
Top signals: {row.get('shap_top1_feature','?')} ({row.get('shap_top1_value',0):+.3f}), {row.get('shap_top2_feature','?')} ({row.get('shap_top2_value',0):+.3f})
Write 3 sentences: what the signals mean behaviourally, the risk level, and recommended compliance action."""

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are a forex compliance analyst. Be concise and professional."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=200, temperature=0.3,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq error for {user_id}: {e}")
        return rule_based()


def safe(row, key, default=None):
    val = row.get(key, default)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return default
    return val

def norm_scores(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# SCORING ENDPOINTS HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def score_user_by_id(user_id: str, generate_llm: bool = False) -> ScoreResponse:
    if not registry.scores_lookup:
        raise HTTPException(status_code=503, detail="scores.csv not loaded")
    row = registry.scores_lookup.get(str(user_id))
    if row is None:
        raise HTTPException(status_code=404, detail=f"User {user_id} not found")

    llm_summary = registry.llm_summaries.get(str(user_id))
    if generate_llm and llm_summary is None:
        llm_summary = _generate_llm_summary(user_id, row)

    # Merge with live score if available
    live = stream.user_scores.get(user_id, {})
    source = "precomputed"
    if live:
        source = live.get("source", "live")

    return ScoreResponse(
        user_id              = user_id,
        if_score             = float(live.get("if_score", safe(row, "if_score", 0.0))),
        lof_score            = float(live.get("lof_score", safe(row, "lof_score", 0.0))),
        trans_score          = float(safe(row, "trans_score", safe(row, "transformer_score", 0.0))),
        ensemble_score       = float(live.get("ensemble_score", safe(row, "ensemble_score", 0.0))),
        calibrated_score     = float(live.get("calibrated_score", safe(row, "calibrated_score", safe(row, "ensemble_score", 0.0)))),
        alert_tier           = str(live.get("alert_tier", safe(row, "alert_tier", "UNKNOWN"))),
        ensemble_flagged     = int(safe(row, "ensemble_flagged", 0)),
        window_7d_score      = float(safe(row, "window_7d_anomaly_score", safe(row, "rolling_7d_volume_zscore", 0.0))),
        recent_spike_flag    = int(safe(row, "recent_spike_flag", 0)),
        shap_top1_feature    = safe(row, "shap_top1_feature"),
        shap_top1_value      = safe(row, "shap_top1_value"),
        shap_top2_feature    = safe(row, "shap_top2_feature"),
        shap_top2_value      = safe(row, "shap_top2_value"),
        shap_top3_feature    = safe(row, "shap_top3_feature"),
        shap_top3_value      = safe(row, "shap_top3_value"),
        trans_top1_feature   = safe(row, "trans_top1_feature"),
        trans_top1_err       = safe(row, "trans_top1_err"),
        llm_risk_summary     = llm_summary,
        composite_risk_score = safe(row, "composite_risk_score"),
        financial_risk_score = safe(row, "financial_risk_score"),
        trading_risk_score   = safe(row, "trading_risk_score"),
        network_risk_score   = safe(row, "network_risk_score"),
        scored_at            = datetime.utcnow().isoformat(),
        score_source         = source,
    )


def score_raw_features(payload: RawFeaturePayload) -> PredictResponse:
    if registry.scaler is None or registry.if_bundle is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    uid     = payload.user_id or f"LIVE_{datetime.utcnow().strftime('%H%M%S')}"
    n_base  = len(registry.scale_cols)
    n_feats = registry.n_features or n_base
    feat_vec = np.zeros((1, n_base))
    col_map  = {c: i for i, c in enumerate(registry.scale_cols)}
    matched  = 0

    for fname, fval in payload.features.items():
        if fname in col_map:
            feat_vec[0, col_map[fname]] = float(fval)
            matched += 1

    if matched == 0:
        raise HTTPException(status_code=422,
            detail=f"No features matched. Known (first 10): {registry.scale_cols[:10]}")

    scaled = registry.scaler.transform(feat_vec)
    if n_feats > n_base:
        scaled = np.hstack([scaled, np.zeros((1, n_feats - n_base))])

    try:
        raw      = registry.if_bundle["model"].decision_function(scaled)[0]
        if_score = float(np.clip(-raw / 0.5, 0, 1))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IF error: {e}")

    lof_score = 0.0
    if registry.lof_bundle:
        try:
            n_lof = registry.lof_bundle["model"].n_features_in_
            lof_in = scaled[:, :n_lof] if scaled.shape[1] >= n_lof \
                     else np.hstack([scaled, np.zeros((1, n_lof - scaled.shape[1]))])
            raw_l = registry.lof_bundle["model"].decision_function(lof_in)[0]
            lof_score = float(np.clip(-raw_l / 0.5, 0, 1))
        except Exception:
            pass

    ensemble = (if_score + lof_score) / 2.0
    tier = tier_from_score(ensemble)

    x_flat   = scaled[0]
    top_idx  = np.argsort(np.abs(x_flat))[::-1][:5]
    top_feats = [
        {"feature": registry.scale_cols[i] if i < n_base else f"derived_{i}",
         "scaled_value": round(float(x_flat[i]), 4),
         "raw_value":    round(float(feat_vec[0, i] if i < n_base else 0), 4)}
        for i in top_idx
    ]

    llm_summary = None
    if tier in ("CRITICAL", "HIGH"):
        llm_summary = _generate_llm_summary(uid, {
            "alert_tier": tier, "ensemble_score": ensemble,
            "if_score": if_score, "lof_score": lof_score,
            "shap_top1_feature": top_feats[0]["feature"] if top_feats else "",
            "shap_top1_value":   top_feats[0]["scaled_value"] if top_feats else 0,
            "shap_top2_feature": top_feats[1]["feature"] if len(top_feats) > 1 else "",
            "shap_top2_value":   top_feats[1]["scaled_value"] if len(top_feats) > 1 else 0,
        })

    return PredictResponse(
        user_id=uid, if_score=round(if_score, 4), lof_score=round(lof_score, 4),
        ensemble_score=round(ensemble, 4), alert_tier=tier,
        flagged=ensemble >= ALERT_THRESHOLD, top_features=top_feats,
        llm_summary=llm_summary, scored_at=datetime.utcnow().isoformat(),
    )


# ══════════════════════════════════════════════════════════════════════════════
# FASTAPI APP
# ══════════════════════════════════════════════════════════════════════════════

app = FastAPI(
    title       = "ForexGuard — Real-Time Anomaly Detection API",
    description = (
        "Real-time forex trader anomaly detection.\n\n"
        "**WebSocket Channels:**\n"
        "- `ws://host/ws/alerts` — push alerts to dashboard instantly\n"
        "- `ws://host/ws/events` — raw event echo for monitoring\n\n"
        "**Models:** IF + LOF + Staged Transformer Autoencoder (Bao et al. 2025)\n\n"
        "**LLM:** Groq `llama-3.1-8b-instant` (free tier)"
    ),
    version     = "4.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve dashboard.html as static
from fastapi.responses import FileResponse

@app.get("/dashboard", tags=["Dashboard"])
async def dashboard():
    return FileResponse("dashboard.html")

@app.on_event("startup")
async def startup():
    global _main_loop, THRESH_CRITICAL, THRESH_HIGH, THRESH_MEDIUM
    _main_loop = asyncio.get_event_loop()
    registry.load()

    # Compute data-driven thresholds from actual scores.csv
    import os
    scores_path = os.path.join(DATA_DIR, "scores.csv")
    if os.path.exists(scores_path):
        thresholds = compute_thresholds_from_scores(scores_path)
        THRESH_CRITICAL = thresholds.get("CRITICAL", THRESH_CRITICAL)
        THRESH_HIGH     = thresholds.get("HIGH",     THRESH_HIGH)
        THRESH_MEDIUM   = thresholds.get("MEDIUM",   THRESH_MEDIUM)
        logger.info(
            f"Thresholds (data-derived from {thresholds.get('n_scores',0)} users): "
            f"CRITICAL>={THRESH_CRITICAL:.4f}  "
            f"HIGH>={THRESH_HIGH:.4f}  "
            f"MEDIUM>={THRESH_MEDIUM:.4f}  "
            f"[{thresholds.get('n_critical',0)} CRIT, "
            f"{thresholds.get('n_high',0)} HIGH, "
            f"{thresholds.get('n_medium',0)} MED, "
            f"{thresholds.get('n_low',0)} LOW]"
        )
    else:
        logger.warning(f"scores.csv not found at {scores_path} — using fallback thresholds")

    stream._loop = asyncio.get_event_loop()
    logger.info("ForexGuard API ready.")


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"])
async def root():
    return {
        "service":       "ForexGuard Real-Time Anomaly Detection API",
        "version":       "4.1.0",
        "status":        "ready" if registry.loaded else "loading",
        "llm_backend":   f"Groq ({GROQ_MODEL})" if HAS_GROQ else "rule-based",
        "users_loaded":  len(registry.scores_lookup),
        "ws_endpoints":  ["/ws/alerts", "/ws/events"],
        "docs":          "/docs",
        "dashboard":     "/dashboard",
    }

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.get("/alerts/thresholds", tags=["Alerts"])
async def get_thresholds():
    """Return data-derived alert thresholds so the dashboard can show them."""
    import os
    thresh_path = os.path.join(MODEL_DIR, "alert_thresholds.json")
    if os.path.exists(thresh_path):
        with open(thresh_path) as f:
            return json.load(f)
    return {
        "CRITICAL": THRESH_CRITICAL,
        "HIGH":     THRESH_HIGH,
        "MEDIUM":   THRESH_MEDIUM,
        "basis":    "runtime",
    }


# ── WebSocket endpoints (RT-01, RT-06, RT-07) ──────────────────────────────────

@app.websocket("/ws/alerts")
async def ws_alerts(websocket: WebSocket):
    """
    Real-time alert channel. Dashboard connects here once.
    Server pushes JSON alert objects the moment any user is scored CRITICAL or HIGH.

    Message format:
    {
        "user_id": "U0042",
        "tier": "CRITICAL",
        "score": 0.87,
        "trigger": "instant",
        "top_feature": "login_failure_rate",
        "llm_summary": "User flagged for...",
        "timestamp": "2024-03-15T14:32:05"
    }
    """
    await ws_manager.connect_alerts(websocket)
    try:
        # Send current alert backlog on connect
        backlog = list(stream.alert_queue)[-20:]
        if backlog:
            await websocket.send_text(json.dumps({"type": "backlog", "alerts": backlog}, default=str))

        # Keep alive — ping/pong
        while True:
            try:
                msg = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if msg == "ping":
                    await websocket.send_text("pong")
            except asyncio.TimeoutError:
                await websocket.send_text(json.dumps({"type": "heartbeat", "ts": datetime.utcnow().isoformat()}))
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
        logger.info(f"WS alert client disconnected. Remaining: {len(ws_manager.alert_connections)}")


@app.websocket("/ws/events")
async def ws_events(websocket: WebSocket):
    """Raw event echo channel for monitoring/debug."""
    await ws_manager.connect_events(websocket)
    try:
        while True:
            await asyncio.wait_for(websocket.receive_text(), timeout=60.0)
    except (WebSocketDisconnect, asyncio.TimeoutError):
        ws_manager.disconnect(websocket)


# ── Scoring ────────────────────────────────────────────────────────────────────

@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_user(
    user_id:      str  = Query(..., example="U0042"),
    generate_llm: bool = Query(False, description="Generate Groq LLM risk summary on demand"),
):
    """
    Return full risk profile for a known user.
    If the user has been seen in the live stream, returns the blended
    (historical + live) score. Otherwise returns the precomputed score.
    """
    return score_user_by_id(user_id, generate_llm=generate_llm)


@app.post("/predict", response_model=PredictResponse, tags=["Scoring"])
async def predict(payload: RawFeaturePayload):
    """
    Score a user from raw feature values. Used for live inference panel in dashboard.
    Returns top contributing features and optional Groq LLM narrative.
    """
    return score_raw_features(payload)


# ── Alerts ─────────────────────────────────────────────────────────────────────

@app.get("/alerts", tags=["Alerts"])
async def list_alerts(
    tier:     Optional[str] = Query(None, description="CRITICAL/HIGH/MEDIUM/LOW"),
    sort_by:  str           = Query("ensemble_score"),
    page:     int           = Query(1, ge=1),
    per_page: int           = Query(20, ge=1, le=100),
):
    if registry.scores_df is None:
        raise HTTPException(status_code=503, detail="Scores not loaded")

    df = registry.scores_df.copy()

    # Merge live scores on top
    if stream.user_scores:
        for uid, live in stream.user_scores.items():
            mask = df["user_id"].astype(str) == str(uid)
            if mask.any():
                df.loc[mask, "ensemble_score"]   = live.get("ensemble_score", df.loc[mask, "ensemble_score"])
                df.loc[mask, "calibrated_score"] = live.get("calibrated_score", df.loc[mask, "calibrated_score"] if "calibrated_score" in df.columns else df.loc[mask, "ensemble_score"])
                df.loc[mask, "alert_tier"]       = live.get("alert_tier", df.loc[mask, "alert_tier"])

    df = df[df["alert_tier"] == tier.upper()] if tier else df[df["ensemble_flagged"] == 1]
    if sort_by in df.columns:
        df = df.sort_values(sort_by, ascending=False)

    total   = len(df)
    page_df = df.iloc[(page - 1) * per_page: page * per_page]

    results = []
    for _, row in page_df.iterrows():
        uid = str(row["user_id"])
        s1v = row.get("shap_top1_value", 0)
        results.append({
            "user_id":          uid,
            "alert_tier":       row.get("alert_tier", "?"),
            "ensemble_score":   round(float(row.get("ensemble_score", 0)), 4),
            "calibrated_score": round(float(row.get("calibrated_score", row.get("ensemble_score", 0))), 4),
            "if_score":         round(float(row.get("if_score", 0)), 4),
            "lof_score":        round(float(row.get("lof_score", 0)), 4),
            "shap_top1":        f"{row.get('shap_top1_feature','?')} ({float(s1v) if s1v and str(s1v)!='nan' else 0:+.3f})",
            "llm_summary":      registry.llm_summaries.get(uid, ""),
            "has_live_score":   uid in stream.user_scores,
        })

    return {"total": total, "page": page, "per_page": per_page,
            "pages": max(1, math.ceil(total / per_page)), "results": results}


@app.get("/alerts/summary/stats", response_model=AlertSummary, tags=["Alerts"])
async def alert_summary():
    if registry.scores_df is None:
        raise HTTPException(status_code=503, detail="Scores not loaded")

    df          = registry.scores_df
    tier_counts = df["alert_tier"].value_counts().to_dict()
    top_types   = []
    try:
        labels_df = pd.read_csv(f"{DATA_DIR}/labels_eval.csv")
        flagged   = set(df[df["ensemble_flagged"] == 1]["user_id"].astype(str))
        fl        = labels_df[labels_df["user_id"].astype(str).isin(flagged)]
        tc        = {}
        for t_str in fl["anomaly_types"].dropna():
            for t in str(t_str).split("|"):
                if t: tc[t] = tc.get(t, 0) + 1
        top_types = [{"type": k, "count": v}
                     for k, v in sorted(tc.items(), key=lambda x: -x[1])[:10]]
    except Exception:
        pass

    return AlertSummary(
        total_users      = len(df),
        flagged_users    = int((df["ensemble_flagged"] == 1).sum()),
        critical_count   = int(tier_counts.get("CRITICAL", 0)),
        high_count       = int(tier_counts.get("HIGH", 0)),
        medium_count     = int(tier_counts.get("MEDIUM", 0)),
        low_count        = int(tier_counts.get("LOW", 0)),
        recent_spikes    = len(stream.alert_queue),
        top_anomaly_types= top_types,
    )


@app.get("/alerts/{user_id}", tags=["Alerts"])
async def get_alert_detail(user_id: str, generate_llm: bool = Query(False)):
    return score_user_by_id(user_id, generate_llm=generate_llm)


# ── Streaming ──────────────────────────────────────────────────────────────────

@app.post("/stream/ingest", tags=["Streaming"])
async def ingest_event(event: EventPayload, background_tasks: BackgroundTasks):
    """
    Ingest a single real-time event. Scoring happens synchronously;
    WebSocket push happens in background.
    Response includes immediate score result.
    """
    result = stream.ingest(event)
    return {
        "status":    "accepted",
        "detail":    result,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/stream/ingest/batch", tags=["Streaming"])
async def ingest_batch(events: List[EventPayload]):
    """Ingest a list of events in one call."""
    results = []
    for ev in events:
        results.append(stream.ingest(ev))
    return {"accepted": len(results), "results": results}


@app.get("/stream/status", tags=["Streaming"])
async def stream_status():
    return stream.get_status()


@app.post("/stream/simulate", tags=["Streaming"])
async def simulate_stream(
    n_events: int = Query(100, ge=1, le=5000),
    n_users:  int = Query(10,  ge=1, le=100),
):
    import random
    event_types = ["login", "trade", "deposit", "withdrawal", "page_view", "login_failed"]
    instruments = ["EUR/USD", "GBP/USD", "USD/JPY", "XAU/USD"]
    for _ in range(n_events):
        uid   = f"SIM_U{random.randint(1, n_users):03d}"
        etype = random.choice(event_types)
        ev = EventPayload(
            user_id=uid, event_type=etype, timestamp=datetime.utcnow().isoformat(),
            ip_address=f"{random.choice([10,50,110,160,190,220])}.{random.randint(1,255)}.{random.randint(1,255)}.{random.randint(1,255)}",
            amount=round(random.uniform(100, 10000), 2) if etype in ("deposit", "withdrawal") else None,
            volume=round(random.uniform(10000, 1000000), 2) if etype == "trade" else None,
            pnl=round(random.uniform(-500, 1000), 2) if etype == "trade" else None,
            instrument=random.choice(instruments) if etype == "trade" else None,
        )
        stream.ingest(ev)
    return {"simulated_events": n_events, "pipeline_status": stream.get_status()}


# ── Models ─────────────────────────────────────────────────────────────────────

@app.get("/models/info", tags=["Models"])
async def model_info():
    return {
        "models_loaded": {
            "isolation_forest": registry.if_bundle is not None,
            "lof":              registry.lof_bundle is not None,
        },
        "llm_backend":      f"Groq ({GROQ_MODEL})" if HAS_GROQ else "rule-based",
        "n_features_if":    registry.n_features,
        "n_base_features":  len(registry.scale_cols) if registry.scale_cols else 0,
        "ensemble_weights": registry.ensemble_weights,
        "model_report":     registry.model_report,
        "load_errors":      registry.load_errors,
    }


@app.get("/features/list", tags=["Models"])
async def feature_list():
    if registry.scale_cols is None:
        raise HTTPException(status_code=503, detail="Scaler not loaded")
    return {"feature_count": len(registry.scale_cols), "features": registry.scale_cols}