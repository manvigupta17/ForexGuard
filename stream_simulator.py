"""
ForexGuard — Phase 4: Async Streaming Event Simulator
======================================================
Replays the full historical event log through the live API at 1000× real-time
speed, simulating a production Kafka consumer that continuously ingests events.

Architecture:
  - Loads portal_events.csv and trade_events.csv, merges and sorts by timestamp.
  - Sends events in concurrent batches via aiohttp with real-time pacing: each
    batch waits proportionally to the simulated time gap between events.
  - Tracks which users trigger new live alerts during replay (users not already
    flagged in pre-computed scores.csv).
  - Prints a progress report every 500 events and a full pipeline status at
    completion.

Usage:
  Start the API first, then run this script:
    uvicorn main:app --host 0.0.0.0 --port 8000
    python stream_simulator.py

Concurrency note:
  Each HTTP request is wrapped in a self-contained coroutine that opens and
  closes its aiohttp response internally. This avoids the CancelledError that
  occurs when passing context managers into asyncio.gather() directly.

  Connection pool is capped at 50 simultaneous connections to prevent OS socket
  exhaustion on Windows.
"""

import asyncio
import json
import warnings
from datetime import datetime

import aiohttp
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_URL         = "http://localhost:8000"
DATA_DIR         = "data"
SPEED_MULTIPLIER = 1000   # replay 1000× faster than wall-clock time
BATCH_SIZE       = 30     # concurrent requests per batch
PRINT_EVERY      = 500    # progress report interval (events)
STARTUP_WAIT_S   = 3      # seconds to wait for API startup before polling
REQUEST_TIMEOUT  = 15     # per-request timeout in seconds


# ---------------------------------------------------------------------------
# Core request coroutine
# ---------------------------------------------------------------------------

async def post_event(
    session: aiohttp.ClientSession,
    url: str,
    ev: dict,
) -> dict:
    """
    Send a single event to the ingest endpoint and return a parsed result dict.

    Each call is self-contained — it opens, reads, and closes the aiohttp
    response within the same coroutine, which is the correct pattern for use
    with asyncio.gather(). Returning plain dicts (not response objects) means
    callers need no context managers.

    Returns:
        dict with keys: ok (bool), tier (str), uid (str), or error (str).
    """
    try:
        async with session.post(
            url,
            json=ev,
            timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT),
        ) as resp:
            if resp.status == 200:
                data   = await resp.json()
                detail = data.get("detail", {})
                return {
                    "ok":   True,
                    "tier": detail.get("current_tier", ""),
                    "uid":  ev.get("user_id", ""),
                }
            return {"ok": False, "status": resp.status}
    except asyncio.CancelledError:
        return {"ok": False, "error": "cancelled"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:80]}


# ---------------------------------------------------------------------------
# API readiness check
# ---------------------------------------------------------------------------

async def wait_for_api(session: aiohttp.ClientSession) -> bool:
    """
    Poll the root endpoint until the API reports that scores are loaded.

    The API loads scores.csv on startup; users_loaded > 0 indicates readiness.
    Returns True once ready (or after 12 attempts with a warning).
    """
    print(f"\n  Waiting {STARTUP_WAIT_S}s for API startup...", end="", flush=True)
    await asyncio.sleep(STARTUP_WAIT_S)

    for _ in range(12):
        try:
            async with session.get(
                f"{BASE_URL}/",
                timeout=aiohttp.ClientTimeout(total=4),
            ) as resp:
                if resp.status == 200:
                    data  = await resp.json()
                    users = data.get("users_loaded", 0)
                    if users > 0:
                        print(
                            f" ready ({users} users loaded, "
                            f"LLM: {data.get('llm_backend', '?')})"
                        )
                        return True
                    print(".", end="", flush=True)
        except Exception:
            print(".", end="", flush=True)
        await asyncio.sleep(1)

    print(" [WARN] API may not be fully ready — proceeding anyway")
    return True


# ---------------------------------------------------------------------------
# Main simulation loop
# ---------------------------------------------------------------------------

async def run_simulator() -> None:
    """
    Load all historical events, merge into a unified time-ordered stream,
    and replay them through the API at SPEED_MULTIPLIER× real-time pace.
    """
    print("=" * 62)
    print("ForexGuard — Async Streaming Simulator")
    print("=" * 62)

    print("\nLoading event data...")
    portal = pd.read_csv(f"{DATA_DIR}/portal_events.csv")
    trades = pd.read_csv(f"{DATA_DIR}/trade_events.csv")
    portal["timestamp"] = pd.to_datetime(portal["timestamp"], format="mixed")
    trades["timestamp"] = pd.to_datetime(trades["timestamp"], format="mixed")

    # Load pre-computed scores to distinguish new live detections from known ones
    try:
        scores_df      = pd.read_csv(f"{DATA_DIR}/scores.csv")
        critical_users = set(
            scores_df[scores_df["alert_tier"] == "CRITICAL"]["user_id"].astype(str)
        )
        high_users = set(
            scores_df[scores_df["alert_tier"] == "HIGH"]["user_id"].astype(str)
        )
        print(
            f"  Pre-computed scores : {len(scores_df)} users  "
            f"({len(critical_users)} CRITICAL, {len(high_users)} HIGH)"
        )
    except Exception as exc:
        print(f"  [WARN] scores.csv: {exc}")
        critical_users, high_users = set(), set()

    # Merge portal and trade events into a unified time-ordered stream
    portal_cols = ["user_id", "timestamp", "event_type", "ip_address", "device_id", "amount"]
    trade_cols  = ["user_id", "timestamp", "instrument", "volume", "pnl"]

    pe = portal[portal_cols].copy()
    pe["volume"] = None
    pe["pnl"]    = None
    pe["instrument"] = None

    te = trades[trade_cols].copy()
    te["event_type"] = "trade"
    te["ip_address"] = None
    te["device_id"]  = None
    te["amount"]     = None

    all_cols   = ["user_id", "timestamp", "event_type", "ip_address",
                  "device_id", "amount", "volume", "pnl", "instrument"]
    all_events = (
        pd.concat([pe[all_cols], te[all_cols]], ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    total = len(all_events)

    print(f"  Total events        : {total:,}")
    print(f"  Speed multiplier    : {SPEED_MULTIPLIER}×")
    print(f"  Batch size          : {BATCH_SIZE} concurrent requests")
    est_s = total / (BATCH_SIZE * 10) + 10
    print(f"  Estimated duration  : ~{est_s:.0f}s\n")

    connector = aiohttp.TCPConnector(
        limit=50,
        limit_per_host=50,
        enable_cleanup_closed=True,
    )

    async with aiohttp.ClientSession(connector=connector) as session:

        # Verify API is reachable before starting replay
        try:
            async with session.get(
                f"{BASE_URL}/health",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    print(f"[ERROR] API returned {resp.status}")
                    return
            print(f"[OK] API reachable at {BASE_URL}")
        except Exception as exc:
            print(f"[ERROR] Cannot reach API: {exc}")
            print("  → Start the API first:")
            print("    uvicorn main:app --reload --host 0.0.0.0 --port 8000")
            return

        await wait_for_api(session)

        # Replay loop
        sent         = 0
        errors       = 0
        rt_alerts    = 0
        alerted_uids: set = set()
        seen_flagged: set = set()
        start_time   = asyncio.get_event_loop().time()
        prev_ts      = all_events["timestamp"].iloc[0]
        batch: list  = []

        for i, row in all_events.iterrows():
            uid = str(row["user_id"])

            if uid in critical_users or uid in high_users:
                seen_flagged.add(uid)

            event = {
                "user_id":    uid,
                "event_type": str(row["event_type"]),
                "timestamp":  str(row["timestamp"]),
                "ip_address": str(row["ip_address"]) if pd.notna(row.get("ip_address")) else None,
                "device_id":  str(row["device_id"])  if pd.notna(row.get("device_id"))  else None,
                "amount":     float(row["amount"])    if pd.notna(row.get("amount"))     else None,
                "volume":     float(row["volume"])    if pd.notna(row.get("volume"))     else None,
                "pnl":        float(row["pnl"])       if pd.notna(row.get("pnl"))        else None,
                "instrument": str(row["instrument"])  if pd.notna(row.get("instrument")) else None,
            }
            batch.append(event)

            if len(batch) >= BATCH_SIZE or i == total - 1:

                # Pace replay proportionally to simulated time gaps
                curr_ts  = row["timestamp"]
                time_gap = (curr_ts - prev_ts).total_seconds() / SPEED_MULTIPLIER
                if 0 < time_gap < 1.0:
                    await asyncio.sleep(time_gap)
                prev_ts = curr_ts

                tasks   = [
                    post_event(session, f"{BASE_URL}/stream/ingest", ev)
                    for ev in batch
                ]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for res in results:
                    if isinstance(res, Exception):
                        errors += 1
                    elif isinstance(res, dict):
                        if res.get("ok"):
                            sent += 1
                            uid_r  = res.get("uid", "")
                            tier_r = res.get("tier", "")
                            # Count only genuinely new live detections
                            if (
                                tier_r in ("CRITICAL", "HIGH")
                                and uid_r
                                and uid_r not in alerted_uids
                                and uid_r not in critical_users
                                and uid_r not in high_users
                            ):
                                rt_alerts += 1
                                alerted_uids.add(uid_r)
                        else:
                            errors += 1

                batch = []

                if sent > 0 and sent % PRINT_EVERY == 0:
                    elapsed  = asyncio.get_event_loop().time() - start_time
                    rate     = sent / max(elapsed, 1)
                    progress = sent / total * 100
                    print(
                        f"  [{progress:5.1f}%]  sent={sent:,}  errors={errors}  "
                        f"flagged_seen={len(seen_flagged)}  "
                        f"rt_alerts={rt_alerts}  "
                        f"rate={rate:.0f} ev/s  "
                        f"elapsed={elapsed:.0f}s"
                    )

    # Final summary
    elapsed_total = asyncio.get_event_loop().time() - start_time

    print(f"\n{'=' * 62}")
    print("Simulation complete")
    print(f"  Events sent            : {sent:,} / {total:,}")
    print(f"  Errors                 : {errors}")
    print(
        f"  Pre-flagged users seen : {len(seen_flagged)}"
        f"  ({len(critical_users)} CRITICAL + {len(high_users)} HIGH)"
    )
    print(
        f"  NEW live detections    : {rt_alerts}"
        f"  (users flagged by stream, not in pre-computed scores)"
    )
    print(f"  Total time             : {elapsed_total:.1f}s")
    print(f"  Throughput             : {sent / max(elapsed_total, 1):.0f} events/s")

    # Fetch and display final pipeline status from API
    try:
        async with aiohttp.ClientSession() as s:
            async with s.get(
                f"{BASE_URL}/stream/status",
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                status = await resp.json()
        print(f"\nFinal pipeline status:")
        print(json.dumps(
            {k: v for k, v in status.items() if k != "recent_alerts"},
            indent=2,
        ))
        if status.get("recent_alerts"):
            print(f"\nLast {len(status['recent_alerts'])} real-time alerts:")
            for alert in status["recent_alerts"][-5:]:
                print(
                    f"  {alert.get('tier', '?'):<10} {alert.get('user_id', '?')}  "
                    f"score={alert.get('score', 0):.3f}  "
                    f"trigger={alert.get('trigger', '?')}"
                )
    except Exception as exc:
        print(f"  [WARN] Could not fetch pipeline status: {exc}")

    print(f"\n{'=' * 62}")
    print(f"Dashboard  : http://localhost:8000/dashboard")
    print(f"API docs   : http://localhost:8000/docs")
    print(f"Alerts     : http://localhost:8000/alerts?tier=CRITICAL")
    print(f"WebSocket  : ws://localhost:8000/ws/alerts")


if __name__ == "__main__":
    try:
        asyncio.run(run_simulator())
    except KeyboardInterrupt:
        print("\n[INFO] Simulator stopped.")
