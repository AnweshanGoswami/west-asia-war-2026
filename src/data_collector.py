"""
src/data_collector.py
────────────────────────────────────────────────────────────────────────────────
Master Polling Loop — Step 8a
West Asia War 2026 Conflict Prediction Engine

Calls all 4 data modules every 15 minutes.
Saves timestamped outputs to data/.

USAGE
─────
  # Run once (manual trigger)
  python src/data_collector.py --mode once

  # Run continuously every 15 minutes
  python src/data_collector.py --mode loop
"""

import logging
import argparse
from datetime import datetime, timezone
from time import sleep
from pathlib import Path

from kinetic_pulse        import run_realtime as firms_realtime
from diplomatic_sentiment import run_realtime as sentiment_realtime
from economic_signals     import run_realtime as economic_realtime
from gdelt_kinetic        import run_realtime as gdelt_realtime

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
POLL_INTERVAL_SECONDS = 15 * 60   # 15 minutes


# ── Single poll cycle ─────────────────────────────────────────────────────────

def run_once() -> dict:
    """
    Run all 4 data modules once. Returns dict of results per module.
    Failed modules are logged and skipped — one failure never blocks others.
    """
    cycle_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    log.info("=" * 60)
    log.info("POLL CYCLE START — %s", cycle_time)
    log.info("=" * 60)

    results = {}

    # 1. NASA FIRMS — kinetic thermal anomalies
    log.info("[1/4] NASA FIRMS...")
    try:
        results["firms"] = firms_realtime()
        log.info("  → FIRMS done")
    except Exception as e:
        log.error("  → FIRMS FAILED: %s", e)
        results["firms"] = None

    # 2. GDELT Kinetic — CAMEO 18/19/20 events
    log.info("[2/4] GDELT Kinetic...")
    try:
        results["gdelt_kinetic"] = gdelt_realtime()
        log.info("  → GDELT Kinetic done")
    except Exception as e:
        log.error("  → GDELT Kinetic FAILED: %s", e)
        results["gdelt_kinetic"] = None

    # 3. GDELT Sentiment — DistilBERT + GMM
    log.info("[3/4] Diplomatic Sentiment...")
    try:
        results["sentiment"] = sentiment_realtime()
        log.info("  → Sentiment done")
    except Exception as e:
        log.error("  → Sentiment FAILED: %s", e)
        results["sentiment"] = None

    # 4. Yahoo Finance — economic signals
    log.info("[4/4] Economic Signals...")
    try:
        results["economic"] = economic_realtime()
        log.info("  → Economic done")
    except Exception as e:
        log.error("  → Economic FAILED: %s", e)
        results["economic"] = None

    # Summary
    # We count 'success' or 'empty' (no new data) as valid system states.
    success = sum(1 for v in results.values() if v is not None and v.get("status") in ["success", "empty"])
    log.info("POLL CYCLE COMPLETE — %d/4 modules succeeded", success)

    return results


# ── Continuous loop ───────────────────────────────────────────────────────────

def run_loop() -> None:
    """
    Run poll cycles continuously every 15 minutes.
    Ctrl+C to stop cleanly.
    """
    log.info("Polling loop started — interval: %d minutes", POLL_INTERVAL_SECONDS // 60)
    log.info("Press Ctrl+C to stop.")

    try:
        while True:
            run_once()
            log.info("Next poll in %d minutes...", POLL_INTERVAL_SECONDS // 60)
            sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        log.info("Polling loop stopped by user.")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Master polling loop — West Asia War 2026"
    )
    parser.add_argument(
        "--mode",
        choices=["once", "loop"],
        required=True,
        help="once: single poll cycle | loop: continuous every 15 min",
    )
    args = parser.parse_args()

    if args.mode == "once":
        run_once()
    elif args.mode == "loop":
        run_loop()