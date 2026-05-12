"""
src/snapshot_manager.py
────────────────────────────────────────────────────────────────────────────────
Snapshot System — Step 8b
West Asia War 2026 Conflict Prediction Engine

Saves dated model output JSON to results/snapshots/YYYY-MM-DD.json every run.
Committed to GitHub. Powers historical replay slider on dashboard (Step 19).

USAGE
─────
  from snapshot_manager import save_snapshot, load_snapshot, list_snapshots

  # Save today's model output
  save_snapshot(date="2026-03-19", data={...})

  # Load a past snapshot
  snap = load_snapshot("2026-03-19")

  # List all snapshots
  dates = list_snapshots()
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR      = Path(__file__).resolve().parent.parent
SNAPSHOTS_DIR = ROOT_DIR / "results" / "snapshots"
SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
LATEST_PATH   = ROOT_DIR / "results" / "latest_output.json"


# ── Snapshot schema ───────────────────────────────────────────────────────────
# This is the contract every snapshot must follow.
# Phase 3 (Lanchester + Monte Carlo) fills in model_output and confidence_interval.
# Until then, those fields are null placeholders.

def _build_snapshot(
    date:         str,
    sentiment:    dict = None,
    kinetic:      dict = None,
    economic:     dict = None,
    model_output: dict = None,
    confidence_interval: dict = None,
    data_type:    str = "realtime",
) -> dict:
    """
    Build a snapshot dict conforming to the project schema.
    Null-safe — missing fields default to None, never crash.
    """
    return {
        "date":      date,
        "data_type": data_type,   # "realtime" | "historical_reconstruction"
        "generated_at": datetime.now(timezone.utc).isoformat(),

        "sentiment": {
            "distilbert_avg":    _get(sentiment, "distilbert_avg"),
            "hostile_weight":    _get(sentiment, "hostile_weight"),
            "diplomatic_weight": _get(sentiment, "diplomatic_weight"),
            "signal_divergence": _get(sentiment, "signal_divergence"),
        },

        "kinetic": {
            "frp_mean":       _get(kinetic, "frp_mean"),
            "anomaly_count":  _get(kinetic, "anomaly_count"),
            "shock_detected": _get(kinetic, "shock_detected", default=False),
            "gdelt_events":   _get(kinetic, "gdelt_events"),
        },

        "economic": {
            "brent_crude_change": _get(economic, "brent_crude_change"),
            "vix_change":         _get(economic, "vix_change"),
            "usd_ils_change":     _get(economic, "usd_ils_change"),
            "gold_change":        _get(economic, "gold_change"),
        },

        # Filled by Phase 3 (Steps 14–18) — null until then
        "model_output": {
            "ceasefire_probability":   _get(model_output, "ceasefire_probability"),
            "escalation_probability":  _get(model_output, "escalation_probability"),
            "stalemate_probability":   _get(model_output, "stalemate_probability"),
            "capitulation_probability":_get(model_output, "capitulation_probability"),
        },

        # Filled by Phase 3 Monte Carlo — null until then
        "confidence_interval": {
            "lower_95": _get(confidence_interval, "lower_95"),
            "upper_95": _get(confidence_interval, "upper_95"),
        },
    }


def _get(d: dict, key: str, default=None):
    """Safe dict getter — returns default if dict is None or key missing."""
    if d is None:
        return default
    return d.get(key, default)


# ── Public interface ──────────────────────────────────────────────────────────

def save_snapshot(
    date:         str  = None,
    sentiment:    dict = None,
    kinetic:      dict = None,
    economic:     dict = None,
    model_output: dict = None,
    confidence_interval: dict = None,
    data_type:    str  = "realtime",
    overwrite:    bool = False,
) -> Path:
    """
    Build and save a snapshot JSON for the given date.

    Args:
        date      : ISO date string e.g. "2026-03-19". Defaults to today.
        sentiment : Dict from diplomatic_sentiment output.
        kinetic   : Dict from kinetic_pulse + gdelt_kinetic output.
        economic  : Dict from economic_signals output.
        model_output       : Dict from Monte Carlo engine (Phase 3).
        confidence_interval: Dict from Monte Carlo engine (Phase 3).
        data_type : "realtime" or "historical_reconstruction"
        overwrite : If False, skip if snapshot already exists (default).

    Returns:
        Path to the saved snapshot file.
    """
    if date is None:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    snap_path = SNAPSHOTS_DIR / f"{date}.json"

    if snap_path.exists() and not overwrite:
        log.info("Snapshot already exists for %s — skipping (use overwrite=True to replace)", date)
        return snap_path

    snapshot = _build_snapshot(
        date=date,
        sentiment=sentiment,
        kinetic=kinetic,
        economic=economic,
        model_output=model_output,
        confidence_interval=confidence_interval,
        data_type=data_type,
    )

    # Write dated snapshot
    with open(snap_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    log.info("Snapshot saved → %s", snap_path)

    # Always overwrite latest_output.json with most recent run
    with open(LATEST_PATH, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    log.info("Latest output updated → %s", LATEST_PATH)

    return snap_path


def load_snapshot(date: str) -> dict | None:
    """
    Load a snapshot by date string.

    Args:
        date: ISO date string e.g. "2026-03-19"

    Returns:
        Snapshot dict, or None if not found.
    """
    snap_path = SNAPSHOTS_DIR / f"{date}.json"
    if not snap_path.exists():
        log.warning("No snapshot found for %s", date)
        return None

    with open(snap_path) as f:
        return json.load(f)


def load_latest() -> dict | None:
    """Load the most recent snapshot (latest_output.json)."""
    if not LATEST_PATH.exists():
        log.warning("No latest_output.json found — no snapshots saved yet.")
        return None
    with open(LATEST_PATH) as f:
        return json.load(f)


def list_snapshots() -> list[str]:
    """
    Return sorted list of all snapshot dates available.

    Returns:
        List of ISO date strings e.g. ["2026-02-28", "2026-03-01", ...]
    """
    files = sorted(SNAPSHOTS_DIR.glob("*.json"))
    dates = [f.stem for f in files]
    log.info("%d snapshots available: %s → %s",
             len(dates), dates[0] if dates else "none", dates[-1] if dates else "none")
    return dates


# ── CLI smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Snapshot Manager — smoke test")
    print(f"Snapshots directory: {SNAPSHOTS_DIR}")

    # Save a test snapshot with dummy data
    test_path = save_snapshot(
        date="2026-03-19",
        sentiment={
            "distilbert_avg":    -0.71,
            "hostile_weight":     0.89,
            "diplomatic_weight":  0.11,
            "signal_divergence":  0.43,
        },
        kinetic={
            "frp_mean":      47.3,
            "anomaly_count": 312,
            "shock_detected": True,
            "gdelt_events":  28,
        },
        economic={
            "brent_crude_change":  3.20,
            "vix_change":          2.1,
            "usd_ils_change":      0.04,
            "gold_change":        12.5,
        },
        data_type="historical_reconstruction",
        overwrite=True,
    )

    # Load it back and verify
    snap = load_snapshot("2026-03-19")
    print("\nLoaded snapshot:")
    print(json.dumps(snap, indent=2))

    # List all snapshots
    dates = list_snapshots()
    print(f"\nAll snapshots: {dates}")