"""
src/benchmark_loader.py
Task 2.3 Part B (fixed): data-pipeline throughput benchmark, no model, no GPU.

Worker count is fixed at DEFAULT_WORKERS (8), with no command-line option.

Method: one persistent iterator. A warm-up is followed by TRIALS timed trials. Each trial length
is computed from the loader's real size, so the iterator never runs out. The gate uses the
MEDIAN train trial. The validation loader is reported for information only and is not gated.

Writes docs/task2_3_partB_loader.json.
Run (from repository root): uv run python src/benchmark_loader.py
"""

import json
import os
import statistics
import sys
import time
from datetime import datetime
import subprocess

import torch

from config import DOCS_DIR, PROJECT_ROOT
from dataloaders import DEFAULT_WORKERS, get_dataloaders
from manifests import build_manifests

BATCH_SIZE = 64
TARGET_IMG_PER_S = 400
WARMUP_BATCHES = 10
TRIALS = 5
MAX_BATCHES_PER_TRIAL = 60
MIN_BATCHES_PER_TRIAL = 5
REPORT = DOCS_DIR / "task2_3_partB_loader.json"


def run_trials(loader):
    """Return per-trial throughput. Trial length is capped so that warm-up plus all trials fit."""
    available = len(loader) - WARMUP_BATCHES
    n_batches = min(MAX_BATCHES_PER_TRIAL, available // TRIALS)
    if n_batches < MIN_BATCHES_PER_TRIAL:
        raise ValueError(f"Loader too short: {len(loader)} batches, need at least "
                         f"{WARMUP_BATCHES + TRIALS * MIN_BATCHES_PER_TRIAL}.")
    it = iter(loader)
    for _ in range(WARMUP_BATCHES):
        next(it)
    rates = []
    for _ in range(TRIALS):
        n_img, t0 = 0, time.perf_counter()
        for _ in range(n_batches):
            x, _ = next(it)
            n_img += x.shape[0]
        rates.append(n_img / (time.perf_counter() - t0))
    return {"batches_per_trial": n_batches,
            "trials_img_per_s": [round(r, 1) for r in rates],
            "median_img_per_s": round(statistics.median(rates), 1),
            "min_img_per_s": round(min(rates), 1),
            "max_img_per_s": round(max(rates), 1)}

SESSIONS_LOG = DOCS_DIR / "task2_3_partB_sessions.jsonl"
def active_power_scheme():
    """Windows power plan at run time. Slow sessions often correlate with a power-saving plan."""
    try:
        out = subprocess.run(["powercfg", "/getactivescheme"], capture_output=True,
                             text=True, timeout=10).stdout.strip()
        return out or "unknown"
    except Exception:
        return "unavailable"

def main():
    print("=" * 70)
    print("ASL-HG: TASK 2.3 PART B (DATALOADER THROUGHPUT)")
    print("=" * 70)
    print(f"torch {torch.__version__}   CPUs: {os.cpu_count()}   CUDA: {torch.cuda.is_available()}")
    print(f"batch {BATCH_SIZE}   workers {DEFAULT_WORKERS} (fixed)   warm-up {WARMUP_BATCHES}   "
          f"trials {TRIALS} x up to {MAX_BATCHES_PER_TRIAL} batches")

    m = build_manifests()
    train_dl, val_dl, _ = get_dataloaders(m["train"], m["val"], m["test"],
                                          batch_size=BATCH_SIZE, num_workers=DEFAULT_WORKERS)
    print(f"batches per epoch: train {len(train_dl)}, val {len(val_dl)}")
    print(f"pin_memory: {train_dl.pin_memory}   persistent_workers: {train_dl.persistent_workers}")

    print("\nTiming train loader (augmented)...")
    tr = run_trials(train_dl)
    print(f"  trials: {tr['trials_img_per_s']}   median {tr['median_img_per_s']} img/s")

    print("Timing validation loader (eval transform, information only)...")
    va = run_trials(val_dl)
    print(f"  trials: {va['trials_img_per_s']}   median {va['median_img_per_s']} img/s")

    passed = tr["median_img_per_s"] >= TARGET_IMG_PER_S
    print("\n" + "=" * 70)
    print(f"GATE median train throughput >= {TARGET_IMG_PER_S} img/s: "
          f"{'PASS' if passed else 'FAIL'} ({tr['median_img_per_s']})")
    print(f"VERDICT: {'PASSED' if passed else 'FAILED'}")
    print("=" * 70)

    scheme = active_power_scheme()
    print(f"power plan: {scheme}")

    session = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "power_plan": scheme,
        "train_trials": tr["trials_img_per_s"],
        "train_median": tr["median_img_per_s"],
        "val_median_info": va["median_img_per_s"],
        "passed": passed,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    with open(SESSIONS_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(session, ensure_ascii=False) + "\n")

    report = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "torch": torch.__version__,
        "cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "batch_size": BATCH_SIZE,
        "num_workers": DEFAULT_WORKERS,
        "worker_threads": 1,
        "pin_memory": bool(train_dl.pin_memory),
        "persistent_workers": bool(train_dl.persistent_workers),
        "target_img_per_s": TARGET_IMG_PER_S,
        "gate_statistic": "median of trials, train loader",
        "train": tr,
        "val_information_only": va,
        "passed": passed,
        "power_plan": scheme,
    }
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {REPORT.relative_to(PROJECT_ROOT).as_posix()}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()