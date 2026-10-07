"""
src/dataloaders.py
 DataLoader factory for the subject-independent splits.

get_dataloaders(train, val, test, ...) accepts DataFrames or CSV paths and returns
(train_loader, val_loader, test_loader).

Settings (roadmap ):
  batch_size=64, num_workers=DEFAULT_WORKERS (8), persistent_workers=True.
  pin_memory = torch.cuda.is_available().
  worker_init_fn limits each worker to one intra-op thread. Without this, 8 workers each start
  a multi-threaded pool on a 16-core CPU, which oversubscribes the cores and makes throughput
  slow and unstable. It changes scheduling only, never the data.

The training loader shuffles with a seeded generator and drops the last incomplete batch.
"""

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import ASLDataset

SEED = 42
DEFAULT_BATCH = 64
DEFAULT_WORKERS = 8   # 8 workers: 6 sessions on this 16-CPU laptop, train median 322-605 img/s (5 of 6 pass). See docs/task2_3_partB_sessions.jsonl.


def _worker_init(_worker_id):
    # Module-level so it can be pickled by the Windows 'spawn' start method.
    torch.set_num_threads(1)


def _as_manifest(m):
    if isinstance(m, pd.DataFrame):
        return m
    return pd.read_csv(m, dtype={"label": str, "filepath": str, "crop_path": str})


def get_dataloaders(train, val, test, batch_size=DEFAULT_BATCH, num_workers=DEFAULT_WORKERS,
                    seed=SEED):
    train_df, val_df, test_df = _as_manifest(train), _as_manifest(val), _as_manifest(test)
    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    common = dict(batch_size=batch_size, num_workers=num_workers,
                  pin_memory=pin, persistent_workers=persistent,
                  worker_init_fn=_worker_init if num_workers > 0 else None)

    train_loader = DataLoader(ASLDataset(train_df, is_train=True), shuffle=True,
                              drop_last=True,
                              generator=torch.Generator().manual_seed(seed), **common)
    val_loader = DataLoader(ASLDataset(val_df, is_train=False), shuffle=False, **common)
    test_loader = DataLoader(ASLDataset(test_df, is_train=False), shuffle=False, **common)
    return train_loader, val_loader, test_loader