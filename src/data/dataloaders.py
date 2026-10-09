from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import ASLDataset

SEED: int = 42
DEFAULT_BATCH: int = 64
DEFAULT_WORKERS: int = 8


def _worker_init(_worker_id: int) -> None:
    """Limit PyTorch intra-op threads per worker to avoid core oversubscription."""
    torch.set_num_threads(1)


def _as_manifest(m: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Normalize input manifest to a pandas DataFrame."""
    if isinstance(m, pd.DataFrame):
        return m
    return pd.read_csv(
        m, dtype={"label": str, "filepath": str, "crop_path": str}
    )


def get_dataloaders(
    train: pd.DataFrame | str | Path,
    val: pd.DataFrame | str | Path,
    test: pd.DataFrame | str | Path,
    batch_size: int = DEFAULT_BATCH,
    num_workers: int = DEFAULT_WORKERS,
    seed: int = SEED,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Construct training, validation, and test DataLoader pipelines."""
    train_df = _as_manifest(train)
    val_df = _as_manifest(val)
    test_df = _as_manifest(test)

    pin = torch.cuda.is_available()
    persistent = num_workers > 0
    common_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin,
        "persistent_workers": persistent,
        "worker_init_fn": _worker_init if num_workers > 0 else None,
    }

    train_loader = DataLoader(
        ASLDataset(train_df, is_train=True),
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        **common_kwargs,
    )
    val_loader = DataLoader(
        ASLDataset(val_df, is_train=False),
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
    )
    test_loader = DataLoader(
        ASLDataset(test_df, is_train=False),
        shuffle=False,
        batch_size=batch_size,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader
