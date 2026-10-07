"""
tests/test_dataset.py
 unit tests for the dataset pipeline.

Data-dependent tests skip if the crops or index are missing. Synthetic tests always run.
Run from the repository root: uv run pytest tests/test_dataset.py -v
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision import transforms as T

from config import NUM_CLASSES, PROJECT_ROOT
from dataloaders import get_dataloaders
from dataset import (FILL, ROTATION_DEG, SCALE_RANGE, TRANSLATE, ASLDataset,
                     KeepPadJitter, build_transform, class_weights)
from manifests import build_manifests

BATCH = 64


@pytest.fixture(scope="module")
def manifests():
    try:
        return build_manifests()
    except FileNotFoundError as exc:
        pytest.skip(f"data not prepared: {exc}")


# ---------------------------------------------------------------- split integrity (data)

def test_train_has_no_val_or_test_signer(manifests):
    tr = set(manifests["train"]["volunteer"])
    va = set(manifests["val"]["volunteer"])
    te = set(manifests["test"]["volunteer"])
    assert tr.isdisjoint(va)
    assert tr.isdisjoint(te)
    assert va.isdisjoint(te)


def test_train_md5_unique(manifests):
    assert manifests["train"]["md5"].is_unique


def test_val_and_test_not_deduplicated(manifests):
    # : de-duplication applies to train only, so val and test keep their full crop output.
    rep = manifests["report"]
    assert rep["val_final"] == len(manifests["val"])
    assert rep["test_final"] == len(manifests["test"])


def test_labels_in_range_and_complete(manifests):
    for name in ("train", "val", "test"):
        ids = manifests[name]["class_id"].to_numpy()
        assert ids.min() >= 0 and ids.max() < NUM_CLASSES, name
        assert len(np.unique(ids)) == NUM_CLASSES, f"{name} is missing a class"


def test_no_blank_rows(manifests):
    from manifests import LOG_CSV  # noqa: F401  (ensures the same source is loaded)
    import pandas as pd
    from config import BLANK_FRAMES_CSV
    blank = set(pd.read_csv(BLANK_FRAMES_CSV, dtype={"filepath": str})["filepath"])
    all_fp = set(manifests["train"]["filepath"]) | set(manifests["val"]["filepath"]) \
        | set(manifests["test"]["filepath"])
    assert not (all_fp & blank)


# ---------------------------------------------------------------- batches (data)

@pytest.fixture(scope="module")
def loaders(manifests):
    # num_workers=0 keeps the tests fast and avoids process start-up on Windows.
    return get_dataloaders(manifests["train"], manifests["val"], manifests["test"],
                           batch_size=BATCH, num_workers=0)


def test_train_batch_shape_and_label_range(loaders):
    train_dl, _, _ = loaders
    x, y = next(iter(train_dl))
    assert tuple(x.shape) == (BATCH, 3, 224, 224)
    assert y.dtype == torch.int64
    assert int(y.min()) >= 0 and int(y.max()) < NUM_CLASSES


def test_eval_batch_shape(loaders):
    _, val_dl, test_dl = loaders
    for dl in (val_dl, test_dl):
        x, _ = next(iter(dl))
        assert tuple(x.shape) == (BATCH, 3, 224, 224)


def test_train_loader_drops_last_incomplete_batch(manifests, loaders):
    train_dl, _, _ = loaders
    n = len(manifests["train"])
    assert len(train_dl) * BATCH <= n < (len(train_dl) + 1) * BATCH


def test_val_loader_covers_every_image(manifests, loaders):
    _, val_dl, _ = loaders
    n = len(manifests["val"])
    assert len(val_dl) * BATCH >= n


def test_eval_transform_is_deterministic(manifests):
    ds = ASLDataset(manifests["val"], is_train=False)
    a, _ = ds[0]
    b, _ = ds[0]
    assert torch.equal(a, b)


# ---------------------------------------------------------------- augmentation rules (synthetic)

def _content_image():
    """224x224 RGB: 40 px gray padding on left and right, content values that never equal 128."""
    arr = np.zeros((224, 224, 3), dtype=np.uint8)
    arr[:] = np.array(FILL, dtype=np.uint8)
    ramp = (60 + (np.arange(224 - 80) % 50)).astype(np.uint8)   # values 60..109
    arr[:, 40:184, :] = ramp[None, :, None]
    return Image.fromarray(arr), arr


def test_keep_pad_jitter_restores_padding_exactly():
    torch.manual_seed(0)
    img, arr = _content_image()
    pad = np.all(arr == np.array(FILL, dtype=np.uint8), axis=2)
    out = np.asarray(KeepPadJitter(brightness=0.8, contrast=0.8, saturation=0.8)(img))
    assert np.all(out[pad] == np.array(FILL, dtype=np.uint8))


def test_keep_pad_jitter_still_changes_content():
    torch.manual_seed(0)
    img, arr = _content_image()
    out = np.asarray(KeepPadJitter(brightness=0.8, contrast=0.8, saturation=0.8)(img))
    content = np.all(arr != np.array(FILL, dtype=np.uint8), axis=2)
    assert np.any(out[content] != arr[content])


def test_affine_fill_is_exact_gray_outside_source():
    # Deterministic shrink by 0.75 with no rotation or translation: the corners are outside
    # the source image and must be exactly FILL.
    img = Image.fromarray(np.full((224, 224, 3), 40, dtype=np.uint8))
    shrink = T.RandomAffine(degrees=0, scale=(0.75, 0.75), fill=FILL)
    out = np.asarray(shrink(img))
    assert tuple(out[0, 0]) == FILL
    assert tuple(out[-1, -1]) == FILL
    assert tuple(out[112, 112]) == (40, 40, 40)


def test_training_pipeline_uses_configured_affine_jitter_and_flip():
    ops = build_transform(True).transforms
    names = [type(t).__name__ for t in ops]
    assert names == ["KeepPadJitter", "RandomAffine", "RandomHorizontalFlip", "ToTensor", "Normalize"]
    affine = ops[1]
    assert tuple(affine.degrees) == (-ROTATION_DEG, ROTATION_DEG)
    assert tuple(affine.translate) == TRANSLATE
    assert tuple(affine.scale) == SCALE_RANGE
    assert tuple(affine.fill) == FILL
    flip = ops[2]
    assert flip.p == 0.5


def test_eval_pipeline_has_no_augmentation():
    names = [type(t).__name__ for t in build_transform(False).transforms]
    assert "RandomHorizontalFlip" not in names
    assert "RandomAffine" not in names
    assert names == ["ToTensor", "Normalize"]


# ---------------------------------------------------------------- class weights (data)

def test_class_weights_are_balanced_per_sample(manifests):
    train = manifests["train"]
    w = class_weights(train)
    counts = train["class_id"].value_counts().reindex(range(NUM_CLASSES)).to_numpy()
    assert w.shape == (NUM_CLASSES,)
    assert torch.all(w > 0)
    # With w_c = N / (K * n_c), the sum over all samples of w_{class(sample)} equals N.
    total = float((torch.tensor(counts, dtype=torch.float32) * w).sum())
    assert abs(total - counts.sum()) < 1e-2