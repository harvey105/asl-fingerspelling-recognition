from pathlib import Path

import pandas as pd
import pytest

from src.utils.error_visualizer import get_canonical_path, get_tri_panel_cases
from src.utils.progression_visualizer import (
    get_bayes_residuals_items,
    get_progression_samples,
)


@pytest.fixture(scope="module")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def train_manifest(project_root: Path) -> pd.DataFrame:
    manifest_path = project_root / "data" / "splits" / "subject_train.csv"
    if not manifest_path.exists():
        pytest.skip("Train manifest data/splits/subject_train.csv not found")
    return pd.read_csv(manifest_path)


@pytest.fixture(scope="module")
def predictions_df(project_root: Path) -> pd.DataFrame:
    preds_path = (
        project_root / "temp" / "diagnostics" / "test_predictions_summary.csv"
    )
    if not preds_path.exists():
        pytest.skip(
            "Predictions summary temp/diagnostics/test_predictions_summary.csv not found"
        )
    return pd.read_csv(preds_path)


def test_get_canonical_path(train_manifest: pd.DataFrame, project_root: Path):
    for test_cls in ("0", "A", "W", "Z", "6"):
        canon_path = get_canonical_path(train_manifest, test_cls)
        assert canon_path.exists(), f"Canonical crop missing for {test_cls}: {canon_path}"


def test_tri_panel_cases_structure(predictions_df: pd.DataFrame):
    cases = get_tri_panel_cases(predictions_df)
    assert len(cases) == 6
    for c in cases:
        assert "category" in c
        assert "pair" in c
        assert "reason" in c
        assert "sample" in c
        assert "true_cls" in c
        assert "pred_cls" in c


def test_progression_samples_structure():
    samples = get_progression_samples()
    assert len(samples) == 5
    for s in samples:
        assert "class" in s
        assert "crop" in s
        assert "note" in s


def test_bayes_residuals_items_structure(predictions_df: pd.DataFrame):
    residuals = get_bayes_residuals_items(predictions_df)
    assert len(residuals) == 6
    for r in residuals:
        assert "title" in r
        assert "desc" in r
        assert "sample" in r
