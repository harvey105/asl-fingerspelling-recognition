import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.error_visualizer import generate_tri_panel_figure
from src.utils.progression_visualizer import (
    generate_bayes_residuals_figure,
    generate_progression_montage,
)


def export_diagnostic_figures(
    predictions_csv: Path,
    train_manifest_csv: Path,
    output_dir: Path,
) -> None:
    df = pd.read_csv(predictions_csv)
    train_manifest = pd.read_csv(train_manifest_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    tri_panel_path = output_dir / "tri_panel_failure_comparisons.png"
    print(f"Generating {tri_panel_path}...")
    generate_tri_panel_figure(df, train_manifest, tri_panel_path)

    progression_path = output_dir / "architectural_progression_montage.png"
    print(f"Generating {progression_path}...")
    generate_progression_montage(df, train_manifest, progression_path)

    bayes_path = output_dir / "bayes_error_bound_residuals.png"
    print(f"Generating {bayes_path}...")
    generate_bayes_residuals_figure(df, train_manifest, bayes_path)

    print("All diagnostic figures successfully generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate qualitative failure case figures"
    )
    parser.add_argument(
        "--predictions-csv",
        type=Path,
        default=PROJECT_ROOT
        / "docs"
        / "metrics"
        / "test_predictions_summary.csv",
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "splits" / "subject_train.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "docs" / "figures" / "diagnostics",
    )
    args = parser.parse_args()

    export_diagnostic_figures(
        args.predictions_csv, args.train_manifest, args.output_dir
    )
