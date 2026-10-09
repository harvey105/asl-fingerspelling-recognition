from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src.config import CROPS_DIR, NUM_VOLUNTEERS

DEFAULT_INSPECT_CLASSES: list[str] = [
    "0",
    "5",
    "W",
    "B",
    "C",
    "H",
    "Y",
    "K",
    "U",
    "3",
]


def generate_crop_inspection_grid(
    df: pd.DataFrame,
    out_path: Path,
    inspect_classes: list[str] = DEFAULT_INSPECT_CLASSES,
    seed: int = 42,
) -> None:
    """Generate a diagnostic grid showing sample crops across volunteer signers."""
    valid_methods = {"mediapipe", "skin", "class_rule_full", "undetected_full"}
    valid_records = df[df["method"].isin(valid_methods)]

    fig, axes = plt.subplots(2, 5, figsize=(14, 6.4))
    for ax, cls_name, volunteer_id in zip(
        axes.flatten(), inspect_classes, range(1, NUM_VOLUNTEERS + 1)
    ):
        ax.axis("off")
        subset = valid_records[
            (valid_records["label"] == cls_name)
            & (valid_records["volunteer"] == volunteer_id)
        ]
        if subset.empty:
            ax.set_title(f"'{cls_name}' P{volunteer_id}: no sample", fontsize=9)
            continue

        selected_row = subset.sample(1, random_state=seed).iloc[0]
        crop_file = (
            CROPS_DIR
            / str(cls_name)
            / (Path(selected_row["filepath"]).stem + ".png")
        )
        with Image.open(crop_file) as img:
            ax.imshow(img.convert("RGB"))

        method_name = selected_row["method"]
        ax.set_title(
            f"'{cls_name}' P{volunteer_id} [{method_name}]",
            fontsize=10,
            weight="bold",
        )

    fig.suptitle(
        "ASL-HG: 20% margin, letterboxed 224x224 crops across signers",
        fontsize=11,
        weight="bold",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
