from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def get_canonical_path(manifest: pd.DataFrame, class_name: str) -> Path:
    sub = manifest[manifest["class_name"] == class_name]
    if len(sub) == 0:
        sub = manifest[manifest["class_name"] == int(class_name)]
    for p in (1, 2, 3):
        p_sub = sub[sub["volunteer_id"] == p]
        if len(p_sub) > 0:
            return PROJECT_ROOT / p_sub.iloc[len(p_sub) // 2]["crop_path"]
    return PROJECT_ROOT / sub.iloc[0]["crop_path"]


def get_tri_panel_cases(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "category": "Identical 2D Silhouette",
            "pair": "Sign '6' mistaken for 'W'",
            "reason": (
                "Both signs extend 3 middle fingers. The thumb touching the pinky\n"
                "in '6' is hidden from the front camera angle."
            ),
            "sample": df[
                (df["class_name"] == "6") & (df["model3_pred_name"] == "W")
            ].iloc[0],
            "true_cls": "6",
            "pred_cls": "W",
        },
        {
            "category": "Identical 2D Silhouette",
            "pair": "Sign '2' mistaken for 'V'",
            "reason": (
                "Both signs form an open V-shape. The thumb tuck difference is\n"
                "obscured under a flat front-facing view."
            ),
            "sample": df[
                (df["class_name"] == "2") & (df["model3_pred_name"] == "V")
            ].iloc[0],
            "true_cls": "2",
            "pred_cls": "V",
        },
        {
            "category": "Static Snapshot of Motion",
            "pair": "Sign '1' mistaken for 'Z'",
            "reason": (
                "Sign 'Z' traces a zigzag in the air. A static single-frame snapshot\n"
                "at the start captures just 1 extended index finger."
            ),
            "sample": df[
                (df["class_name"] == "1") & (df["model3_pred_name"] == "Z")
            ].iloc[0],
            "true_cls": "1",
            "pred_cls": "Z",
        },
        {
            "category": "Static Snapshot of Motion",
            "pair": "Sign 'I' mistaken for 'J'",
            "reason": (
                "Sign 'J' traces a hook motion starting from the pinky posture of 'I'.\n"
                "Without motion over time, a static frame looks identical to 'I'."
            ),
            "sample": df[
                (df["class_name"] == "I") & (df["model3_pred_name"] == "J")
            ].iloc[0],
            "true_cls": "I",
            "pred_cls": "J",
        },
        {
            "category": "Subtle Finger Tucking",
            "pair": "Sign 'N' mistaken for 'M'",
            "reason": (
                "Thumb tucked under 2 fingers ('N') vs 3 fingers ('M'). The thumb\n"
                "tip protrusion is hidden beneath the folded fingers."
            ),
            "sample": df[
                (df["class_name"] == "N") & (df["model3_pred_name"] == "M")
            ].iloc[0],
            "true_cls": "N",
            "pred_cls": "M",
        },
        {
            "category": "Finger Overlap in 2D",
            "pair": "Sign 'R' mistaken for 'U'",
            "reason": (
                "In 'R', the index crosses the middle finger in 3D depth. Seen directly\n"
                "from the front, the crossing flattens into parallel fingers ('U')."
            ),
            "sample": df[
                (df["class_name"] == "R") & (df["model3_pred_name"] == "U")
            ].iloc[0],
            "true_cls": "R",
            "pred_cls": "U",
        },
    ]


def generate_tri_panel_figure(
    df: pd.DataFrame, train_manifest: pd.DataFrame, output_path: Path
) -> None:
    cases = get_tri_panel_cases(df)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Segoe UI",
        "DejaVu Sans",
        "Helvetica",
        "Arial",
    ]

    n_cases = len(cases)
    fig = plt.figure(figsize=(15.2, 2.5 * n_cases), dpi=300)
    gs = fig.add_gridspec(
        n_cases,
        4,
        width_ratios=[1.8, 1.0, 1.0, 1.0],
        wspace=0.20,
        hspace=0.28,
    )

    for i, c in enumerate(cases):
        s = c["sample"]
        gt_path = get_canonical_path(train_manifest, c["true_cls"])
        test_path = PROJECT_ROOT / s["crop_path"]
        pred_path = get_canonical_path(train_manifest, c["pred_cls"])

        ax_desc = fig.add_subplot(gs[i, 0])
        ax_desc.axis("off")
        desc = (
            f"{c['category']}\n"
            f"{c['pair']}\n\n"
            f"Visual explanation:\n{c['reason']}"
        )
        ax_desc.text(
            0.02,
            0.5,
            desc,
            transform=ax_desc.transAxes,
            fontsize=8.5,
            va="center",
            ha="left",
            color="#1e293b",
            linespacing=1.35,
            bbox=dict(
                boxstyle="square,pad=0.55",
                facecolor="#f8fafc",
                edgecolor="#cbd5e1",
                lw=1.0,
            ),
        )

        ax_gt = fig.add_subplot(gs[i, 1])
        ax_gt.imshow(Image.open(gt_path).convert("RGB"))
        ax_gt.set_title(
            f"True Sign: '{c['true_cls']}' (Target)",
            fontsize=9.0,
            fontweight="600",
            color="#047857",
            pad=5,
        )
        ax_gt.axis("off")

        ax_test = fig.add_subplot(gs[i, 2])
        ax_test.imshow(Image.open(test_path).convert("RGB"))
        ax_test.set_title(
            f"Test Sample (P{s['volunteer_id']})\nPredicted: '{c['pred_cls']}' ({s['model3_conf']*100:.1f}%)",
            fontsize=9.0,
            fontweight="700",
            color="#b91c1c",
            pad=5,
        )
        ax_test.axis("off")

        ax_pred = fig.add_subplot(gs[i, 3])
        ax_pred.imshow(Image.open(pred_path).convert("RGB"))
        ax_pred.set_title(
            f"Mistaken Sign: '{c['pred_cls']}' (Reference)",
            fontsize=9.0,
            fontweight="600",
            color="#1d4ed8",
            pad=5,
        )
        ax_pred.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
