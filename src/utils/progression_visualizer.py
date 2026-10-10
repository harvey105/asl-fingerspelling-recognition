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


def get_progression_samples() -> list[dict[str, str]]:
    return [
        {
            "class": "K",
            "crop": "data/processed_crops/K/P10_K_928.png",
            "note": "Thumb placed between index and middle.\nM1 & M2 confuse it with 'V'; M3 resolves it.",
        },
        {
            "class": "E",
            "crop": "data/processed_crops/E/P10_E_901.png",
            "note": "Fingertips curled resting against thumb.\nM1 predicts 'S', M2 predicts 'M'; M3 recovers 'E'.",
        },
        {
            "class": "8",
            "crop": "data/processed_crops/8/P9_8_801.png",
            "note": "Middle finger bent touching thumb.\nM1 predicts '3', M2 predicts '7'; M3 isolates '8'.",
        },
        {
            "class": "D",
            "crop": "data/processed_crops/D/P9_D_801.png",
            "note": "Index pointing straight up with thumb loop.\nM1 predicts 'U'; M2 and M3 recover 'D'.",
        },
        {
            "class": "M",
            "crop": "data/processed_crops/M/P9_M_801.png",
            "note": "Thumb tucked under 3 folded fingers.\nM1 predicts 'N'; M2 and M3 resolve 'M'.",
        },
    ]


def generate_progression_montage(
    df: pd.DataFrame, train_manifest: pd.DataFrame, output_path: Path
) -> None:
    samples = get_progression_samples()
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Segoe UI",
        "DejaVu Sans",
        "Helvetica",
        "Arial",
    ]

    n_rows = len(samples)
    fig = plt.figure(figsize=(16.8, 2.5 * n_rows), dpi=300)
    gs = fig.add_gridspec(
        n_rows,
        6,
        width_ratios=[2.3, 1.0, 1.0, 1.15, 1.15, 1.15],
        wspace=0.22,
        hspace=0.28,
    )

    for i, item in enumerate(samples):
        s = df[df["crop_path"] == item["crop"]].iloc[0]
        crop_path = PROJECT_ROOT / s["crop_path"]
        canon_path = get_canonical_path(train_manifest, item["class"])

        ax_desc = fig.add_subplot(gs[i, 0])
        ax_desc.axis("off")
        desc = (
            f"Target Sign: '{item['class']}' (Signer P{s['volunteer_id']})\n\n"
            f"Handshape detail:\n{item['note']}"
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

        ax_ref = fig.add_subplot(gs[i, 1])
        ax_ref.imshow(Image.open(canon_path).convert("RGB"))
        ax_ref.set_title(
            f"True '{item['class']}' (Ref)",
            fontsize=9.0,
            fontweight="600",
            color="#047857",
            pad=5,
        )
        ax_ref.axis("off")

        ax_crop = fig.add_subplot(gs[i, 2])
        ax_crop.imshow(Image.open(crop_path).convert("RGB"))
        ax_crop.set_title(
            f"Test Crop (P{s['volunteer_id']})",
            fontsize=9.0,
            fontweight="600",
            color="#334155",
            pad=5,
        )
        ax_crop.axis("off")

        for m_idx, (m_num, m_name, col_key) in enumerate(
            [
                (1, "Model 1: Simple", "model1"),
                (2, "Model 2: Complex", "model2"),
                (3, "Model 3: Transfer", "model3"),
            ]
        ):
            pred = s[f"{col_key}_pred_name"]
            conf = s[f"{col_key}_conf"] * 100
            correct = s[f"{col_key}_correct"] == 1
            col_tone = "#047857" if correct else "#b91c1c"
            mark = "Correct" if correct else "Mistaken"

            ax_m = fig.add_subplot(gs[i, 3 + m_idx])
            ax_m.axis("off")
            txt = (
                f"{m_name}\n"
                f"Prediction: '{pred}'\n"
                f"Confidence: {conf:.1f}%\n"
                f"Result: {mark}"
            )
            ax_m.text(
                0.05,
                0.5,
                txt,
                transform=ax_m.transAxes,
                fontsize=8.5,
                va="center",
                ha="left",
                color=col_tone,
                linespacing=1.35,
                bbox=dict(
                    boxstyle="round,pad=0.5",
                    facecolor="#f0fdf4" if correct else "#fef2f2",
                    edgecolor=col_tone,
                    lw=1.1,
                ),
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def get_bayes_residuals_items(df: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            "title": "Hidden Thumb Placement",
            "desc": (
                "Sign 'N' mistaken for 'M' (39.3% conf).\n"
                "The thumb tucks under 2 fingers in 'N' vs 3 in 'M'. From this camera\n"
                "angle, the thumb tip cannot be seen beneath the curled fingers."
            ),
            "ref_cls": "M",
            "sample": df[
                (df["class_name"] == "N")
                & (df["model3_pred_name"] == "M")
                & (df["volunteer_id"] == 9)
            ].iloc[0],
        },
        {
            "title": "Flattened Finger Crossing",
            "desc": (
                "Sign 'R' mistaken for 'U' (30.0% conf).\n"
                "The index crosses behind the middle finger in 3D space. Facing the camera\n"
                "directly, the crossing looks like two parallel fingers ('U')."
            ),
            "ref_cls": "U",
            "sample": df[
                (df["class_name"] == "R")
                & (df["model3_pred_name"] == "U")
                & (df["volunteer_id"] == 10)
            ].iloc[0],
        },
        {
            "title": "Palm Shadow Obscuring Thumb",
            "desc": (
                "Sign 'N' mistaken for 'M' (41.2% conf).\n"
                "Slight hand rotation casts a shadow over the palm crease,\n"
                "making it hard to verify how many fingers the thumb rests under."
            ),
            "ref_cls": "M",
            "sample": df[
                (df["class_name"] == "N")
                & (df["model3_pred_name"] == "M")
                & (df["volunteer_id"] == 9)
            ].iloc[40],
        },
        {
            "title": "Angle Hiding Depth",
            "desc": (
                "Sign 'R' mistaken for 'U' (31.5% conf).\n"
                "Wrist tilt lines up the crossed fingers directly with the lens, removing\n"
                "the visible overlap needed to tell 'R' from 'U'."
            ),
            "ref_cls": "U",
            "sample": df[
                (df["class_name"] == "R")
                & (df["model3_pred_name"] == "U")
                & (df["volunteer_id"] == 10)
            ].iloc[30],
        },
        {
            "title": "Thumb Tip Covered by Joint",
            "desc": (
                "Sign 'T' mistaken for 'N' (35.8% conf).\n"
                "In 'T', the thumb tip pops out between index and middle fingers.\n"
                "Here, the bent index joint blocks the thumb tip from view."
            ),
            "ref_cls": "N",
            "sample": df[
                (df["class_name"] == "T") & (df["model3_pred_name"] == "N")
            ].iloc[0],
        },
        {
            "title": "Foreshortened Knuckle Lines",
            "desc": (
                "Sign 'N' mistaken for 'M' (38.7% conf).\n"
                "Viewing the clenched fist straight-on compresses finger creases\n"
                "into a single dark line, hiding the exact knuckle count."
            ),
            "ref_cls": "M",
            "sample": df[
                (df["class_name"] == "N")
                & (df["model3_pred_name"] == "M")
                & (df["volunteer_id"] == 9)
            ].iloc[70],
        },
    ]


def generate_bayes_residuals_figure(
    df: pd.DataFrame, train_manifest: pd.DataFrame, output_path: Path
) -> None:
    residuals = get_bayes_residuals_items(df)
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Segoe UI",
        "DejaVu Sans",
        "Helvetica",
        "Arial",
    ]

    n_rows = len(residuals)
    fig = plt.figure(figsize=(16.8, 2.5 * n_rows), dpi=300)
    gs = fig.add_gridspec(
        n_rows,
        4,
        width_ratios=[2.2, 1.0, 1.0, 1.4],
        wspace=0.22,
        hspace=0.28,
    )

    for i, item in enumerate(residuals):
        s = item["sample"]
        crop_path = PROJECT_ROOT / s["crop_path"]
        ref_path = get_canonical_path(train_manifest, item["ref_cls"])

        ax_diag = fig.add_subplot(gs[i, 0])
        ax_diag.axis("off")
        diag_text = f"{item['title']}\n\n{item['desc']}"
        ax_diag.text(
            0.02,
            0.5,
            diag_text,
            transform=ax_diag.transAxes,
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

        ax_test = fig.add_subplot(gs[i, 1])
        ax_test.imshow(Image.open(crop_path).convert("RGB"))
        ax_test.set_title(
            f"Test Crop: True '{s['class_name']}'\n(P{s['volunteer_id']})",
            fontsize=9.0,
            fontweight="700",
            color="#b91c1c",
            pad=5,
        )
        ax_test.axis("off")

        ax_ref = fig.add_subplot(gs[i, 2])
        ax_ref.imshow(Image.open(ref_path).convert("RGB"))
        ax_ref.set_title(
            f"Mistaken Sign: '{item['ref_cls']}'\n(Reference)",
            fontsize=9.0,
            fontweight="600",
            color="#1d4ed8",
            pad=5,
        )
        ax_ref.axis("off")

        ax_meta = fig.add_subplot(gs[i, 3])
        ax_meta.axis("off")
        meta_text = (
            f"2D Camera Limitation\n\n"
            f"True sign:     '{s['class_name']}'\n"
            f"Predicted:     '{s['model3_pred_name']}'\n"
            f"Confidence:    {s['model3_conf']*100:.1f}%\n"
            f"Root cause:    Hidden depth in 2D"
        )
        ax_meta.text(
            0.02,
            0.5,
            meta_text,
            transform=ax_meta.transAxes,
            fontsize=8.5,
            va="center",
            ha="left",
            color="#b91c1c",
            linespacing=1.35,
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="#fef2f2",
                edgecolor="#fca5a5",
                lw=1.1,
            ),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
