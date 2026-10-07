"""
src/eda_figures.py
Task 2.1, Part C (figures): the three EDA artifacts required by the roadmap for ASL-HG.

Samples are chosen with a fixed seed and no quality filter, so every image shown is
representative and reproducible. Reads the cache written by audit_signers.py and decodes
only the sample images.

Outputs (docs/figures/):
  eda_class_balance.png       class counts, with the 1,000 target line
  eda_36_class_grid.png       6x6 grid, one seeded sample per class
  eda_volunteer_mosaic.png    one seeded image per volunteer, same class, with brightness stats

Run (from the repository root): uv run python src/eda_figures.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from config import CLASSES, DOCS_DIR, IMAGES_PER_CLASS, NUM_CLASSES, PROCESSED_DIR, PROJECT_ROOT, NUM_VOLUNTEERS

SEED = 42
INDEX_CSV = PROCESSED_DIR / "asl_hg_image_index.csv"
FIG_DIR = DOCS_DIR / "figures"
MOSAIC_CLASS = "A"          # one fixed sign, so signer differences are not confounded with the sign
DIGITS = set(CLASSES[:10])

assert NUM_CLASSES == 36, "The 6x6 grid assumes 36 classes."


def load(rel_path):
    with Image.open(PROJECT_ROOT / rel_path) as im:
        return im.convert("RGB").copy()


def class_balance(df, out):
    counts = df["label"].value_counts().reindex(CLASSES)
    colors = ["#d95f02" if c in DIGITS else "#2b5c8f" for c in CLASSES]
    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.bar(CLASSES, counts.values, color=colors, edgecolor="black", linewidth=0.6)
    ax.axhline(IMAGES_PER_CLASS, color="crimson", linestyle="--", linewidth=1.2,
               label=f"Target: {IMAGES_PER_CLASS:,} per class")
    ax.set_ylim(0, IMAGES_PER_CLASS * 1.3)
    ax.set_title(f"ASL-HG class balance ({len(df):,} images; orange = digits, blue = letters)",
                 fontsize=12, weight="bold")
    ax.set_xlabel("Class")
    ax.set_ylabel("Image count")
    ax.legend(loc="upper center", frameon=True)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def grid_36(df, out):
    fig, axes = plt.subplots(6, 6, figsize=(12, 13))
    for k, (ax, cls) in enumerate(zip(axes.flatten(), CLASSES)):
        vol = (k % NUM_VOLUNTEERS) + 1                   # rotate signers P1..P10 across classes
        pool = df[(df["label"] == cls) & (df["volunteer"] == vol)]
        row = pool.sample(1, random_state=SEED).iloc[0]
        ax.imshow(load(row["filepath"]))
        ax.set_title(f"'{cls}'  (P{vol})", fontsize=10, weight="bold", pad=3)
        ax.axis("off")
    fig.suptitle("ASL-HG: one seeded sample per class; signer rotates P1-P10 (volunteer in brackets)",
                 fontsize=13, weight="bold", y=0.995)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def volunteer_mosaic(df, out):
    sub = df[df["label"] == MOSAIC_CLASS]
    vols = sorted(sub["volunteer"].unique())
    fig, axes = plt.subplots(2, 5, figsize=(14, 6.8))
    for ax, v in zip(axes.flatten(), vols):
        row = sub[sub["volunteer"] == v].sample(1, random_state=SEED).iloc[0]
        ax.imshow(load(row["filepath"]))
        ax.set_title(f"P{v}   mean L {row['mean_L']:.0f}   border L {row['border_mean_L']:.0f}",
                     fontsize=9.5, weight="bold", pad=4)
        ax.axis("off")
    fig.suptitle(f"ASL-HG: signers and lighting, sign '{MOSAIC_CLASS}' (one seeded image per volunteer)",
                 fontsize=12, weight="bold", y=0.99)
    fig.text(0.5, 0.01,
             "Train signers P1–P7, validation P8, test P9–P10. Brightness is from the whole image "
             "and the outer 10% border. Border values are a background proxy only.",
             ha="center", fontsize=9, style="italic", color="#444444")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(out, dpi=200)
    plt.close(fig)


def main():
    print("=" * 70)
    print("ASL-HG: TASK 2.1 PART C (EDA FIGURES)")
    print("=" * 70)

    df = pd.read_csv(INDEX_CSV, dtype={"md5": str, "dhash": str})
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    class_balance(df, FIG_DIR / "eda_class_balance.png")
    print("  saved eda_class_balance.png")
    grid_36(df, FIG_DIR / "eda_36_class_grid.png")
    print("  saved eda_36_class_grid.png")
    volunteer_mosaic(df, FIG_DIR / "eda_volunteer_mosaic.png")
    print("  saved eda_volunteer_mosaic.png")

    print("=" * 70)
    print(f"Figures written to {FIG_DIR.relative_to(PROJECT_ROOT).as_posix()}/")


if __name__ == "__main__":
    main()