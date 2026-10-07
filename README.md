# Real-Time American Sign Language (ASL) Fingerspelling Recognition

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange.svg)](https://pytorch.org/)
[![Package Manager](https://img.shields.io/badge/uv-astral.sh-purple.svg)](https://github.com/astral-sh/uv)

> **Course:** Deep Learning Midterm Project, supervised by **Dr. Nguyễn Thị Kim Ngân**
> 
> **Modality:** Image classification across 29 classes (`A`–`Z`, `del`, `nothing`, `space`)
> 
> **Targets:** Model 3 test accuracy 98.5–99.8%; CPU inference under 15 ms per frame (Intel Core i5-13500H)

---

## 1. Project Status

| Task | Status |
|---|---|
| 1.1 Repository and environment | Done for setup; `uv.lock` committed |
| 2.1 Dataset integrity audit and EDA | Done (integrity verified; EDA figures and write-up in `docs/task2_1_eda_analysis.md`) |
| 2.4 Stratified split | Done (exact rubric counts; residual near-duplicate leakage being measured) |
| 2.4 External real-world set | Pending (team photo collection, 3 per class) |
| 2.2 Hand ROI cropping | Not started |
| 2.3 Background re-compositing and DataLoader | Not started |
| 3.x–6.x Models, evaluation, app, report | Not started |

## 2. Overview

Benchmark ASL datasets show hands against plain indoor walls. A CNN trained on raw images can learn background features that correlate with a label instead of finger shape (shortcut learning [1]). Such a model can score highly on the benchmark test split and still perform poorly on webcam images taken in a different room.

The project addresses this with a three-tier model progression and three planned pipeline steps:

1. **Hand ROI cropping (Task 2.2):** MediaPipe hand detection with 10–15% margin padding and a skin-color fallback, producing standardized crops and foreground masks computed offline.
2. **Background re-compositing (Task 2.3):** During training, the foreground is blended with a bank of cluttered backgrounds held in RAM, with probability p = 0.5.
3. **Temporal word assembly (Task 5.1):** A sliding-window majority vote filters single-frame flicker; a letter is appended only after it is held for N consecutive frames.

## 3. Model Progression (Rubric)

All three models use identical data splits and are evaluated on the same test instances for Accuracy, Precision (macro), Recall (macro), and F1 (macro).

| Tier | Architecture | Structural requirement | Target accuracy | Size / latency |
|---|---|---|---|---|
| Model 1 | Simple CNN, from scratch | Sequential Conv-BN-ReLU-Pool stages; **no skip connections** | 91–94% | under 1.5M parameters |
| Model 2 | Mini-ResNet, from scratch | Residual blocks $\mathbf{x}_{l+1} = \mathcal{F}(\mathbf{x}_l) + \mathbf{x}_l$; spatial dropout 0.2–0.4 | 95–97% | roughly 4–8M parameters |
| Model 3 | Transfer learning, MobileNetV3-Small (or EfficientNet-B0) | Pretrained torchvision backbone, new 29-class head; two-phase fine-tuning | 98.5–99.8% | ~0.06 GFLOPs; under 15 ms on CPU |

## 4. Repository Layout

Items marked **(planned)** do not exist yet.

```text
asl-fingerspelling-recognition/
├── .gitignore                    # ignores raw data, weights, .venv; keeps split CSVs
├── pyproject.toml                # dependencies, managed with uv
├── uv.lock                       # exact resolved versions (commit)
├── README.md
├── data/
│   ├── raw/                      # Kaggle download (not committed)
│   │   ├── asl_alphabet_train/   # 87,000 images: the only folder used for splits
│   │   └── asl_alphabet_test/    # Kaggle's separate small folder: not used
│   ├── processed/                # dHash cache; crops and masks (Task 2.2, planned)
│   ├── backgrounds/              # background bank (Task 2.3, planned)
│   ├── external_test/            # team webcam photos (not committed) + external_manifest.csv (committed)
│   └── splits/                   # train.csv, val.csv, test.csv, class_mapping.json, split_report.json (committed)
├── src/
│   ├── verify_dataset.py         # Task 2.1: integrity audit
│   ├── split_data.py             # Task 2.4: group-aware stratified split
│   ├── audit_leakage.py          # Task 2.4: residual leakage diagnostic
│   ├── setup_external_test.py    # Task 2.4: external test manifest
│   ├── preprocess_roi.py         # Task 2.2 (planned)
│   ├── dataset.py                # Task 2.3 (planned)
│   ├── train.py                  # Task 3.1 (planned)
│   ├── evaluate.py               # Task 4.1 (planned)
│   ├── gradcam.py                # Task 4.3 (planned)
│   ├── utils.py
│   └── models/
│       ├── simple_cnn.py         # Model 1 (planned)
│       ├── complex_cnn.py        # Model 2 (planned)
│       └── transfer_cnn.py       # Model 3 (planned)
├── notebooks/
│   └── 01_dataset_eda.ipynb      # Task 2.1
├── app/                          # Task 5.x (planned)
├── weights/                      # *.pt checkpoints (not committed)
├── docs/
│   ├── figures/                  # EDA and report figures
│   ├── task2_1_integrity_report.json
│   ├── task2_1_eda_analysis.md
│   ├── task2_2_roi_cropping.md   # (planned)
│   ├── task2_4_stratifiedsplit_benchmark.md
│   ├── task2_4_leakage_report.json
│   ├── ablation_results.md       # (planned)
│   └── error_analysis.md         # (planned)
└── tests/
    ├── __init__.py
    └── test_env.py               # environment smoke test
```

## 5. Setup

The project uses [uv](https://github.com/astral-sh/uv). Use `uv sync` to install and `uv run` to execute. Do not install packages with `uv pip install` or `pip`: `uv run` removes packages that `pyproject.toml` does not declare. Add packages with `uv add`.

### 5.1 Install uv

- **Windows (PowerShell):** `powershell -c "irm https://astral.sh/uv/install.ps1 | iex"`
- **macOS / Linux:** `curl -LsSf https://astral.sh/uv/install.sh | sh`

### 5.2 Create the environment (from the repository root)

```bash
uv sync                          # creates .venv and installs uv.lock
uv run python tests/test_env.py  # expected: ALL ENVIRONMENT CHECKS PASSED SUCCESSFULLY!
```

### 5.3 Add a package

```bash
uv add <package>                 # records the change in pyproject.toml and uv.lock
uv add --dev <package>           # development-only tools
```

`protobuf` is pinned in `pyproject.toml` for MediaPipe compatibility. Do not change that pin without re-running `tests/test_env.py`.

### 5.4 Run the data scripts (in this order)

```bash
uv run python src/verify_dataset.py    # integrity audit; exit code 0 = passed
uv run python src/split_data.py        # group-aware split (first run hashes all images, ~2 min)
uv run python src/audit_leakage.py     # residual near-duplicate leakage report
uv run python src/setup_external_test.py  # creates class folders for team photos; writes manifest
```

### 5.5 Hardware

- **CPU (presentation laptop, Intel Core i5-13500H):** no extra steps. Inference uses `torch.set_num_threads(4)`.
- **NVIDIA GPU:** install the CUDA build of PyTorch, as configured for your machine.
- **AMD GPU (RX 6700 XT):** ROCm builds of PyTorch are Linux-only. Use that machine for ROCm training.
- **Kaggle GPU (T4/P100):** used for heavy training. Split CSV paths are relative to the project root, so set the root explicitly on Kaggle.

## 6. Dataset and Split Policy

- **Source:** [Kaggle ASL Alphabet (grassknoted)](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **Scale:** 87,000 RGB JPEG images, 200×200 px, 29 classes with 3,000 images each (~1.1 GB).
- **Integrity (Task 2.1):** all 87,000 images in `data/raw/asl_alphabet_train/` decode fully, are 200×200 RGB, and are non-empty. No stray files. Report: `docs/task2_1_integrity_report.json`.
- **Used:** `asl_alphabet_train/` only. The separate Kaggle test folder is not used in any split or analysis.
- **Split (Task 2.4):** 70 / 15 / 15, stratified by class, seed 42. Counts: 60,900 / 13,050 / 13,050, which is exactly 2,100 / 450 / 450 per class.
- **Grouping:** images are grouped by 64-bit difference hash (dHash). At Hamming distance 0 (identical hashes), each group stays in one split. Final threshold: t = 0, the largest value at which all groups fit the 450-image val/test quota. Threshold sweep: `data/splits/split_report.json`.
- **Known limitation:** near-duplicates at distance 1 or more are not grouped, so some cross-split near-duplicates remain. The residual amount is measured in `docs/task2_4_leakage_report.json` and reported in the write-up. Test accuracy on `test.csv` should be read as an in-distribution estimate that may be optimistic.
- **Stability rule:** the split CSVs are frozen once training starts. Any regeneration must be recorded in `docs/task2_4_stratifiedsplit_benchmark.md`, and every model is trained and evaluated on the same files.
- **External test set (Task 2.4, pending):** 3 original webcam or phone photos per class (87 total; rubric range 60–100), taken under varied lighting and rooms by team members. Used for zero-shot generalization. Photos are not committed; they are shared through cloud storage, and the manifest is committed.

## 7. Git Workflow and Collaboration

1. **Never commit directly to `main`.** Work on `dev` or feature branches.
2. **Branch names:** `feature/task-X.X-short-description` (e.g., `feature/task-3.2-model1-simple-cnn`), `fix/short-description`.
3. **Do not commit:** raw images (`data/raw/`), external photos (`data/external_test/*.jpg`), weights (`weights/*.pt`), virtual environments (`.venv/`), the dHash cache (`data/processed/`).
4. **Do commit:** `pyproject.toml`, `uv.lock`, `data/splits/` (CSVs, `class_mapping.json`, `split_report.json`), `data/external_test/external_manifest.csv`, and `docs/`.
5. **Share checkpoints and external photos** through cloud storage (Google Drive or OneDrive), and record the link in this README.
6. **Pull requests:** pass `uv run python tests/test_env.py`, then open the PR into `dev`. Require at least one teammate review.

## References

[1] R. Geirhos et al., "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, 2(11), 665–673, 2020.