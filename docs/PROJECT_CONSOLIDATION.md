
# ASL Fingerspelling Recognition: Project Consolidation

**Read this first in every new session.** It is the single source of truth for project status, decisions, deviations from the task assignment, reusable assets, and the working protocol. If a later document contradicts this file, this file wins until it is updated.

**Last updated:** Task 2.4 complete (report: `docs/task2_4_splits_external.md`). Next: Task 3.1.

---

## 0. How to use this document

- New sessions read Sections 1, 2, 8, 9, and 10 first.
- Section 6 defines the workflow for every task.
- Section 7 defines the report format for each `docs/task2_x_*.md` file.
- Section 9 lists every place where the project departs from the leader's task assignment. Any teammate or the professor should read it before quoting a number.
- Every number is labelled:
  - **[Measured]**: computed by our scripts on this machine or on the dataset files.
  - **[Reported]**: from the paper, the dataset page, or the leader's roadmap. Not verified by us.
  - **[Expected]**: a target or estimate, not yet tested.
  - **[Inferred]**: derived by arithmetic from measured values.

---

## 1. Project identity

- **Course:** Deep Learning midterm project. Supervisor: Dr. Nguyễn Thị Kim Ngân.
- **Goal:** real-time ASL fingerspelling recognition from a webcam, with a three-tier model progression and a live demo on a laptop CPU.
- **Model progression (midterm requirements):**
  1. Simple CNN built from convolution, pooling, and fully connected layers (no skip connections).
  2. Complex CNN built from residual blocks (Mini-ResNet), trained from scratch.
  3. Transfer learning and fine-tuning of a pretrained backbone (MobileNetV3-Small or EfficientNet-B0).
- **Required evaluation:** Accuracy, Precision (macro), Recall (macro), and F1 (macro) on the same test files for all three models.
- **Performance targets [Reported, roadmap]:** Model 3 in the rubric range; CPU inference under 15 ms per frame on the presentation laptop, which the roadmap lists as an Intel Core i5-13500H.
- **Development machine (measured):** Windows, 16 logical CPUs reported by Python, no CUDA GPU, PyTorch 2.14.1+cpu. The CPU model and physical core count were not recorded in our logs.
- **Team and roles:** *[Member table and deadlines from the roadmap. To be entered by the team.]*
- **Environment:** Python 3.10 managed with `uv`. PyTorch, torchvision, MediaPipe, OpenCV, scikit-learn, pandas, matplotlib, tqdm, pytest. Heavy training on Kaggle T4/P100 or the RX 6700 XT machine (Linux, ROCm).

---

## 2. Dataset and current decisions

### 2.1 Dataset comparison

| Attribute | ASL Alphabet, grassknoted (archived) | ASL-HG, Mendeley (current) |
|---|---|---|
| Images | 87,000 | **36,000 files [Measured]** |
| Classes | 29 (A–Z, del, nothing, space) | **36 (0–9, A–Z) [Measured]** |
| Images per class | 3,000 | **1,000 files; 893–1,000 distinct contents [Measured]** |
| Signers | 1 | **10 (P1–P10), 3,600 files each [Measured]** |
| Image size and mode | 200×200 RGB | **300×300 RGB [Measured]** |
| Byte-identical copies | not measured | **435 groups, 994 files; 199 cross-signer groups, all P5↔P6, all in classes I and V [Measured]** |
| Hands per image | not measured | **One hand for every class except 0, which is two-handed [Measured by inspection of a sample]** |

### 2.2 Current status of key decisions

- **D2 (active):** the main dataset is ASL-HG.
- **D7 (resolved):** 36 classes. The midterm text specifies classification but no class count. See Section 8.
- **D10 (resolved, revised):** exact-copy de-duplication in the training split only. See Section 8.
- **Raw data:** `data/raw/` holds the 36 class folders directly. The Mendeley `processed/` folder is never read.

---

## 3. Evidence from completed tasks

Full analyses: `docs/task2_1_eda_analysis.md`, `docs/task2_2_roi_cropping.md`, `docs/task2_3_dataset_pipeline.md`, `docs/task2_4_splits_external.md`.

### Task 2.1: dataset integrity and signer structure

- **Structure [Measured]:** 36 class folders, 1,000 files each, 100 per signer per class. All decode. All 300×300 RGB. No stray files. The filename class equals the folder class everywhere.
- **Subject split [Measured]:** zero exact-copy groups cross train, validation, or test. Zero validation or test images are within dHash distance 1 of a training image.
- **Random 70/15/15 split [Measured]:** 214 of 435 copy groups cross splits. About 86% of validation and test images are within dHash distance 1 of a training image.
- **Signer lighting [Measured]:** signer mean luminance ranges from 160 (P4, P7) to 202 (P3). Within-signer standard deviation is 5–8 for most signers, and 27 for P3.
- **Burst structure [Measured]:** consecutive same-signer median Hamming distance is 1 (mean 2.8). Random same-class, cross-signer median is 30 (1st percentile 14). The roadmap's "~0.75" consecutive figure was **not** reproduced.
- **Camera [Reported, conflicting]:** the paper's §4.1 says all images came from one OPPO Reno5. Its §4 and the dataset page describe ten volunteers with smartphone cameras. Unresolved.

### Task 2.2: hand detection, crop, and letterbox

- **Output [Measured]:** 35,988 PNG crops at 224×224, 20% margin (D15), letterboxed with (128, 128, 128). Errors: 0.
- **Method split [Measured]:** 34,980 MediaPipe single-hand boxes; 1,000 class-0 full-frame by rule (D19); 8 undetected non-zero full-frame (D20); 12 blank frames excluded (D18).
- **Blank frames [Measured]:** 12, all H/P8. Threshold on edge fraction $e_{40} \le 0.001$, with a 369× gap between the 12th and 13th lowest values.
- **Letterbox scale [Measured]:** per-class median scale $224 / \text{margin side}$ ranges from **0.747 (class 0)** to **1.325 (class M)**. Spread is about **1.77×**. *(Correction: the earlier 0.83–1.33 figure excluded class 0.)*
- **Class 0 [Measured]:** median scale 0.747 against 0.941 for the other classes (ratio 0.79). This is the largest single-class deviation.
- **Partial-entry frames [Measured]:** about 13 non-blank H/P8 frames show only a fingertip or knuckles at the edge. They are kept and flagged.
- **Validation gates [Measured]:** all PASS, including 0 padding errors and 0 non-224×224 files.

### Task 2.3: dataset, augmentation, and loader

- **Manifests [Measured]:** written crops 35,988. Train 25,200 before de-duplication, 24,777 after removing 423 exact copies. Validation 3,588. Test 7,200.
- **Train class counts [Measured]:** 595 (I) to 700 (F). Class weight ratio 1.176.
- **Augmentation [Measured]:** scale range covers the observed spread. Padding restored exactly after colour jitter. No flip.
- **Throughput [Measured]:** six sessions with the final code, training median 322–605 img/s, five of six above the 400 img/s gate. Details in Section 3 of the Task 2.3 report.
- **Profile [Measured]:** per image, PNG decode 5.8 ms (44%) and padding-restoring colour jitter 6.0 ms (46%). Affine 0.4 ms. Normalise 0.9 ms.
- **Tests [Measured]:** 17 of 17 pass on the data.

### Task 2.4: splits, random baseline, and external benchmark

- **Subject split on disk [Measured]:** immutable, checksummed CSVs: `subject_train.csv` (24,777), `subject_val.csv` (3,588), `subject_test.csv` (7,200). 423 exact duplicates excluded and logged in `excluded_train_copies.csv`. Zero signer leakage across splits.
- **Random split baseline [Measured]:** immutable, checksummed CSVs over the identical 35,988 crop universe: `random_train.csv` (25,190), `random_val.csv` (5,399), `random_test.csv` (5,399). Max class deviation from 15% is 0.80 images.
- **Frozen random leakage [Measured]:** 113 validation rows (2.1%) and 118 test rows (2.2%) share an exact MD5 with train; 221 of 435 copy groups (50.8%) cross splits. Replaces the Task 2.1 estimate for Ablation 1.
- **External benchmark set [Measured]:** 180 total photos (36 classes $\times$ 5) collected by team members (`cbp`, `hck`, `ntt`, `phm`). Exactly 4 original photos and 1 horizontally mirrored photo (`hck_*_02.jpg`) per class.
- **Mirror verification [Measured]:** exactly 36 mirror pairs verified by 64-bit dHash pixel comparison (`d_flip = 0`, `d_orig = 36–58`). Zero unpaired or spurious mirrors.
- **External normalisation & cropping [Measured]:** all 180 photos normalised to a 300×300 square (D37) and cropped via `preprocess_roi.process_one` to 224×224 letterbox with neutral gray. Zero byte-identical matches with Mendeley training data.
- **External domain scale gap [Measured]:** 87.8% of external crops fall inside the training reference scale range `[0.747, 1.589]`. Camera-distance variance in unconstrained webcam captures causes ~6× more tail outliers than training's own internal distribution. Validated against an 85% domain gate (D40).
- **Manifest [Measured]:** `data/external_test/manifest.csv` written and verified (180 rows, 7 columns, git-tracked). Headline metrics evaluate 144 originals only (D39).

---

## 4. Task status

Status key: **Done**, **Next**, **Open**.

| Task | Description | Status | Notes |
|---|---|---|---|
| 1.1 | Repository, layout, environment | Done | Locked pyproject.toml and smoke test verified. |
| 2.1 | Dataset integrity and EDA | **Done** | Report: `docs/task2_1_eda_analysis.md`. |
| 2.2 | Hand ROI, crop, letterbox | **Done** | Report: `docs/task2_2_roi_cropping.md`. |
| 2.3 | Dataset, augmentation, DataLoader | **Done** | Report: `docs/task2_3_dataset_pipeline.md`. |
| 2.4 | Subject split CSVs, random split CSVs, external test set, verification | **Done** | Report: `docs/task2_4_splits_external.md`. Subject (24,777/3,588/7,200), random (25,190/5,399/5,399), external (180: 144 orig + 36 mirr). |
| 3.1 | Unified training engine (`src/train.py`) | **Next** | Ingest frozen manifests, support AMP, Label Smoothing ($\alpha=0.1$), AdamW/SGD, Cosine Annealing, checkpointing. |
| 3.2 | Model 1 (simple CNN) | Open | Sequential ConvNet from scratch without skip connections (<1.5M params). |
| 3.3 | Model 2 (Mini-ResNet) | Open | Complex CNN from scratch with $\mathcal{F}(x) + x$ residual skip highways. |
| 3.4 | Model 3 (transfer learning) | Open | Pretrained MobileNetV3-Small / EfficientNet-B0 fine-tuned on 36 classes. |
| 3.5 | Ablations | Open | Ablation 1 (random vs subject split), Ablation 2 (letterbox vs squash), Ablation 3 (regularisation/label smoothing), scale ablation (D21). |
| 4.1 | Test benchmarking and evaluation table | Open | Macro Acc/Prec/Rec/F1. Headline 144 originals vs all-180 external; stratify by scale gap (D40). |
| 4.2 | Confusion matrices and error analysis | Open | Qualitative analysis on ambiguous pairs. |
| 4.3 | Grad-CAM | Open | Spatial attribution on final conv layer hooks. |
| 5.1 | Debounce filter and word assembly | Open | 10-frame sliding window debounce buffer. |
| 5.2 | Model optimisation for CPU | Open | TorchScript tracing and thread locking on laptop CPU. |
| 5.3 | Dashboard and offline fallback | Open | Streamlit / Gradio live demo interface. |
| 6.1–6.3 | Report Parts 1–2, slides, rehearsal | Open | Written theory, experimental results, and slide deck. |

---

## 5. Reusable assets and code status

| File | Status | Notes |
|---|---|---|
| `pyproject.toml` | Done | Locked dependencies; `protobuf>=4.25.3` and `mediapipe==0.10.14`. |
| `uv.lock` | Done | Locked and synced. |
| `.gitignore` | Done | Tracks split CSVs and `data/external_test/manifest.csv`; untracks raw/processed images and weights. |
| `README.md` | Rewrite pending | Update to reflect ASL-HG 36-class pipeline. |
| `src/config.py` | Done | Global constants, paths, class maps (0–35), `CLASSES`, `CLASS_TO_ID`. |
| `src/roi_detector.py` | Done | Core MediaPipe detection engine with YCrCb Otsu fallback. |
| `src/verify_dataset.py` | Done | Task 2.1 integrity audit. |
| `src/audit_signers.py` | Done | Task 2.1 signer index cache. |
| `src/audit_cross_split.py` | Done | Task 2.1 cross-split leakage audit. |
| `src/audit_random_split.py` | Done | Task 2.1 random split analysis. |
| `src/eda_figures.py` | Done | Task 2.1 visual artifacts generator. |
| `src/hand_roi_audit.py` | Done | Task 2.2 detection feasibility audit. |
| `src/blank_scan.py` | Done | Task 2.2 blank frame detection engine. |
| `src/preprocess_roi.py` | Done | Task 2.2 production cropping & letterbox padding pipeline. |
| `src/verify_crops.py` | Done | Task 2.2 crop verification audit. |
| `src/manifests.py` | Done | Task 2.3 in-memory manifest builder with train de-duplication. |
| `src/dataset.py` | Done | Task 2.3 `KeepPadJitter`, `build_transform`, `ASLDataset`, `class_weights`. |
| `src/dataloaders.py` | Done | Task 2.3 `get_dataloaders` factory with `DEFAULT_WORKERS = 8`. |
| `src/benchmark_loader.py` | Done | Task 2.3 throughput profiler. |
| `src/profile_transforms.py` | Done | Task 2.3 transform profiling diagnostic. |
| `src/freeze_splits.py` | Done | Task 2.4 Part A. Freezes subject split; write-once, write-and-verify. |
| `src/freeze_random_splits.py` | Done | Task 2.4 Part B. Freezes random split for Ablation 1; measures leakage. |
| `src/setup_external_test.py` | Done | Task 2.4 Part C. Provisions 36 class directories and performs photo census. |
| `src/prepare_external.py` | Done | Task 2.4 Part C. Normalises to 300x300, crops via Task 2.2, checks 15 gates, generates `manifest.csv`. |
| `src/diagnose_external_scale.py` | Done | Task 2.4 Part C diagnostic for camera-distance domain gap. |
| `tests/conftest.py` | Done | Puts `src/` on import path. |
| `tests/test_dataset.py` | Done | 17 unit tests, all passing. |
| `tests/test_env.py` | Done | Environment smoke test. |
| `data/splits/` | Frozen | `subject_*.csv`, `random_*.csv`, `class_mapping.json`, `excluded_train_copies.csv`, `SHA256SUMS*.txt`. |
| `data/external_test/` | Done | 180 photos, `manifest.csv` (git-tracked, 180 rows, 7 columns). |

**Folder structure:**

```
data/raw/                          ASL-HG class folders 0-9, A-Z (never modified)
data/processed/                    caches and logs: index, blank scan, crop log, external crop log
data/processed_crops/              Task 2.2 output: 35,988 letterboxed 224x224 PNG crops
data/splits/                       Task 2.4 output: subject and random CSVs, class_mapping.json,
                                   excluded_train_copies.csv, SHA256SUMS.txt
data/external_test/                Task 2.4 Part C: class folders (photos untracked) and manifest.csv (tracked)
data/external_processed/           Task 2.4 Part C: 300x300 normalised intermediate frames
data/external_crops/               Task 2.4 Part C: 224x224 letterboxed crops of external photos
docs/                              reports (task2_*), JSON evidence, figures/
```

---

## 6. Working protocol (applies to every task)

1. **Break the task into 2–3 parts.** Name each part, state its objective and output.
2. **One part at a time.** Each part includes code, the exact command, and the output to paste back.
3. **The student runs the code and pastes the full printed output**, including errors.
4. **Analyse the output and check the logic** before moving on. Fix code, not only prose.
5. **After all parts are done**, write `docs/task2_x_<short_name>.md` using the Section 7 template.
6. **Update this file:** status (Section 4), decisions (Section 8), deviations (Section 9), open items (Section 10).

**Rules:**
- Never report an unmeasured number. Label it **[Reported]** or **[Expected]**.
- State limitations next to every result.
- Exact-copy identity uses MD5. dHash is used only for near-duplicate measurement.
- Raw data is never modified.
- Split CSVs are immutable once written. Regenerating them requires a logged decision in Section 8.
- Every model is evaluated on the same test files.

---

## 7. Report format (`docs/task2_x_*.md`)

```
# Task 2.x: <Name>

**Status / Date / Consolidation reference / Scripts**

Labels used throughout.

## Summary
## Requirement check against the task assignment   (what was asked, what was met, deviations)
## Part A: <name>
### Relevance (deep learning)
### Definitions and formulas
### Objective
### Methodology
### Code output analysis
### Conclusion
## Part B: ...
## Limitations
## Decisions made in this task   (Decision, Reason, What it means for teammates, Status)
## Files produced               (File, Role, Used by, Notes)
## Hand-off to the next task
## Notes for teammates
## Reproduction
```

---

## 8. Decision log

| # | Session | Decision | Reason | Status |
|---|---|---|---|---|
| D1 | 1 | Grassknoted split at t = 0 | Matched rubric counts | Superseded by D2 |
| D2 | 1 | Main dataset is ASL-HG | Grassknoted random-split accuracy inflated by near-duplicates | Active |
| D3 | 1 | Grassknoted kept as archived baseline | Shows the split effect | Active |
| D4 | 1 | External test size 3 per class (87) | Replaced by D16 | Superseded |
| D5 | 1 | Git tracks the external manifest, not the photos | Manifest is shared; photos are large | Active; file name is `manifest.csv` |
| D6 | 1 | Signer-disjoint splitting is primary | Measures generalisation across people | Active; detailed in D9 |
| D7 | 1 | **36 classes (0–9, A–Z). Resolved.** | The midterm text specifies classification without a class count. The 36 classes match the dataset. | **Resolved (Session 2)** |
| D8 | 2 | Class order `0`–`9` then `A`–`Z`; `class_id` 0–35 | Matches the roadmap; digit 0 is the two-handed sign, distinct from letter O | Active |
| D9 | 2 | Subject split: train P1–P7, validation P8, test P9–P10 | Zero exact-copy crossings; zero validation or test images within distance 1 of train | Active |
| D10 | 2 | **Exact-copy de-duplication in the training split only, keeping the first file per MD5 (423 removed). Validation and test are not de-duplicated.** | Duplicates in train would give I and V extra gradient weight without new information. Validation and test keep full crop output. | **Resolved (Session 2); deviation in counts (Section 9)** |
| D11 | 2 | Random stratified 70/15/15 (seed 42) used only for Ablation 1 | 221 of 435 copy groups cross splits | Active |
| D12 | 2 | MD5 for exact copies; dHash for near-duplicates only | dHash collides (35,441 MD5 vs 17,382 dHash values) | Active |
| D13 | 2 | 36-class grid rotates signers | Previous grid showed one signer | Done |
| D14 | 2 | Crops in `data/processed_crops/`; caches in `data/processed/` | Keeps outputs separate from caches | Active |
| D15 | 2 | Hand-box margin: 20% of the box's longer side | Within the roadmap range; no clipped fingertips in review | Active |
| D16 | 2 | External real-world set: 2–3 photos per class across 36 classes (72–108) | Roadmap value; replaced by D38 | Superseded by D38 |
| D17 | 2 | Crops are lossless PNG with the original file stem | JPEG re-encoding breaks exact padding checks | Active |
| D18 | 2 | 12 blank frames excluded (H/P8, $e_{40} \le 0.001$) | Clear gap in ranking; visually blank | Active |
| D19 | 2 | Class 0 uses the full frame by fixed rule (`class_rule_full`) | Single-hand box clips the second hand; no two-hand logic | Active; scale limitation recorded |
| D20 | 2 | No centre-crop fallback. Undetected non-zero frames keep full frame (`undetected_full`, 8 images) | Centre crop invents a box with no hand | Active |
| D21 | 2 | Letterbox scale is a nuisance variable across signs (0.747–1.325, ~1.77×). Augmentation weakens it; Task 3.5 ablates it. | Augmentation is class-independent, so it cannot remove class-dependent size difference. | Active; Task 3.5 requirement |
| D22 | 3 | Augmentation scale range $[0.75, 1.35]$ | Covers zoom needed to bring every class to unit scale (0.754–1.339) | Active |
| D23 | 3 | Augmentation order: `KeepPadJitter`, then `RandomAffine`, then `ToTensor` and normalise | Colour jitter must not alter padding or rotation fill | Active |
| D24 | 3 | Normalisation uses ImageNet mean and standard deviation | Matches the pretrained backbone in Model 3 | Active |
| D25 | 3 | No horizontal or vertical flip | Mirrored signs change meaning (roadmap) | Active |
| D26 | 3 | 8 DataLoader workers, fixed | Best measured setting on 16-CPU laptop; re-measure on training machine | Active; **re-gate on training machine** |
| D27 | 3 | One intra-op thread per worker | Avoids oversubscription on a 16-CPU machine | Active |
| D28 | 3 | No uint8 cache now | Gate passes in five of six sessions; cache would add ~5.3 GB | Deferred; first option if gate fails on GPU machine |
| D29 | 3 | Factory `get_dataloaders` lives in `src/dataloaders.py`, not `dataset.py` | Keeps dataset free of loader policy; import path documented | Active; **deviation from leader text** |
| D30 | 3 | Default `num_workers` is 8, not leader's 4 | Four workers failed throughput gate on this laptop | Active; **deviation** |
| D31 | 3 | `pin_memory = torch.cuda.is_available()` | True on CUDA machine (leader's setting); avoids warning on CPU-only machines | Active; **deviation in form** |
| D32 | 3 | Rotation (±12°) is inside one `RandomAffine` call, not separate `RandomRotation` | Same uniform distribution; one geometric transform means one interpolation | Active; **deviation in form** |
| D33 | 3 | Colour jitter wrapped as `KeepPadJitter`, restoring padding pixels exactly | Plain `ColorJitter` changes letterbox bars, which network could read as augmentation cue | Active; **deviation in form** |
| D34 | 3 | Throughput benchmark is a separate script, `benchmark_loader.py` | Benchmarks depend on machine state and should not gate unit tests | Active; **deviation in form** |
| D35 | 3 | Train 24,777 and validation 3,588, not 25,200 and 3,600 | Follows D10 and D18 | Active; **deviation in counts** |
| D36 | 3 | Throughput gate uses median of 5 timed trials | One slow trial or session should not decide gate; machine state logged | Active |
| D37 | Task 2.4 | External photos normalised to a square 300×300 frame before detection | Keeps letterbox scale directly comparable to Mendeley capture frame (300×300) without modifying validated Task 2.2 detector code. | Active |
| D38 | Task 2.4 | External test set: 36 classes $\times$ 5 photos = 180 total (exactly 4 original + 1 mirrored per class). Mirror pairs verified by pixel dHash. | Gives a steadier estimate than 72–108 photos. Mirrored photos represent plausible left-handed captures for single-hand static signs. | Active; **deviation from task text (X12)** |
| D39 | Task 2.4 | Headline external metrics use 144 originals only; all-180 reported second. | Mirrored photo shares its source's hand, background, and lighting — not an independent sample. | Active |
| D40 | Task 2.4 | External domain-scale gate set to 85% (measured: 87.8% inside training p1–p99 range `[0.747, 1.589]`). | Real-world camera distance variance causes an expected domain gap. Task 4.1 must report accuracy stratified by scale. | Active; **documented domain finding** |

---

## 9. Deviations from the task assignment

This table is for the professor and teammates. Each row says what was asked, what we did, and why.

| # | Task assignment says | We did | Why | Impact |
|---|---|---|---|---|
| X1 | `dataset.py` contains DataLoader factory | Factory in `dataloaders.py` (D29) | Separates dataset from loader policy | Import path: `from dataloaders import get_dataloaders` |
| X2 | `num_workers` default 4 | Default 8 (D30) | Four workers failed gate on this laptop | Teammates on another machine must check core count |
| X3 | `pin_memory=True` | `torch.cuda.is_available()` (D31) | Avoids warning on CPU-only machines | Identical behaviour on a GPU machine |
| X4 | `RandomRotation(degrees=12)` and `RandomAffine(degrees=0, scale=(0.94, 1.06))` | One `RandomAffine(degrees=12, scale=(0.75, 1.35))` (D32, D22) | Scale range widened for measured spread (D21) | Rotation distribution unchanged; scale is wider than roadmap |
| X5 | Plain `ColorJitter` | `KeepPadJitter` wrapping `ColorJitter` (D33) | Padding must stay at exactly (128, 128, 128) | Natural pixels of exactly (128, 128, 128) are not jittered |
| X6 | Throughput tests in `tests/test_dataset.py` | `benchmark_loader.py` (D34) | Machine-dependent timing does not belong in unit tests | Throughput is reported separately |
| X7 | Train 25,200 (700 per class), validation 3,600 (100 per class) | Train 24,777 (595–700), validation 3,588 (H = 88) (D10, D18, D35) | Duplicate and blank removal | Per-class counts not equal; class weights used (ratio 1.18) |
| X8 | Random split over all 36,000 images | Random split over the 35,988 written crops (D18) | Blank frames are not valid samples | Random split counts: 25,190 / 5,399 / 5,399 |
| X9 | Split CSV columns `filepath, class_name, class_id, volunteer_id` | Same four columns plus `crop_path`, `md5` | Loader needs crop path; audit needs hash | Extra columns allowed; required columns unchanged |
| X10 | External manifest `data/external_test/manifest.csv` | Same name. 180 rows, 7 columns | Matches leader name; manual metadata spreadsheet bypassed | `.gitignore` tracks `manifest.csv` |
| X11 | Throughput above 400 img/s "on local CPU/GPU" | Five of six sessions pass on this laptop (D36) | Local CPU is allowed by assignment | Re-measure on training machine |
| X12 | 72–108 external photos, 2–3 per class, original photos | 180 photos, 5 per class (4 original, 1 mirrored) (D38, D39) | Steadier estimate; mirror pairs verified by pixel dHash | Headline set: 144 originals. Report both numbers in Task 4.1. |

---

## 10. Open items

**Resolved in Tasks 2.1–2.4:**
- ~~Class set~~ → 36 classes (D7).
- ~~Signer identity~~ → P1–P10 in filenames (Task 2.1).
- ~~Image size and colour~~ → 300×300 RGB (Task 2.1).
- ~~Random vs subject leakage~~ → measured on frozen sets (Task 2.4 Part B).
- ~~Blank frames~~ → 12 found and excluded (Task 2.2 B-1).
- ~~Class 0 cropping~~ → full-frame rule (D19).
- ~~Letterbox correctness~~ → 0 failures (Task 2.2 Part C).
- ~~Train de-duplication~~ → train only (D10).
- ~~Padding survives augmentation~~ → `KeepPadJitter` (D33). Test passes.
- ~~Split CSVs frozen on disk~~ → `subject_*.csv` and `random_*.csv` (Task 2.4 Part A & B).
- ~~External benchmark collection~~ → 180 photos processed, manifest generated (Task 2.4 Part C).
- ~~Resolution normalisation for external photos~~ → 300×300 normalisation before detection (D37).

**Still open:**
1. **Visual confirmation of `docs/figures/aug_check_part_a.png`.** The letterbox bars and the rotation corners should be the same grey in the augmented rows. Confirm before submission.
2. **Throughput on the training machine.** Re-gate with `benchmark_loader.py` in Task 3.1. Record core count.
3. **Camera source conflict (Reported).** Paper §4.1 (one device) vs §4 (ten volunteers). Raise with dataset authors.
4. **Accuracy reproduction (Task 3.4).** Reproduce the leader's reported 90.39% (subject) and 96–98% (random) with a script, seed, and config.
5. **Augmentation-off ablation (Task 3.5).** Test the claim that augmentation is needed for 85%+ generalisation.
6. **Scale ablation (Task 3.5, D21).** Required to separate hand-size shortcut from signal.
7. **Random seed for augmentation (Task 3.1).** Ensure `torch.manual_seed` is set before training.
8. **Background clustering.** Test whether background correlates with signer.
9. **Finer near-duplicate measure.** Confirm the 30 test images within dHash distance 5 with CNN embedding (Task 4.1).
10. **Team roles and deadlines.** Enter the member table in Section 1.
11. **Licence and citation.** Check Mendeley licence; cite paper and DOI.
12. **README rewrite.** Update repository README to reflect ASL-HG pipeline.
13. **Per-signer reporting (Task 4.1).** Test has two signers (P9, P10), validation has one (P8). Report per signer.
14. **Scale-stratified external evaluation (Task 4.1, D40).** Report external test accuracy stratified by whether each photo's letterbox scale fell inside or outside `[0.747, 1.589]`.
15. **Headline vs all-photo external reporting (Task 4.1, D39).** Report headline accuracy on the 144 originals and secondary accuracy on the full 180 photos.

---

## 11. Next: Task 3.1 plan (Unified PyTorch Training Engine & Checkpoint Manager)

With all data splits, augmentations, DataLoaders, and external benchmark sets frozen and verified, Phase 3 (Model Implementation & Training) can begin.

### Objective
Build a modular, reusable training and validation harness in `src/train.py` that can train Model 1 (Simple CNN), Model 2 (Mini-ResNet), and Model 3 (MobileNetV3-Small) through an identical interface.

### Key Components to Implement
1. **Data Ingestion:** Load frozen CSV splits via `dataloaders.get_dataloaders('data/splits/subject_train.csv', 'data/splits/subject_val.csv', 'data/splits/subject_test.csv', batch_size=64)`.
2. **Loss Function with Label Smoothing:** Implement `CrossEntropyLoss(label_smoothing=0.1)` to mitigate overconfident memorization on training signers P1–P7.
3. **Optimizers & Schedulers:** Support AdamW and SGD with momentum, coupled with Cosine Annealing learning rate scheduling (`torch.optim.lr_scheduler.CosineAnnealingLR`).
4. **Hardware Acceleration & Efficiency:**
   - Automatic Mixed Precision (`torch.amp.autocast`) to maximize throughput on CUDA/ROCm GPUs.
   - Non-blocking host-to-device memory transfers (`inputs.to(device, non_blocking=True)`).
5. **Automated Checkpointing & Logging:**
   - Save `weights/best_model.pt` based on minimum validation loss or maximum validation accuracy.
   - Stream epoch-by-epoch metrics (`epoch`, `train_loss`, `train_acc`, `val_loss`, `val_acc`, `lr`) to a structured CSV / JSONL log for instant plotting in Task 6.2.

---

*Update the date and status tables after each task.*