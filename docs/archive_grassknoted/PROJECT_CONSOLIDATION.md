# ASL Fingerspelling Recognition: Project Consolidation

**Read this first in every new session.** It is the single source of truth for project status, decisions, reusable assets, and the working protocol. If a later document contradicts this file, this file wins until it is updated.

---

## 0. How to use this document

- New sessions start by reading Sections 1, 2, 9, and 10.
- Section 6 defines the workflow for every task.
- Section 7 defines the report format for every `docs/task2_x_*.md` file.
- Update Sections 4, 5, and 8 whenever a task finishes or a decision changes.

---

## 1. Project identity

- **Course:** Deep Learning midterm project. Supervisor: Dr. Nguyễn Thị Kim Ngân.
- **Goal:** real-time ASL fingerspelling recognition from a webcam, with a three-tier model progression and a live demo on a laptop CPU.
- **Model progression (rubric):**
  - Model 1: simple sequential CNN, from scratch, no skip connections.
  - Model 2: residual (Mini-ResNet) CNN, from scratch.
  - Model 3: pretrained MobileNetV3-Small (or EfficientNet-B0) with transfer learning.
- **Performance targets:** Model 3 accuracy in the rubric range; CPU inference under 15 ms per frame on an Intel Core i5-13500H.
- **Team and roles:** *[paste the member table and deadlines from the roadmap file]*
- **Environment:** Windows (primary). Python 3.10 or 3.11 managed with `uv`. PyTorch, torchvision, MediaPipe, OpenCV, Albumentations, scikit-learn, Gradio or Streamlit. Heavy training on Kaggle T4/P100 or on the RX 6700 XT machine (Linux, ROCm).

---

## 2. Dataset history and current decision

| Attribute | ASL Alphabet, grassknoted (archived) | ASL-HG, Mendeley (current) |
|---|---|---|
| Source | kaggle.com/datasets/grassknoted/asl-alphabet | data.mendeley.com/datasets/j4y5w2c8w9/1 (paper: sciencedirect.com/science/article/pii/S2352340926000454) |
| Images | 87,000 | 36,000 (leader's figure; **verify**) |
| Classes | 29 (A–Z, del, nothing, space) | 36 (A–Z, 0–9) (leader's figure; **verify**) |
| Images per class | 3,000 (verified) | 1,000 implied by 36,000 / 36; leader also cites ~113 "unique" per class. **Reconcile.** |
| Signers | 1 | 10 (leader's figure; **verify**) |
| Backgrounds | 1 | at least 10 (leader's figure; **verify**) |
| Near-duplicate frames | very high (we measured this) | still substantial (leader's observation; **measure**) |
| Role in project | archived baseline showing leakage | main dataset |

**Decision (Session 1):** switch to ASL-HG.

**Reason:** grassknoted's random-split test accuracy (98–100%) mostly reflects near-duplicate frames from a single signer. It does not measure generalization. On ASL-HG, a signer-disjoint split (signers 1–7 train, 8 validation, 9–10 test) gave 90.39% with transfer learning, which is the honest estimate. A random mix of ASL-HG gave 96–98%, which is inflated by the same leakage.

**Open decision, must be confirmed with Dr. Ngân before Task 2.1 is final:**

- The class set changes from 29 to 36. Rubric items that assume 29 classes, and the quotas 60,900 / 13,050 / 13,050 (2,100 / 450 / 450 per class), no longer apply as written.
- Ask whether the rubric accepts the new class set and how to report split counts. Record the answer in Section 8.

---

## 3. Evidence collected so far

### 3.1 Grassknoted (measured by us; archived)

- 40,173 of 87,000 images have an exact twin (identical 64-bit dHash) in the same class, which is 46%.
- 76% of images have a same-class neighbour at Hamming distance ≤ 1.
- Median distance between consecutive filenames is 1. Median distance between random pairs in the same class is 27. Filename order follows recording sessions.
- A random split leaves 40% of held-out images with a training neighbour at distance ≤ 1. Exact-hash leakage is zero by construction.
- At threshold t = 1, the largest near-duplicate group is 489 images (class `nothing`). Grouping cannot satisfy the 450-image val/test quota above t = 0.

### 3.2 ASL-HG (reported by leader; not yet verified by us)

- Random mix: 96–98% test accuracy.
- Signer-disjoint split (train 1–7, val 8, test 9–10), transfer model: 90.39% test accuracy.

### 3.3 Interpretation

The gap between random and signer-disjoint accuracy measures how much a random split leaks. Report both numbers in Part 2, Section IV. The grassknoted results, which show the largest gap, are the strongest argument for the signer-disjoint protocol.

---

## 4. Task status

Status key: **Done** (reusable as is), **Reuse** (keep logic, adapt), **Redo** (must be redone for ASL-HG), **Open** (not started).

| Task | Description | Status | Notes |
|---|---|---|---|
| 1.1 | Repository, layout, environment | Done | Keep. Verify `pyproject.toml` (see Section 5). |
| 2.1 | Dataset integrity audit and EDA | Redo | Grassknoted results archived. Integrity, class counts, signer and background audit must be redone on ASL-HG. |
| 2.2 | Hand ROI cropping (MediaPipe + fallback, masks precomputed) | Open | Logic is dataset-agnostic. Check the ASL-HG image size first. |
| 2.3 | In-memory background re-compositing, DataLoader | Open | Background bank must come only from training signers' scenes or external images, never from test signers. |
| 2.4 | Stratified split and external real-world set | Redo | Replace the grouped random split with a signer-disjoint split. Keep the leakage audit as a within-signer check. Re-evaluate the external set (see Section 9). |
| 3.1 | Unified training engine | Open | `src/train.py`, dataset-agnostic. |
| 3.2 | Model 1 | Open | |
| 3.3 | Model 2 | Open | |
| 3.4 | Model 3 (transfer) | Open | The leader's 90.39% run needs a reproducible script and recorded seed. |
| 3.5 | Ablations | Open | Include background swap and dropout ablations. |
| 4.1 | Test benchmarking, evaluation table | Open | Report per-signer accuracy as well as the aggregate. |
| 4.2 | Confusion matrices, error analysis | Open | |
| 4.3 | Grad-CAM | Open | |
| 5.1 | Debounce filter and word assembly | Open | |
| 5.2 | Model optimization for CPU | Open | |
| 5.3 | Dashboard and offline fallback | Open | |
| 6.1–6.3 | Report Parts 1–2, slides, rehearsal | Open | |

---

## 5. Reusable assets

| File | Status | Change needed |
|---|---|---|
| `pyproject.toml` | Reuse | Add `[tool.uv] package = false` and remove the `[build-system]` and `[tool.hatch...]` sections, since this is a script project, not a library. Keep the `protobuf` pin unless `test_env.py` shows it is no longer needed. |
| `uv.lock` | Reuse | Regenerate after the `pyproject.toml` change. Commit it. |
| `.gitignore` | Reuse | Keep `data/raw/*` ignored and the manifest tracked. Ignore `data/processed/`. Keep split CSVs tracked. Add the new dataset's folder names if they differ. |
| `README.md` | Rewrite dataset sections | Replace the grassknoted dataset description with ASL-HG. Update the status table and class counts. Keep the layout and setup sections. |
| `src/verify_dataset.py` | Reuse, parameterize | Hard-codes 29 classes, 3,000 per class, and 87,000 total. Make these come from one config. Keep the full-decode logic (`Image.load()`). |
| `src/split_data.py` | Replace | The grouped random split is superseded. Keep the indexing and hashing functions. Write a signer-disjoint split (see Section 10). |
| `src/audit_leakage.py` | Reuse, extend | Keep the nearest-neighbour logic. Add a signer-level check: no signer appears in more than one split. |
| `src/setup_external_test.py` | Reuse | Keep the manifest logic. Change the class list. Add a `signer` or `source` field to the manifest. |
| `tests/test_env.py` | Keep | Smoke test. |
| `notebooks/01_dataset_eda.ipynb` | Rewrite | Use the old cells as templates. Fix the known issues in the grassknoted write-up (unsorted globs, overstated captions, stale numbers). Do not reuse the old figures. |
| `docs/task2_1_eda_analysis.md` | Archive | Move to `docs/archive_grassknoted/`. |
| `docs/task2_4_stratifiedsplit_benchmark.md` | Archive | Move to `docs/archive_grassknoted/`. |
| `docs/task2_1_integrity_report.json`, `docs/task2_4_leakage_report.json`, `data/splits/split_report.json` | Archive | Move to `docs/archive_grassknoted/`. These are evidence for the baseline comparison. |
| `data/splits/*.csv`, `class_mapping.json` | Archive, then regenerate | Old splits are invalid for ASL-HG. Move them to `data/archive_grassknoted/`. |
| `data/processed/image_dhash.csv` | Delete | Belongs to the old dataset. |

**Recommended archive structure:**

```
docs/archive_grassknoted/     (old reports, kept as the leakage baseline)
data/archive_grassknoted/     (old split CSVs, not used for training)
```

Keeping the grassknoted results as a documented baseline turns the dataset change into a finding: the same model scores 98–100% under random splitting and is evaluated more honestly under signer-disjoint splitting. Include this in Part 2, Section IV.

---

## 6. Working protocol (applies to every task)

For each task `2.x`, `3.x`, and so on:

1. **Break the task into 2–3 parts.** Name each part, state its objective, and state what output it will produce.
2. **Assistant provides code for one part at a time.** Each part includes the code, the exact command to run, and the output to paste back.
3. **Student runs the code and pastes the full printed output**, not a summary or screenshot where text is possible. Include any errors in full.
4. **Assistant analyzes the output, checks logic, and fixes issues** before moving on. Corrections are made in code, not only in prose.
5. **After all parts of a task are done**, write `docs/task2_x_<short_name>.md` using the template in Section 7.
6. **Update this file:** the status table (Section 4), the decision log (Section 8), and the open questions (Section 9).

**Rules:**

- Never report a number that has not been measured. Label unverified figures as "reported" and cite the source.
- State limitations next to every result.
- Once training starts, the split CSVs are frozen. Regenerating them requires a logged decision.
- Every model is evaluated on the same test files.

---

## 7. Report format for each task (`docs/task2_x_*.md`)

Each task report has this structure. Each part gets its own subsection with the same five sub-parts.

```
# Task 2.x: <Name>

## Summary
(3–5 sentences: what was done, the main result, the main limitation)

## Part A: <name>
### Relevance (deep learning)
Why this step exists. Which failure mode it prevents or which principle it applies.
### Definitions and formulas
Key terms and equations, written as math, not only prose.
### Objective
What this part must achieve, stated measurably.
### Methodology
What the code does: inputs, parameters, seed, outputs.
### Code output analysis
Key numbers and tables pasted from the run. Interpret each one. Say what it does and does not show.
### Conclusion
One paragraph. What changed in the project because of this part.

## Part B: ...

## Limitations
## Decisions made in this task
## Files produced
```

---

## 8. Decision log

| # | Session | Decision | Reason | Status |
|---|---|---|---|---|
| D1 | 1 | Grassknoted split at t = 0 frozen and documented | Matches rubric counts; measured leakage | Superseded by D2 |
| D2 | 1 | Switch main dataset to ASL-HG | Random-split accuracy on grassknoted is inflated by near-duplicates from one signer; ASL-HG has 10 signers | Active |
| D3 | 1 | Grassknoted kept as archived baseline | Shows the effect of the split protocol | Active |
| D4 | 1 | External real-world set collected at 3 photos per class (87 total) | Rubric asks for 60–100 | Re-evaluate: with 10 signers, check whether the set is still needed for the rubric and whether it still serves the live demo |
| D5 | 1 | `.gitignore` tracks `external_manifest.csv` | Manifest must be shared; photos stay out of git | Active |
| D6 | 1 | Use signer-disjoint splitting as the primary evaluation protocol | Measures generalization across people | Active |
| D7 | 1 | *[pending]* Class set changes from 29 to 36, and the quotas are revised | Requires confirmation from Dr. Ngân | Open |

---

## 9. Open questions and risks

1. **Class set and rubric.** Confirm with Dr. Ngân that 36 classes (A–Z, 0–9) and the revised quotas are acceptable (D7).
2. **Count discrepancy.** 36,000 / 36 gives 1,000 per class, but the leader's notes say about 113 "unique" per class. Check whether the folders hold 1,000 files each and what "unique" means (deduplicated images, or distinct recording sessions). Do not state a count in the report until verified.
3. **Signer identity.** The signer-disjoint protocol needs a signer label for every image. Check whether signer IDs appear in filenames, folder structure, or a metadata file. If they do not, the protocol cannot be applied, and the paper's supplementary material must be used. This is the most important open item.
4. **Image size and background count.** Verify the resolution and the number of backgrounds. Both affect Tasks 2.2 and 2.3.
5. **Validation and test size.** Validation is one signer and the test set is two signers. Single-signer validation is noisy for model selection. Report per-signer accuracy in Task 4.1. If time allows, consider leave-one-signer-out evaluation for the final report.
6. **Reproducibility of the leader's 90.39%.** The run needs a script, a seed, and a recorded configuration before it is cited in the report.
7. **Licensing and citation.** Check the Mendeley dataset's license and cite the paper in the report.
8. **Live-demo domain shift.** Webcam conditions differ from any dataset. Keep an external real-world check, even if the rubric count changes, because the live demo depends on it.
9. **Team roles and deadlines.** Copy from the roadmap file into Section 1.

---

## 10. Current task and next steps

**Current task:** Task 2.1 for ASL-HG, followed by the signer-disjoint version of Task 2.4.

**Task 2.1 parts (proposed):**

- **Part A: Structure and integrity.** Verify the folder layout, class names, per-class counts, image size, color mode, and decode success. Confirm whether signer and background metadata exist.
- **Part B: Signer and background audit.** Compute per-signer and per-background counts. Compare brightness and near-duplicate structure across signers. Produce the EDA figures.
- **Part C: Write-up.** Create `docs/task2_1_eda_analysis.md` from the outputs of A and B.

**Task 2.4 (signer-disjoint) parts (proposed, after 2.1):**

- **Part A: Signer index.** Build a table of image path, class, signer, and background, and check that every image has a signer label.
- **Part B: Split and verify.** Assign signers 1–7 to train, 8 to validation, and 9–10 to test. Check that no signer appears in two splits, that every class appears in every split, and that the counts are as expected. Save the split CSVs.
- **Part C: Leakage and write-up.** Run the within-signer and cross-signer near-duplicate audit. Write `docs/task2_4_*.md`.

*(These parts are proposals. Confirm the signer labels in Task 2.1 Part A before finalizing 2.4.)*

---

*Last updated: Session 1. Update the date and the status tables after each task.*