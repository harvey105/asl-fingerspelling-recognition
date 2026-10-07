# Task 2.3: DataLoader and Augmentation Pipeline (ASL-HG)

**Status:** Complete for Parts A, B, and C. One visual check open (Section 10 of the consolidation).
**Date:** 2026-10-06
**Consolidation reference:** PROJECT_CONSOLIDATION.md, Sections 4, 8, 9, 10
**Scripts:** `src/manifests.py`, `src/dataset.py`, `src/check_dataset_part_a.py`, `src/dataloaders.py`, `src/benchmark_loader.py`, `src/profile_transforms.py`, `tests/conftest.py`, `tests/test_dataset.py`

Labels: **[Measured]** = computed by our scripts on this machine or on the dataset files. **[Reported]** = from the paper, dataset page, or roadmap. **[Expected]** = a target or estimate not yet tested. **[Inferred]** = derived by arithmetic from measured values.

---

## Summary

Task 2.3 turns the Task 2.2 crops into PyTorch datasets, augmentations, and loaders that Task 2.4 and the training code can use directly. It makes three things fixed for every later task.

1. **Splits.** Train is signers P1–P7 with exact-copy duplicates removed (24,777 images). Validation is signer P8 (3,588 images). Test is signers P9–P10 (7,200 images). Blank frames are excluded everywhere.
2. **Augmentation.** Rotation ±12°, translation 6%, zoom 0.75–1.35, and colour jitter 0.25. Letterbox padding and rotation fill are both exactly (128, 128, 128), so an augmented border looks the same as an unaugmented one.
3. **Loader.** Batch 64, 8 workers, one thread per worker, seeded shuffle, and `drop_last` on training. Training throughput is about 570–605 img/s in the last three sessions on this laptop. Five of six sessions pass the 400 img/s gate.

The main limitations are these. The throughput number belongs to this machine and must be re-measured on the training machine. Augmentation weakens the hand-size shortcut across signs but does not remove it (D21). Several factory defaults differ from the leader's text (Section "Requirement check"). Those differences are listed and justified below.

---

## Requirement check against the task assignment

Each leader requirement, its status, and the evidence.

| # | Requirement | Status | Evidence and note |
|---|---|---|---|
| R1 | `ASLDataset` in `src/dataset.py` | Met | Class is in `dataset.py`. |
| R2 | Ingest crops from `data/processed_crops/` using paths from split CSVs | Partial | Reads `crop_path` from a DataFrame or CSV. The CSVs are written in Task 2.4. |
| R3 | ImageNet normalisation | Met | Formula gate: max error 0. D24. |
| R4 | `is_train` toggle | Met | Tests cover both modes. |
| R5 | `RandomRotation(degrees=12)` | Met in effect | Inside `RandomAffine(degrees=12)`. D32. |
| R6 | `ColorJitter(0.25, 0.25, 0.25)` | Met with a wrapper | `KeepPadJitter`. D33. |
| R7 | `RandomAffine(degrees=0, translate=(0.06, 0.06), scale=(0.94, 1.06))` | Deviates on scale | Scale is (0.75, 1.35). D22, D21. |
| R8 | No horizontal flip | Met | Tests check train and eval. D25. |
| R9 | Factory `batch_size` default 64 | Met | |
| R10 | Factory `num_workers` default 4 | Deviates | Default 8. D26, D30. |
| R11 | `pin_memory=True` | Conditional | `torch.cuda.is_available()`. D31. |
| R12 | `persistent_workers=True` | Met | When workers > 0. |
| R13 | Throughput above 400 img/s | Met with caveat | Five of six sessions pass. Re-measure on training machine. D36. |
| R14 | `dataset.py` holds `ASLDataset` and the factory | Deviates | Factory in `dataloaders.py`. D29. |
| R15 | Tests verify batch shape (64, 3, 224, 224) and label range 0–35 | Met | 17 tests pass. |
| R16 | Tests include throughput benchmarks | Deviates | Separate script. D34. |
| R17 | Dataset decoupled from model code | Met | No model imports. |
| R18 | `get_dataloaders(train_csv, val_csv, test_csv, batch_size=64)` | Met in signature | Accepts DataFrames or paths. CSVs are written in Task 2.4. |
| R19 | Roadmap claim: augmentation is needed for 85%+ generalisation | Not verified | **[Reported].** Task 3.5 must include an augmentation-off run. |

---

## Part A: Manifests, dataset class, augmentation

### Relevance (deep learning)

The network sees exactly what the pipeline gives it. Three things can go wrong silently. A validation or test signer can leak into training. An augmentation can add a cue that never occurs at evaluation time. Or normalisation can differ from the one the pretrained backbone expects. Part A checks each one before any model is trained.

### Definitions and formulas

- **Manifest:** the list of rows (file, class, signer, split) from which a split is built. Built in memory here, frozen to CSV in Task 2.4.
- **Exact-copy de-duplication (D10):** within a group of byte-identical files (same MD5), keep one file, the first by filepath. Applied to the training split only.
- **Normalisation:**
  $$x' = \frac{x - \mu}{\sigma}, \qquad \mu = (0.485,\ 0.456,\ 0.406), \qquad \sigma = (0.229,\ 0.224,\ 0.225)$$
  These are ImageNet statistics. The backbone in Model 3 was pretrained with them, so we keep them rather than recomputing from the crops (D24).
- **Inverse-frequency class weight:** $w_c = \dfrac{N}{K\,n_c}$, with $N$ the training size, $K = 36$, and $n_c$ the class count after de-duplication. The weights sum to $N$ over the training set.
- **Scale:** letterbox resampling factor $s = 224 / \text{margin side}$ from Task 2.2. Augmentation zoom must cover $1/s$ for every class so that each sign can be seen at a common scale (D22).

### Objective

1. Build leak-free subject splits with the documented exclusions and de-duplication.
2. Make every training transform match the roadmap, with no flip and an exact border rule.
3. Verify that normalisation equals the formula and that the zoom range covers the observed spread.

### Methodology

`manifests.py` joins the cached index with the Task 2.2 log. It keeps only rows with a written crop, assigns splits by signer from `config.py`, and removes train duplicates (D10).

`dataset.py` defines the transforms. The training pipeline is `KeepPadJitter` → `RandomAffine` → `ToTensor` → `Normalize`. The evaluation pipeline is `ToTensor` → `Normalize`.

`check_dataset_part_a.py` runs four checks: manifests, tensors, scale coverage, and a figure.

### Code output analysis

**Manifest counts [Measured]:**

| Item | Value |
|---|---|
| Written crops | 35,988 (12 blanks excluded) |
| Train before de-duplication | 25,200 (P1–P7) |
| Exact copies removed from train | 423 [Measured] |
| Train after de-duplication | 24,777 |
| Validation | 3,588 (P8; class H = 88 because 12 blanks are removed) |
| Test | 7,200 (P9–P10; 200 per class exactly) |
| Train per class | min 595 (I), max 700 (F) |
| Class weight ratio (max / min) | 1.176 (weights 0.983–1.157) |

**Where the 423 removed copies come from [Inferred].** The 199 cross-signer groups (P5↔P6, classes I and V) contain 400 files, so 201 are removed from them. The remaining 222 are within-signer duplicates in other classes. Per-class removal is 105 from I and 101 from V.

**Gates [Measured, all PASS]:** no blank rows; signers match splits; splits share no signer; every crop file exists; train MD5 unique; all 36 classes in every split; batch shape (16, 3, 224, 224); labels in range; normalisation equals the formula (max error 0); no horizontal flip; scale range covers the observed spread.

**Scale coverage (D22) [Measured]:** class-median scale runs from 0.747 (class 0) to 1.325 (class M), a spread of 1.77×. Zoom needed to map each class to unit scale is 0.754–1.339. The configured range 0.75–1.35 covers it.

**Information only [Measured]:** on 512 validation crops, the channel means are (0.711, 0.613, 0.605). This is not a gate. It is well away from the ImageNet mean (0.485, 0.456, 0.406), which is expected for skin on light backgrounds. The normalisation is still applied consistently, so pretrained weights remain valid.

**Figure [Measured, visual check open]:** `docs/figures/aug_check_part_a.png` shows evaluation crops and two training augmentations for six classes.

### Conclusion

The splits are leak-free. The training transform follows the roadmap except where a recorded decision changes the zoom range (D22) or the jitter wrapper (D33). Normalisation uses the pretrained convention. Scale is handled as a weakened nuisance (D21), not a removed one.

---

## Part B: DataLoader factory and throughput

### Relevance (deep learning)

A GPU is wasted when the loader cannot keep it fed. The roadmap's 400 img/s gate is a proxy for "the GPU will not be starved." On a machine with a GPU, the real test is the training machine. On this CPU-only laptop, the gate tests whether the CPU pipeline keeps up with the augmentation cost.

### Definitions

- **Worker:** a separate process that runs the dataset transforms while the main process trains. Workers prepare upcoming batches in parallel.
- **Intra-op thread:** a thread inside one PyTorch operation. Without a limit, each worker may start a multi-threaded pool, so eight workers on 16 CPUs can compete for far more threads than cores (oversubscription).
- **Median throughput:** images per second over several timed trials. The median is the gate, so one slow trial cannot decide the result.
- **`drop_last`:** the final incomplete batch is discarded. Every training batch then has 64 images, which keeps batch statistics consistent for BatchNorm.

### Design choices

| Setting | Value | Reason | Decision |
|---|---|---|---|
| Batch size | 64 | Roadmap | Met |
| Workers | 8 | Best measured setting here | D26, D30 |
| Threads per worker | 1 | Avoids oversubscription | D27 |
| `persistent_workers` | True | Avoids restarting workers each epoch | Met |
| `pin_memory` | `torch.cuda.is_available()` | Pinned memory is for GPU transfer | D31 |
| Train shuffle | seeded generator (42) | Reproducible order | Met |
| Train `drop_last` | True | Full batches for BatchNorm | Met |

**Batches per epoch [Measured]:** training 387 (9 images are dropped each epoch; which ones depends on the shuffle), validation 57, test 113.

### Methodology

`benchmark_loader.py` runs 10 warm-up batches, then five timed trials on one persistent iterator. Each trial length is computed from the loader's real size, so the iterator cannot run out. The gate is the median training trial. Validation is reported but not gated. Each session appends a record, including the Windows power plan, to `docs/task2_3_partB_sessions.jsonl`.

`profile_transforms.py` times each stage of the training pipeline in one process over 300 seeded training crops.

### Sweep history (earlier benchmark versions)

These numbers come from earlier benchmark code with shorter windows and no thread limit. They are context, not final evidence.

| Workers | Train img/s | Note |
|---|---|---|
| 4 | 164, 255 | Below gate |
| 6 | 425, 267, 267 | Unstable |
| 8 | 622, 475 | Above gate |
| 10 | 189 | Below gate |

Decision: 8 workers, the best setting in every sweep.

### Final benchmark sessions [Measured]

Final code: 8 workers, 1 thread per worker, 5 trials, gate on the median.

| Session | Power plan | Train trials (img/s) | Train median | Validation median (info) | Gate |
|---|---|---|---|---|---|
| A | not recorded | 322.0, 286.7, 323.8, 289.8, 322.6 | 322.0 | 448.6 | **FAIL** |
| B | not recorded | 592.1, 509.9, 576.6, 529.2, 593.7 | 576.6 | 1212.5 | PASS |
| C | not recorded | 608.4, 549.3, 597.4, 545.7, 609.7 | 597.4 | 1096.7 | PASS |
| D | Balanced | 591.2, 561.6, 611.1, 577.7, 592.1 | 591.2 | 1159.6 | PASS |
| E | Balanced | 334.9, 530.6, 572.7, 592.5, 581.7 | 572.7 | 1142.6 | PASS |
| F | Balanced | 605.1, 549.0, 608.6, 549.6, 615.0 | 605.1 | 1207.2 | PASS |

**Reading the table.** Session A is the only failure, and all five of its trials are low. That points to a whole-session slowdown from machine state, not to one stall. Sessions D–F ran on the same power plan and their medians are within about 6% of each other. Sessions A–C ran before the power plan was logged, so their plan is unknown.

### Profiling [Measured]

Single process, 300 seeded training crops:

| Stage | ms per image | Share |
|---|---|---|
| PNG decode | 5.76 | 43.7% |
| `KeepPadJitter` (colour jitter with padding restore) | 6.03 | 45.8% |
| `RandomAffine` | 0.43 | 3.3% |
| `ToTensor` and normalise | 0.94 | 7.1% |
| **Total** | **13.17** | 100% |

**Consistency check [Inferred].** One process handles about 1000 / 13.17 ≈ 76 training images per second. Eight workers on a CPU-bound pipeline should approach 8 × 76 ≈ 607 img/s. Measured training medians are 573–605, which is 94–100% of that ceiling. For validation, the cost without augmentation is about 6.70 ms, so the ceiling is about 1,190 img/s. Measured validation medians are 1,143–1,213. Both are close to their ceilings, so the loader is CPU-bound and scales with workers as expected.

**Decision: no cache now (D28).** Decoding and jitter each take about 45% of the time. The gate passes with margin in five of six sessions. A uint8 cache would remove decoding only, at about 5.3 GB (train 3.7 GB, validation 0.5 GB, test 1.1 GB). That is not justified while the gate holds. It is the first option if the training machine fails the gate.

**Note on the profiler's printed advice.** It says a cache is the fix when decoding dominates, but jitter is also above 30%. If one more optimisation is needed, the next target is `KeepPadJitter`, for example batching it on a GPU or simplifying it.

### Conclusion

The loader passes the gate in five of six sessions on this laptop. The measured numbers agree with the CPU-bound model. The failing session is recorded, and the gate is re-measured on the training machine in Task 3.1.

---

## Part C: Unit tests and figure check

### Relevance (deep learning)

Tests protect properties that the earlier parts rely on. A later change that silently breaks one of them, such as a flip, a leak, or a padding change, would invalidate every result, and no accuracy figure would reveal it.

### Test results [Measured]

`uv run pytest tests/test_dataset.py -v`: **17 passed**, none skipped.

| Group | What the tests check |
|---|---|
| Split integrity | No validation or test signer in train; train MD5 unique; validation and test not de-duplicated; labels in range and complete; no blank rows |
| Batches | Train batch shape (64, 3, 224, 224) and label range; eval batch shape; train drops the last incomplete batch; validation covers every image; eval is deterministic |
| Augmentation rules | Padding restored exactly after jitter; content still jittered; affine fill exact outside the source; configured rotation, translation, and zoom values; no horizontal or vertical flip; eval has no augmentation |
| Class weights | Positive, and weighted total equals N |

### Figure check

`aug_check_part_a.png` was regenerated after the `KeepPadJitter` change. The tests cannot inspect it visually. **Open:** in the augmented rows, the left and right letterbox bars and the rotation corners should be the same grey (128), and no row should be mirrored. Confirm and record before submission.

### Known gap

The restore of padding is tested. The number of natural pixels that are exactly (128, 128, 128) and therefore escape jitter is not counted. Expected to be negligible; unverified.

---

## Errors found and fixed during Task 2.3

| # | Problem | Cause | Fix | Lesson for teammates |
|---|---|---|---|---|
| 1 | Normalisation gate failed | The gate assumed the crops have ImageNet channel means | Gate now checks the formula; channel means are reported only | Use the pretrained convention. Do not assume dataset statistics. |
| 2 | Augmentation changed the letterbox padding | `ColorJitter` ran on the whole image, after the affine | `KeepPadJitter` restores padding pixels exactly | Padding must be (128, 128, 128) in training and in evaluation |
| 3 | First `KeepPadJitter` order broke the fill | Moving jitter first left the rotation fill and the padding visually different | Jitter is applied first with padding restored, then affine with fill 128 | Check both borders in the figure after any transform change |
| 4 | Benchmark ignored the worker default | Script had its own default of 4 | Benchmark imports `DEFAULT_WORKERS`; the flag was removed | One source of truth for machine-specific settings |
| 5 | Benchmark crashed with `StopIteration` | Trial length was fixed, but the validation loader is shorter | Trial length computed from the loader length | Compute timing windows from the loader, not from constants |
| 6 | Throughput varied widely | Eight workers each started a multi-threaded pool | `worker_init_fn` sets one thread per worker | Always limit worker threads when using several CPU workers |
| 7 | Torchvision attribute names | `RandomAffine` stores degrees as a list and zoom as `scale` | Tests use the real attribute names | Check attribute names on the installed torchvision version |
| 8 | Profiler crashed with an undefined name | Placeholder left in the script | Import of `PROJECT_ROOT` | Run each script once after editing |

---

## Limitations

1. **Machine-specific throughput.** The gate was measured on a 16-CPU Windows laptop with no GPU. Re-measure on the training machine. On a machine with fewer cores, set `num_workers` to the available cores.
2. **Scale shortcut weakened, not removed (D21).** Augmentation is class-independent, so the network still sees class-dependent hand size, now over a wider range. Task 3.5 must include a scale ablation.
3. **Padding exemption.** Natural pixels that are exactly (128, 128, 128) are not jittered. Unmeasured.
4. **De-duplication is train-only (D10).** Validation and test keep their within-signer copies. This does not cause cross-split leakage, since the copies never cross splits, but it does mean the evaluation set contains repeated images from the same signer.
5. **Class weights are mild.** Ratio 1.18. Report per-class results as well as the average.
6. **Two test signers.** Per-signer accuracy must be reported in Task 4.1.
7. **Partial-entry frames in validation.** About 13 non-blank frames of H/P8 (from Task 2.2) are in validation. Read validation H with this in mind.
8. **Augmentation parameters are roadmap defaults**, except the scale range (D22). Not tuned. Augmentation-off ablation required (Task 3.5).
9. **Throughput excludes the model.** It measures the data pipeline only.
10. **Power plan not recorded for sessions A–C.**
11. **Augmentation seeding.** The DataLoader derives per-worker seeds from the global random generator, so augmentation is reproducible only if `torch.manual_seed` is set before training.

---

## Decisions made in this task

| # | Decision | Why | What it means for teammates | Status |
|---|---|---|---|---|
| D7 | 36 classes (0–9, A–Z) | Dataset has 36 classes; midterm text specifies classification only | Use `class_id` 0–35 from `config.CLASS_TO_ID` only | **Resolved** |
| D10 | Train-only exact-copy de-duplication, first per MD5 (423 removed); validation and test not de-duplicated | Duplicates would give I and V extra gradient weight; full evaluation set | Train has 24,777 images. Do not de-duplicate again. | **Resolved; deviation** |
| D21 | Scale is a nuisance variable; augmentation weakens it; Task 3.5 ablates it | Measured spread is 1.77× and augmentation cannot remove a class-dependent size | Report per-class scale when you report per-class accuracy | Active |
| D22 | Zoom range [0.75, 1.35] | Covers the observed class spread (zoom 0.754–1.339) | Do not narrow the range without rechecking the Part A scale gate | Active; deviation from roadmap |
| D23 | Order: `KeepPadJitter`, `RandomAffine`, `ToTensor`, `Normalize` | Colour jitter must not change padding or rotation fill | Do not reorder the transforms | Active |
| D24 | ImageNet mean and standard deviation | Matches the pretrained backbone in Model 3 | Keep the normalisation for Model 3. Do not recompute from the crops. | Active |
| D25 | No horizontal or vertical flip | Mirrored signs change meaning | Tests fail if a flip is added | Active |
| D26 | 8 workers | Best measured here | Pass `num_workers` explicitly on other machines | Active; **re-gate on training machine** |
| D27 | One intra-op thread per worker | Avoids oversubscription | Keep `worker_init_fn` | Active |
| D28 | No uint8 cache | Gate passes in five of six sessions; cache costs about 5.3 GB | First option if the training machine fails the gate | Deferred |
| D29 | Factory in `dataloaders.py` | Keeps dataset module free of loader policy | Import `get_dataloaders` from `dataloaders` | Active; **deviation** |
| D30 | Default 8 workers (leader: 4) | Four workers failed the gate here | Teammates should set workers to their core count | Active; **deviation** |
| D31 | `pin_memory = torch.cuda.is_available()` | Leader's `True` gives a warning on CPU | On GPU machines the effect is identical to `True` | Active; **deviation in form** |
| D32 | Rotation inside one `RandomAffine` call | Same distribution; one interpolation step | Rotation range is set by `ROTATION_DEG` in `dataset.py` | Active; **deviation in form** |
| D33 | `KeepPadJitter` wraps `ColorJitter` | Keeps padding exactly (128, 128, 128) | Use `build_transform`, not raw `ColorJitter` | Active; **deviation in form** |
| D34 | Throughput in a separate script | Machine-dependent timing must not gate unit tests | Run `benchmark_loader.py` for throughput | Active; **deviation in form** |
| D35 | Train 24,777 and validation 3,588 | Follows D10 and D18 | Per-class counts are unequal; use class weights | Active; **deviation in counts** |
| D36 | Median of five trials; session log with power plan | One slow trial or session should not decide the gate; machine state must be visible | Compare session logs, not single runs | Active |

---

## Files produced

| File | Role | Used by | Notes |
|---|---|---|---|
| `src/manifests.py` | Builds subject splits in memory: exclusions, signer split, train-only de-duplication | `dataloaders.py`, tests, Task 2.4 Part A | Task 2.4 freezes its output to CSV. Do not reimplement the split elsewhere. |
| `src/dataset.py` | `KeepPadJitter`, `build_transform`, `ASLDataset`, `class_weights`, normalisation constants | `dataloaders.py`, tests, Task 3.x training code | No loader or model code here. |
| `src/dataloaders.py` | `get_dataloaders` factory, `DEFAULT_WORKERS = 8`, worker thread limit | Task 3.1 training code | Entry point for teammates. |
| `src/check_dataset_part_a.py` | Part A gates and augmentation figure | Report reproduction | Re-run after any transform change. |
| `src/benchmark_loader.py` | Throughput gate, session log | Report, Task 3.1 re-gate | Appends one line per run to the session log. |
| `src/profile_transforms.py` | Per-stage cost of the training pipeline | Decision on caching (D28) | Diagnostic only; writes nothing. |
| `tests/conftest.py` | Makes `src/` importable for pytest | Tests | |
| `tests/test_dataset.py` | 17 tests on splits, batches, augmentation rules, class weights | Every later change to the data code | Run before committing data code. |
| `docs/task2_3_partA_dataset.json` | Part A report | Verification | |
| `docs/task2_3_partB_loader.json` | Latest benchmark report | Verification | Overwritten each run. |
| `docs/task2_3_partB_sessions.jsonl` | Per-session throughput log with power plan | Report evidence | Append-only. Do not delete. |
| `docs/figures/aug_check_part_a.png` | Augmentation check figure | Visual review | Visual check open. |

**Delete:** `src/finish_part_b_report.py`.

---

## Hand-off to the next task

### Task 2.4: subject-independent partitioning, random split, external real-world set

Task 2.4 has four parts. Parts A and B can run without any team photos. Part C depends on team collection. Part D verifies everything.

**Part A: Subject-independent split CSVs (code).**
- *Objective:* freeze the subject split to `data/splits/subject_train.csv`, `subject_val.csv`, `subject_test.csv`.
- *Method:* call `manifests.build_manifests()`. Do not re-derive the split anywhere else. Write these columns, and only these required ones plus the extras:
  - `filepath`: raw relative path, for example `data/raw/K/P3_K_042.jpg`.
  - `class_name`: string label, for example `"0"` or `"K"`. Read with `dtype=str, keep_default_na=False`.
  - `class_id`: integer 0–35 from `config.CLASS_TO_ID`.
  - `volunteer_id`: integer 1–10.
  - Extras: `crop_path`, `md5`.
- *Also write:* `data/splits/class_mapping.json`, `data/splits/excluded_train_copies.csv` (the 423 removed rows with the MD5 group they belong to, as the audit trail), and `data/splits/SHA256SUMS.txt`.
- *Write-once:* the script refuses to overwrite an existing CSV. A change requires a new logged decision in Section 8 of the consolidation and a deliberate `--new-version` flag.
- *Gates:* counts 24,777 / 3,588 / 7,200; no volunteer in two splits; all 36 classes in each split; no MD5 duplicates in train; every `crop_path` exists; no blank rows; `class_name`↔`class_id` matches `config`.
- *Expected:* exactly the counts above. The leader's 25,200 and 3,600 are not reached (D35).

**Part B: Random split CSVs for Ablation 1 (code).**
- *Objective:* `data/splits/random_train.csv`, `random_val.csv`, `random_test.csv`, a stratified 70/15/15 split with seed 42, used only in Task 3.5.
- *Method:* the same two-stage `train_test_split` as `audit_random_split.py`, applied to the **35,988 written crops** (blank frames excluded, D18). **No de-duplication**, because this is the naive protocol that Ablation 1 is meant to compare against.
- *Expected:* about 25,192 / 5,398 / 5,398 [Expected]. The script records the exact counts.
- *Important:* the Task 2.1 Part B-3 leakage numbers were measured on the earlier 36,000-file version with blanks. **Re-run leakage on the frozen random split** and report the new numbers. Do not reuse the old ones.
- *Gates:* counts sum to 35,988; every class in every split; no blank rows; stratification within ±1 image per class.

**Part C: External real-world benchmark (team work plus code).**
- *C1. Collection (team).* 72–108 photos: 2–3 per class across all 36 classes. Mix at least three environments (classroom, dormitory, outdoor) and at least two lighting types. Include both a phone and a laptop webcam. Each team member contributes. Photograph hands only. Do not include faces. Record consent in the team folder.
- *C2. Naming and manifest.* Files under `data/external_test/<class>/<member>_<class>_<n>.jpg`. The manifest `data/external_test/manifest.csv` has the leader's columns: `filepath`, `class_name`, `volunteer_name`, `lighting_type`, `camera_source`. The `.gitignore` tracks the manifest and ignores the photos.
- *C3. Resolution normalisation (code, required).* Training crops came from 300×300 frames. External photos are of arbitrary size. Before detection, resize each photo so its **longer side is 300 px**, keeping the aspect ratio. Then run the Task 2.2 pipeline unchanged. This keeps the letterbox scale comparable to training. Without this step the scale is measured in source pixels, which encodes resolution rather than hand size. Also make `roi_detector.py` read the frame size from the image rather than assuming 300×300 (code change).
- *C4. Domain check (code).* Every external crop's letterbox scale should fall inside the training scale range (0.747–1.325 observed, zoom range 0.75–1.35). Report the external per-class scale. Check that no external photo has an MD5 equal to a Mendeley image; a match would mean a copied image and must be removed.
- *C5. Use rule.* The external set is for final evaluation in Task 4.1 only. It is never used for training, early stopping, or model selection.
- *Blocker status:* C1 cannot be done by code. C3 and C4 can be written now and tested on any photo the team has already taken.

**Part D: Verification script (code).**
- `src/check_splits.py` runs every gate from Parts A, B, and C, prints PASS/FAIL, and writes `docs/task2_4_split_verification.json`. Exit code 0 only if all gates pass.
- Output includes the per-class count table, the signer table, the scale table for external data, and the SHA-256 check of every frozen file.
- `src/setup_external_test.py` is rewritten to create the 36 class folders and validate the manifest columns.

**Report:** `docs/task2_4_splits_external.md`, in the Section 7 format, with one part for each of A–D.

### Task 3.1: training engine (summary for the next owner)

- Call `get_dataloaders(train_csv, val_csv, test_csv, batch_size=64)`. The factory accepts paths directly after Task 2.4.
- Call `torch.manual_seed(42)` before building the model and loaders.
- Pass `num_workers` as the machine's core count if it differs from 8.
- Use `class_weights(train_manifest)` as the `CrossEntropyLoss` weight. The ratio is only 1.18, so the effect is small.
- Re-run `benchmark_loader.py` on the training machine and record the session.

---

## Notes for teammates

1. **Use `get_dataloaders`.** Do not write a new `DataLoader` for training. The factory already sets `drop_last`, the seeded shuffle, worker threads, and persistent workers.
2. **Pass `num_workers` from the machine.** The default is 8 for this laptop. Check the core count on Kaggle and the RX 6700 XT machine. The value 8 is **[Expected]** for Kaggle's typical setup and unmeasured.
3. **Windows needs a main guard.** Any script that uses workers must put its work under `if __name__ == "__main__":`. Otherwise spawned workers re-run the script.
4. **Class weights come from the training manifest after de-duplication.** Pass `class_weights(train_manifest)` to `CrossEntropyLoss(weight=...)`.
5. **Evaluate with the eval transform only.** Validation and test use `is_train=False`.
6. **Do not add flips.** The test suite checks for them.
7. **Report per signer.** Test has P9 and P10. Validation has P8.
8. **Remember the partial-entry frames.** Class H in validation includes about 13 partial-entry frames.
9. **Scale is a confound.** Include the scale ablation in Task 3.5 (D21).
10. **Pretrained weights expect ImageNet normalisation.** Keep it for Model 3.
11. **Seed before training.** Call `torch.manual_seed(42)` so augmentation is reproducible.

---

## Reproduction, in order

```
uv run python src/check_dataset_part_a.py
uv run pytest tests/test_dataset.py -v
uv run python src/benchmark_loader.py          (repeat so the session log has several lines)
uv run python src/profile_transforms.py
```

---

## Corrections to earlier reports

- **Task 2.2 scale spread.** The report said 0.83–1.33 across classes, about 1.6×. Correct: 0.747–1.325, about 1.77×, with class 0 included.
- **D10 wording.** The earlier consolidation said copies were kept as delivered. Correct: train-only de-duplication (423 removed).
- **Train size.** The earlier summary implied 25,200 train images. Correct: 24,777 after de-duplication, 25,200 before.