# Task 2.1: Dataset Integrity, Signer Structure, and EDA (ASL-HG)

**Status:** Measurement complete. One figure (36-class grid) pending regeneration (Section 8).
**Date:** 2026-10-05
**Consolidation reference:** PROJECT_CONSOLIDATION.md, Sections 4, 8, 9
**Scripts:** `src/verify_dataset.py`, `src/audit_signers.py`, `src/audit_cross_split.py`, `src/audit_random_split.py`, `src/eda_figures.py`, with shared constants in `src/config.py`

Labels used throughout:
- **[Measured]**: computed by our scripts on the files in `data/raw/`.
- **[Reported]**: stated in the paper or dataset page. Not verified by us.
- **[Expected]**: a target from the roadmap or the rubric. Not yet tested.

---

## Summary

The ASL-HG raw set is structurally clean. It has 36 class folders with 1,000 images each (36,000 total), and each of 10 volunteers contributed exactly 100 images per class. Every image decodes fully and is 300×300 RGB. There are no stray files.

Two dataset properties matter for modelling. First, 199 groups of byte-identical images (400 files) link volunteers P5 and P6, and all of them are in classes I and V. Because of these copies, I and V have 893 and 894 distinct images instead of 1,000. Second, images within a signer's recording burst are very similar to their neighbours.

We measured the effect of the split protocol directly. Under the subject-independent split (train P1–P7, validation P8, test P9–P10), no exact copy group crosses a split, and no test image is within Hamming distance 1 of a training image. Under a random stratified 70/15/15 split, 214 of 435 copy groups cross splits, and about 86% of validation and test images have a training image within Hamming distance 1. This gap is the leakage that Ablation 1 must convert into an accuracy gap. We have not measured accuracy yet (Tasks 3.4 and 3.5).

The main limitation is that this task measures data structure. It does not measure model performance, and it cannot establish the causes of the signer-level brightness differences.

---

## Part A: Structure, Counts, Filenames, Decoding

### Relevance (deep learning)

Every later result depends on the labels and the sample counts being right. A wrong label, a missing class, a truncated file, or a file with a mismatched filename silently corrupts training, and the error appears later as a strange accuracy figure. Part A rules out these failures before any model is trained. It also establishes the signer label, which the subject-independent split needs.

### Definitions

- **Class:** one of 36 labels, digits `0`–`9` and letters `A`–`Z`. Digit `0` is the two-handed ASL zero sign, which is distinct from letter `O` (paper, Table 1).
- **Volunteer (signer):** one of 10 contributors, identified in the filename prefix `P<id>`.
- **Index:** the number in the filename, running 1 to 1,000 within each class. Volunteer `k` covers indices `(k−1)·100 + 1` through `k·100`. For example, P5 covers 401–500 and P6 covers 501–600.
- **Full decode:** the image is read into memory and all pixels are decoded (`Image.load()`), which catches truncated files that a header-only check would miss.

### Objective

Verify, with measurements rather than assumptions, that:
1. The folder structure matches the 36-class layout with no missing or extra folders.
2. Each class has 1,000 images, and each volunteer has 100 images per class.
3. Every image decodes and has the expected size (300×300) and mode (RGB).
4. The filename class matches the folder class for every file.

### Methodology

`config.py` discovers the dataset root by searching for the folder that directly contains all 36 class folders. In the download, that folder is `data/raw/`, with no nested subfolder. `verify_dataset.py` then:
- lists every class folder and counts image and non-image entries,
- reads each image once, computes an MD5 hash of its bytes, and fully decodes it,
- parses each filename with the pattern `P<id>_<class>_<index>.<ext>`,
- checks that the filename class equals the folder class,
- checks the volunteer-by-class count matrix,
- flags byte-identical files.

The script is read-only. It writes `docs/task2_1_partA_inspection.json`.

### Code output analysis [Measured]

| Check | Expected | Measured |
|---|---|---|
| Class folders | 36, none missing | 36, none missing, no extra folders |
| Non-image files under `raw/` | 0 | 0 |
| Images per class | 1,000 | 1,000 for all 36 classes |
| Total images | 36,000 | 36,000 |
| Full decode | all | 36,000 of 36,000, zero failures |
| Size | 300×300 | 36,000 of 36,000 |
| Mode | RGB | 36,000 of 36,000 |
| Filenames parsed | all | 36,000 of 36,000 |
| Filename class vs folder | all match | 0 mismatches |
| Images per volunteer | 3,600 | 3,600 for each of P1–P10 |
| Volunteer × class cells | 100 each | 0 off-target cells |

The structural and metadata checks pass with no exceptions. The one warning is the byte-identical copies, which Part B analyses.

### Conclusion

The dataset is complete, uniformly formatted, and correctly labelled. The filenames carry signer identity and index, which makes the subject-independent split possible. Part A establishes that all subsequent counts are measured on a complete set.

---

## Part B: Signer Structure, Redundancy, and Split Leakage

### Relevance (deep learning)

A CNN learns whatever features reduce training loss. If the training set contains near-copies of the test images, the network can score well by recognising those copies rather than learning hand shape. This is the shortcut-learning problem [1] in the form of data leakage. A random split of a burst-captured dataset is the usual way this happens. The subject-independent split is the standard defence, because the test signers' hands never appear in training.

Part B measures three things: how much redundancy exists within a signer, how much exists across signers, and how much each split protocol leaks.

### Definitions and formulas

**Exact identity (MD5).** Two files are byte-identical if their MD5 hashes match. This is a reliable definition of an exact copy.

**Difference hash (dHash, 64 bits).** The image is converted to grayscale and resized to 9×8 pixels. Let $p_{r,c}$ be the pixel at row $r \in \{1..8\}$ and column $c \in \{1..9\}$. The 64 bits are

$$
b_{r,c} = \mathbb{1}\left[\,p_{r,c+1} > p_{r,c}\,\right], \quad c \in \{1..8\}
$$

This captures horizontal intensity gradients, so it is insensitive to small brightness changes but sensitive to structure.

**Hamming distance.** For two 64-bit hashes $x, y$:

$$
d_H(x, y) = \sum_{i=1}^{64} \mathbb{1}[x_i \neq y_i] = \mathrm{popcount}(x \oplus y)
$$

A distance of 0 means identical hash bits. It does **not** mean an identical image. The 9×8 hash is coarse. In the index, 35,441 distinct MD5 values correspond to only 17,382 distinct dHash values, so many visually different images share a hash. For this reason, exact copies are defined by MD5 and near-duplicates by dHash, and the two are never mixed.

**Near-duplicate leakage at threshold $t$.** A held-out image $h$ is a near-duplicate of training if

$$
\min_{i \in \text{train}, \; y_i = y_h} d_H(h_{\text{hash}}, i_{\text{hash}}) \le t
$$

We report $t = 1$ as the primary threshold and $t = 5$ as secondary.

**Border luminance proxy.** Mean grayscale intensity of the outer 10% of the image frame (on each side). Because hands tend to sit in the centre, the border is mostly background. This is a proxy only. Backgrounds are not counted, and the dataset has no background metadata.

**Probability that a copy pair crosses a random split.** For a pair of identical images assigned independently with proportions $0.70, 0.15, 0.15$:

$$
P(\text{different splits}) = 1 - (0.70^2 + 0.15^2 + 0.15^2) = 0.465
$$

This is approximate, because the split is stratified without replacement, but it gives a benchmark for the measured crossing rate.

### Objective

1. **B-1 (Exact copies and signer statistics):** find byte-identical copies, check whether they span signers, and profile each signer's brightness and colour.
2. **B-2 (Subject split leakage):** check whether any exact copy or near-duplicate crosses the roadmap's subject-independent split.
3. **B-3 (Random split leakage):** measure the same quantities under a stratified 70/15/15 random split, which is needed for Ablation 1.

### Methodology

`audit_signers.py` computes, for each image, its MD5, dHash, mean and standard deviation of luminance, mean R/G/B, and border statistics. Results are cached in `data/processed/asl_hg_image_index.csv`. Burst structure is measured with three kinds of pair:
- **Consecutive index, same signer:** adjacent indices within each (signer, class) block. This assumes filename index tracks capture order.
- **Random same class, same signer:** 2,000 seeded random pairs per class.
- **Random same class, cross signer:** 2,000 seeded random pairs per class.

`audit_cross_split.py` uses the roadmap's subject split: train P1–P7 (25,200 images), validation P8 (3,600), test P9–P10 (7,200). It counts copy groups crossing a split and computes, for each held-out image, its nearest same-class distance to the training split.

`audit_random_split.py` builds a stratified 70/15/15 split with seed 42 in memory only. It writes nothing to `data/splits/`, which is reserved for Task 2.4. It runs the same two checks.

### B-1: Exact copies and signer statistics

**Exact copies [Measured].**

| Quantity | Value |
|---|---|
| Byte-identical groups (size > 1) | 435 (994 files) |
| Groups within one signer | 236 |
| Groups spanning signers | 199 (400 files) |
| Groups spanning classes | 0 |
| Redundant copies (36,000 − 35,441 distinct MD5) | 559 |

Every cross-signer group is P5 with P6. The volunteer-pair matrix has 199 in the (P5, P6) cell and zero elsewhere. All cross-signer groups fall in classes I and V (99 groups in I, 100 in V). The filename indices show a consistent offset of 100, which is exactly one volunteer block. Most P6 images in I and V therefore appear to be copies of P5 images. The files alone do not show which copy was original.

**Distinct images per class [Measured].** Distinct MD5 per class is 893 for I and 894 for V. Other classes range from 972 (class `5`) to 1,000 (class `F`). The 1,000-per-class figure counts files, not unique images.

**Per-signer statistics [Measured].**

| Signer | Mean L | SD of image mean L | Mean R | Mean G | Mean B | Border L | Border SD L |
|---|---|---|---|---|---|---|---|
| P1 | 168.1 | 5.5 | 169.5 | 168.6 | 162.2 | 174.1 | 21.7 |
| P2 | 177.6 | 8.2 | 187.6 | 173.6 | 171.9 | 192.4 | 35.7 |
| P3 | 201.7 | 27.1 | 206.2 | 202.3 | 186.2 | 212.3 | 43.8 |
| P4 | 160.2 | 15.4 | 173.4 | 155.6 | 149.1 | 169.6 | 43.6 |
| P5 | 183.7 | 6.0 | 196.9 | 180.4 | 166.3 | 192.0 | 29.4 |
| P6 | 174.2 | 7.5 | 177.9 | 173.9 | 166.1 | 175.7 | 17.9 |
| P7 | 160.5 | 5.8 | 166.8 | 159.3 | 149.9 | 164.9 | 39.7 |
| P8 | 193.5 | 6.2 | 208.2 | 186.9 | 188.8 | 204.3 | 33.1 |
| P9 | 190.5 | 7.7 | 199.4 | 187.4 | 183.4 | 204.5 | 35.7 |
| P10 | 200.2 | 5.6 | 208.9 | 197.9 | 189.7 | 210.0 | 43.0 |

Within-signer spread (SD of image mean L) is small for most signers, at 5–8. P3 is the outlier at 27.1, and P4 is also high at 15.4. Signer means range from 160 (P4, P7) to 202 (P3), so lighting varies much more between signers than within them.

**Burst structure [Measured].**

| Pair type | n | Median | Mean | P99 | Share ≤ 1 | Share ≤ 5 |
|---|---|---|---|---|---|---|
| Consecutive index, same signer | 35,640 | 1 | 2.78 | 21 | 0.505 | 0.868 |
| Random same class, same signer | 7,055 | 5 | 5.96 | 24 | 0.171 | 0.580 |
| Random same class, cross signer | 64,864 | 30 | 29.7 | 45 | 0.0002 | 0.0005 |

Consecutive images within a signer are very close: half are within distance 1, and 87% are within distance 5. Random pairs in the same signer are much further apart, and random cross-signer pairs are almost never close. The median consecutive distance by signer ranges from 0 (P1) to 3 (P10).

Two points about the cross-signer distribution need care. The roadmap states that cross-signer hands are "completely distinct (Hamming distance 23 to 34)." The measured median of 30 falls in that range, but the 1st percentile is 14 and the minimum is 0 because of the copy groups. The "23 to 34" band describes the centre of the distribution, not a bound. The roadmap's consecutive mean of about 0.75 is also not reproduced: the measured mean is 2.78, with median 1.

**Interpretation.** Within-signer redundancy is high, and exact copies are concentrated in one signer pair. Any split that puts images from one recording burst into two different sets will leak. The signer-level grouping is therefore the correct unit for splitting.

### B-2: Subject-independent split leakage

**Measured [Measured].**

| Check | Result |
|---|---|
| Copy groups across signers | 199 |
| Copy groups crossing train/val/test | **0** |
| Validation: nearest same-class distance to train | min 6, p1 9, median 16 |
| Validation: share ≤ 1 / ≤ 5 | 0.0 / 0.0 |
| Test: nearest same-class distance to train | min 4, p1 6, median 17 |
| Test: share ≤ 1 / ≤ 5 | 0.0 / 0.0042 (about 30 of 7,200) |

The 199 copy groups all sit inside train, because P5 and P6 are both training signers. No validation or test image has an exact match in train, and only 30 test images (0.42%) are within distance 5. Validation has none within distance 5.

**Interpretation.** The subject split is clean for exact copies and for near-duplicates at the thresholds we use. The 30 test images within distance 5 are most likely similar handshapes rather than copies. The 9×8 dHash is too coarse to confirm this. A finer measure (for example, a CNN embedding distance) is the right test and belongs with Task 4.1.

### B-3: Random split leakage (for Ablation 1)

**Measured [Measured].** Random stratified split: train 25,200, validation 5,400, test 5,400 (700 / 150 / 150 per class).

| Check | Validation | Test |
|---|---|---|
| Nearest same-class distance = 0 | 63.2% | 61.2% |
| Nearest same-class distance ≤ 1 | 86.9% | 86.4% |
| Nearest same-class distance ≤ 5 | 99.8% | 99.8% |
| Copy groups crossing a split | 214 of 435 (523 files) | — |

Random splitting places a near-identical training image next to most held-out images. The 214 of 435 copy groups crossing splits (49%) is close to the 47% expected for independent random assignment, which confirms the split was made as intended.

**Comparison with the subject split.**

| Metric | Random | Subject |
|---|---|---|
| Test: share ≤ 1 | 86.4% | 0.0% |
| Test: share ≤ 5 | 99.8% | 0.4% |
| Copy groups crossing a split | 214 | 0 |

**Caveats.**
1. The distance-0 counts include hash collisions, so they are an upper-bound-style measure. The copy-group crossing rate uses MD5 and is the reliable number.
2. These are leakage measurements, not accuracy results. The roadmap's expected accuracies (about 98.5% random, 90.4% subject) are reported by the leader and have not been reproduced.
3. The two protocols have different test sizes (5,400 random, 7,200 subject). Ablation 1 must report this.
4. Random splitting leaks two things at once: exact copies, and burst neighbours from the same signer. These are not separated here. An optional check would record, for each test image, whether its nearest train neighbour is from the same signer.

### Conclusion

The subject split removes measurable leakage at both the exact-copy and near-duplicate levels, and random splitting does not. The P5/P6 copy defect lies inside the training set and does not contaminate evaluation. Part B therefore supports the subject split as the primary protocol and provides the leakage measurement that Ablation 1 needs.

---

## Part C: Figures

### Relevance (deep learning)

Figures are the first evidence a reader sees, and they must be accurate. A grid that appears to show variety when it shows a single signer misrepresents the dataset. The figures serve three functions: they confirm class balance, show the diversity of signs, and show the signer and lighting variation that motivates the subject split.

### Definitions

- **Seeded sample:** an image chosen with a fixed random seed (42), so the figure can be regenerated exactly.
- **Border proxy:** see Part B. Labelled as a background proxy wherever it appears.

### Objective

Produce the three artifacts the roadmap requires under `docs/figures/` (class balance, 36-class grid, signer and lighting mosaic) using only the cached index and a small number of decoded images. No figure should imply a property the data does not have.

### Methodology

`eda_figures.py` reads the cache and decodes only the sample images. It does not filter by quality, so the figures show what the data contains.

### Figure audit [Measured by inspection]

| Figure | Status | Finding |
|---|---|---|
| `eda_class_balance.png` | Correct; cosmetic fix pending | Bars match Part A (1,000 each). The legend overlaps the U–Z bars. |
| `eda_volunteer_mosaic.png` | Correct | One image per signer, sign A. Labels and brightness values are consistent with Part B. |
| `eda_36_class_grid.png` | **Defective, regeneration pending** | All 36 tiles come from P5. The sampler selects the same row position in every class, and the row ordering is by volunteer block, so the tiles all share one signer. The caption promises variety that the figure does not show. |

Two qualitative observations from the mosaic are relevant to the analysis, but are not measured. P1, P6, and P7 share a grey-wall background, and P8, P9, and P10 share a bright, plain background. If background correlates with signer, the subject split does not fully test background generalisation, because a test signer's background may resemble a training signer's. This needs a background clustering measurement, listed in Section 7.

### Conclusion

The class balance and the volunteer mosaic are accurate. The 36-class grid must be regenerated with signers rotated across classes before it can be used in the report. Until then, the grid should not be cited as evidence of signer diversity.

---

## 6. Overall Interpretation for the Project

Task 2.1 establishes four things that shape the rest of the project.

1. **The dataset is clean and complete.** There are no label, size, or decoding problems, and the signer label needed for splitting is present in every filename.
2. **The dataset has duplicate structure.** The P5/P6 copies in classes I and V are a defect that must be documented. They do not affect the subject split because both signers are in training.
3. **The subject split is the correct primary protocol.** It has zero exact copy crossings and essentially zero near-duplicate leakage. Random splitting leaks heavily, so random-split accuracies cannot be read as generalisation.
4. **Signer-level brightness varies a lot.** Mean luminance ranges from 160 to 202 across signers, while within-signer variation is small for most. Any accuracy gap between validation or test and training may partly reflect lighting shifts rather than hand-shape differences. Task 4.1 must report per-signer accuracy so the two effects can be separated.

---

## 7. Limitations

1. **Structure, not accuracy.** No model has been trained. The leakage numbers show the mechanism; the accuracy gap must be measured in Tasks 3.4 and 3.5.
2. **Hash resolution.** The 9×8 dHash has many collisions (35,441 distinct MD5 vs 17,382 distinct dHash). Near-duplicate results are indicative, not definitive.
3. **Filename index as capture order.** The consecutive-pair analysis assumes the index follows capture order. This is supported by the distance pattern (adjacent pairs are far closer than random pairs) but is not independently verified.
4. **Background is not measured.** The dataset has no background metadata, and the border statistic is a proxy. Background count and background-signer correlation remain unknown. A background-clustering analysis is an open item.
5. **Skin tone is not measured.** The paper claims a balance across skin tones and genders. This cannot be checked from the data. Mean RGB is confounded by background and lighting.
6. **Camera source conflicts.** The paper's Section 4.1 says all images were captured on one OPPO Reno5 smartphone. Its Section 4 and the dataset page describe ten volunteers using smartphone cameras. The per-signer brightness differences suggest varied capture conditions, so the single-camera claim is probably inaccurate. Unresolved; should be raised with the dataset authors.
7. **Geography.** The paper states that collection took place in Mirpur, Dhaka. This is reported, not verified.
8. **Validation and test size.** Validation is one signer (P8) and test is two (P9, P10). Single-signer validation is noisy for model selection. Per-signer reporting is required.
9. **Roadmap figures not reproduced.** The roadmap's consecutive mean (about 0.75) and cross-volunteer band (23–34) are not confirmed by measurement. The measured values are reported in Part B.
10. **Figure regeneration pending.** The 36-class grid is defective as described above.

---

## 8. Decisions Made in This Task

| # | Decision | Reason | Status |
|---|---|---|---|
| D8 | Class set is 36 classes, ordered `0`–`9` then `A`–`Z`. | Matches the paper and roadmap. Digit zero is distinct from letter O. | Active, pending D7 |
| D9 | Subject-independent split (train P1–P7, validation P8, test P9–P10) is the primary evaluation protocol. | Zero exact-copy crossings and zero near-duplicates at distance ≤ 1 for validation and test. | Active |
| D10 | The 199 P5/P6 copy groups in classes I and V are documented as a dataset defect and kept as delivered. A deduplicated training set is a sensitivity run in Task 3.5. | Removing them changes class balance and the signer structure. Both signers are in training, so evaluation is unaffected. | Active; confirm with team |
| D11 | Random stratified 70/15/15 (seed 42) is used only for Ablation 1, not as the primary evaluation. | Measured leakage: 214 of 435 copy groups cross splits; 86% of test images within distance 1 of train. | Active |
| D12 | Exact copies are identified by MD5. dHash is used only for near-duplicate measurement and is never treated as an identity test. | dHash has many collisions (35,441 MD5 vs 17,382 dHash). | Active |
| D13 | The 36-class grid is regenerated with signers rotated across classes. | The current grid shows one signer for all 36 classes. | Pending regeneration |

Per-class unique counts (distinct MD5) replace the "1,000 unique per class" claim. The raw files remain 1,000 per class.

---

## 9. Open Items

1. **D7, class set (Dr. Ngân).** Confirm that 36 classes and the revised quotas are acceptable. The subject split does not use quotas, so only the class-set question remains.
2. **Camera source.** Confirm with the dataset authors whether a single device or multiple devices were used, given the conflict in the paper.
3. **P5/P6 defect.** Report the 199 copy groups to the dataset authors through the Mendeley page.
4. **Background clustering.** Measure whether backgrounds correlate with signer, so the background-shortcut claim can be tested rather than assumed.
5. **Accuracy reproduction.** Reproduce the leader's 90.39% (subject) and 96–98% (random) results with a script, a seed, and a recorded configuration (Task 3.4).
6. **Near-duplicate measurement.** Replace the 9×8 dHash with a finer measure, or a CNN embedding distance, to confirm the 30 test images within distance 5.
7. **Licensing and citation.** Check the Mendeley licence and cite the paper [15] and the data article.

---

## 10. Files Produced

| File | Purpose |
|---|---|
| `src/config.py` | Shared constants, filename parser, dataset-root discovery |
| `src/verify_dataset.py` | Part A: structure, counts, decoding, filenames |
| `src/audit_signers.py` | Part B-1: exact copies, signer statistics, burst structure |
| `src/audit_cross_split.py` | Part B-2: subject-split leakage |
| `src/audit_random_split.py` | Part B-3: random-split leakage (in memory, no splits written) |
| `src/eda_figures.py` | Part C: figures |
| `data/processed/asl_hg_image_index.csv` | Cached per-image index (not committed) |
| `docs/task2_1_partA_inspection.json` | Part A report |
| `docs/task2_1_partB_signers.json` | Part B-1 report |
| `docs/task2_1_partB2_cross_split.json` | Part B-2 report |
| `docs/task2_1_partB3_random_split.json` | Part B-3 report |
| `docs/figures/dataset/eda_class_balance.png` | Class balance |
| `docs/figures/dataset/eda_36_class_grid.png` | 36-class grid (regeneration pending) |
| `docs/figures/dataset/eda_volunteer_mosaic.png` | Signer and lighting mosaic |
| `docs/figures/dataset/eda_signer_luminance.png` | Per-image luminance by signer |
| `docs/figures/dataset/eda_signer_hamming.png` | Hamming distance distributions |

Reproduction, from the repository root, in this order: `verify_dataset.py`, `audit_signers.py`, `audit_cross_split.py`, `audit_random_split.py`, `eda_figures.py`.

The old grassknoted reports are archived under `docs/archive_grassknoted/` and are not used here. The Mendeley `processed/` folder is not read. It should be moved out of `data/processed/`, so it does not collide with our outputs.

---

## References

[1] R. Geirhos et al., "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, 2(11), 665–673, 2020.
[15] M. F. I. Pranto et al., "ASL-HG: American Sign Language hand gesture image dataset," Mendeley Data, v1, 2025. doi:10.17632/j4y5w2c8w9.1. Accompanying article: *Data in Brief*, 65 (2026) 112492.