# Task 2.1: Dataset Integrity Audit and Exploratory Data Analysis

**Status:** Integrity audit passed. EDA figures and notebook complete. Remaining edits are listed in Section 9.

## 1. Dataset and Integrity Audit

- **Source:** Kaggle ASL Alphabet (grassknoted). 29 class folders (A–Z, `del`, `nothing`, `space`), RGB JPEG, 200×200 px.
- **Used:** `data/raw/asl_alphabet_train/` only. The Kaggle test folder is a separate set and is excluded from all splits and analyses.
- **Method:** `src/verify_dataset.py` opens each image once and fully decodes it with `Image.load()`, which detects truncated pixel data that header-only checks miss. It checks folder structure, per-class counts, zero-byte files, stray files, decode failures, size, and color mode. It writes `docs/task2_1_integrity_report.json`.

| Check | Result |
|---|---|
| Class folders | 29 of 29 present; no missing or extra folders |
| Images per class | 3,000 for every class |
| Total images | 87,000 / 87,000 |
| Zero-byte files | 0 |
| Decode failures | 0 |
| Images not 200×200 | 0 (all 87,000 are 200×200) |
| Images not RGB | 0 (all 87,000 are RGB) |
| Stray files | 0 |
| **Verdict** | **Passed** |

## 2. Class Distribution (`eda_class_distribution.png`)

Every class has exactly 3,000 images, so the empirical class prior is uniform: p(y) = 1/29 ≈ 0.034.

- Balanced priors mean the loss has no majority-class bias, so class-weighted losses are not required for this dataset.
- Macro-averaged metrics are appropriate because each class counts equally.
- This describes the Kaggle source only. Webcam use may have a different class mix (for example, more `nothing` frames), so a balanced prior does not guarantee balanced performance in deployment.

## 3. Class Overview (`eda_29_class_mosaic.png`)

The 5×6 grid shows one sample per class. It is a visual sanity check that each sample is consistent with its label, not proof of correctness. The `nothing` class shows an empty scene, which makes it a useful control.

## 4. Lighting and Background Audit (`eda_lighting_and_background_audit.png`)

**Measured:** mean grayscale intensity of 600 randomly sampled images (seed 42). The mean is 131.3, and values range from about 80 to 175. The histogram has a dominant mode near 140–145 and a secondary bump near 100.

This is a per-image mean over the whole frame, so it mixes hand and background pixels. The two-mode shape may reflect darker hands against a brighter wall, but the modes cannot be assigned to hand and background without a hand mask. Task 2.2 produces masks, and this measurement should be repeated with them.

**Within-class variation (class A):** the five frames shown are at positions 100–104 in the lexicographically sorted file list (`A1, A10, A100, …`). They are not consecutive recording frames.

- Frames at positions 100, 102, 103, and 104: the hand moves slightly while the ceiling seam and window molding stay in place.
- Position 101: a strongly underexposed silhouette. Lighting varies within a class, not only across classes.

## 5. Near-Duplicate Structure (Measured)

Within each class, every image was compared with every other image using a 64-bit difference hash (9×8 grayscale, `src/split_data.py`). The distribution of nearest-neighbour distances is:

| Nearest same-class neighbour, Hamming distance | Images | Cumulative share |
|---|---|---|
| 0 (identical hash) | 40,173 | 46.2% |
| ≤ 1 | 66,401 | 76.3% |
| ≤ 2 | 78,788 | 90.6% |
| ≤ 5 | 86,620 | 99.6% |

**Interpretation.** The dataset is highly redundant. Most images have a visually near-identical neighbour in their own class, which means a random per-image split would place near-duplicates in both train and test. This is the measured basis for the group-aware split in Task 2.4.

**Limits of this measurement:** dHash is a coarse 64-bit descriptor. A distance of 1 to 5 does not guarantee the same photo, and uniform images (such as blank walls) collide more easily. The histogram is a proxy and should be checked against image pairs by eye before it is reported as fact. Cross-class near-duplicates were not measured.

## 6. Shortcut-Learning Risk (Hypothesis)

In the frames inspected, the background is almost identical within a class. If the background is also consistent across classes, a CNN could learn background features that correlate with the label, which is an instance of shortcut learning [1].

This is a hypothesis drawn from a few frames, not a measured result. Two tests address it:

1. **Background-swap ablation (Task 3.5):** train Model 2 with `bg_swap=False` and `bg_swap=True`, then compare them on the external webcam set.
2. **Per-class background statistics:** once masks exist (Task 2.2), measure background color and edge statistics outside the hand for each class.

Tasks 2.2 and 2.3 are designed to reduce this risk. The ablation will show whether the reduction is needed.

## 7. Visually Confusable Groups (`eda_ambiguous_pairs.png`)

| Group | Classes | Visual difference (from the figure annotations) |
|---|---|---|
| Fist tuck | M, N, S | Thumb under three fingers (M), under two (N), or folded across the front (S) |
| Thumb placement | A, S, E | Thumb beside the index finger (A), across the knuckles (S), fingers curled over the thumb (E) |
| Two-finger shapes | K, V, U | Spread fingers (V), thumb wedged between (K), fingers pressed together (U) |

The M and N examples are motion-blurred, so the figure illustrates the groups but is not a clean reference. The thumb occupies a small area in these images. Most errors are expected in these groups, which must be confirmed with the confusion matrix (Task 4.2).

**Architectural reasoning (to be tested):**

- Repeated pooling reduces spatial resolution. If the thumb cue spans only a few pixels after early downsampling, a plain CNN may lose it. Model 1 is therefore expected to struggle on M vs N. This is a prediction, not a result.
- Residual connections $\mathbf{x}_{l+1} = \mathcal{F}(\mathbf{x}_l) + \mathbf{x}_l$ [2] ease optimization of deeper networks and let information pass unchanged within a stage. They do not restore resolution lost to pooling. If Model 2 needs finer detail, the more direct lever is reducing early downsampling (for example, a stride-1 stem or fewer pooling layers). The choice should be checked with ablations.
- MobileNetV3's squeeze-and-excitation blocks [3] reweight channels, which may help with subtle cues. The EDA does not show this directly.

## 8. Limitations

- The findings come from the images alone. The collection protocol of the source dataset is not documented in our materials, so we make no claims about how it was recorded.
- The visual inspection is qualitative. The brightness, hand-position, and background claims need the measurements listed in Section 9.
- Sample sizes are small for some checks: 600 images for brightness and about five frames per class for visual inspection.
- The near-duplicate measurement uses a coarse hash and only within-class comparisons.

## 9. Remaining Work for This Section

| Item | Action |
|---|---|
| Notebook Cell 6 caption | Replace "proving the risk of CNN shortcut learning" with "consistent with the risk of shortcut learning; tested in Task 3.5" |
| Notebook markdown 1.1 | Remove "differ by fewer than 10–15 localized pixels" and the "hands ≈100, walls ≈145" statement |
| Notebook Cell 5 | State that the filter uses contrast (standard deviation), not sharpness |
| Notebook execution | Run **Restart and Run All** and save so outputs match the code |
| Hand position variation | Not yet measured; the required audit (bounding-box or skin-mask centroid spread per class) follows Task 2.2 masks |
| Per-class brightness | Not yet measured; compare brightness and background color per class to test the "varies slightly across folders" note |

## 10. Carry-Forward to Later Tasks

| Finding | Action |
|---|---|
| Balanced classes | Report macro metrics; no class weighting |
| Near-duplicate structure (76% have a neighbour within d ≤ 1) | Group-aware split (done, t = 0); measure residual leakage (`src/audit_leakage.py`) |
| Background may correlate with label | Crop (Task 2.2), background swap (Task 2.3), ablation (Task 3.5) |
| M/N/S, A/S/E, K/V/U confusion | Confusion matrix (Task 4.2), Grad-CAM (Task 4.3), review Model 2 downsampling |
| Brightness varies within class | Brightness augmentation (Task 2.3) |

## 11. Anticipated Questions

**Q1. Why not train directly on the 87,000 raw images?**
> The dataset is clean and balanced, so raw training would probably score well on a random split. Our concern is whether a random split would be meaningful: 76% of images have a near-identical neighbour in their own class at Hamming distance ≤ 1, so random splitting would leak near-duplicates into the test set. We group identical hashes into the same split and measure the residual leakage. We also test generalization on an external set of webcam photos rather than assuming it is large.

**Q2. Which letters will the model confuse most, and why?**
> We expect the fist group (M, N, S) and the thumb-placement group (A, S, E) to produce most errors, because their differences are small and centered on the thumb. We will confirm this with the confusion matrix. The M/N difference is whether the thumb sits under two or three fingers. A CNN with aggressive early pooling may lose that detail, so we will check how much downsampling Model 2 uses.

## References

[1] R. Geirhos et al., "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, 2(11), 665–673, 2020.
[2] K. He et al., "Deep Residual Learning for Image Recognition," CVPR, 2016.
[3] A. Howard et al., "Searching for MobileNetV3," ICCV, 2019.