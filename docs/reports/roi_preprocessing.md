# Task 2.2: Hand ROI Detection, Crop, and Letterbox Standardisation (ASL-HG)

**Status:** Complete. All gates pass. Crop output is ready for Task 2.3.
**Date:** 2026-10-05
**Consolidation reference:** PROJECT_CONSOLIDATION.md, Sections 4, 8, 9
**Scripts:** `src/roi_detector.py` (shared), `src/hand_roi_audit.py` (Part A), `src/blank_scan.py` (Part B-1), `src/preprocess_roi.py` (Part B-2), `src/verify_crops.py` (Part C)

Labels: **[Measured]** = computed by our scripts on `data/raw/`. **[Expected]** = roadmap target.

---

## Summary

Each raw image is either written as a 224×224 letterboxed PNG crop or excluded as blank. 35,988 crops are written from 36,000 raw images, and 12 blank frames are excluded. The hand is found by MediaPipe (single hand) for 34,980 images (97.2%) and cropped with a 20% margin. Class 0, the two-handed sign, is kept as the full frame by a fixed class rule (1,000 images), because a single-hand box clips the second hand. Eight non-zero images where MediaPipe failed are kept as the full frame.

The main limitation is a scale confound. The letterbox scale (224 divided by the margin box side) varies from 0.83 to 1.33 across signs, a spread of about 1.6×. Because the crop follows the detected hand, the hand's apparent size partly encodes the sign. Class 0 is at the low end of this range (0.747). Task 2.3 must widen scale augmentation to cover the observed spread.

---

## Part A: Detection feasibility

### Relevance (deep learning)

The crop must keep the hand and remove background. A detector that fails silently, or that boxes the wrong region, would make the crop look valid while removing the signal. Part A measures how often each detector succeeds before we choose the pipeline.

### Definitions

- **MediaPipe Hands (single hand):** a landmark model that returns 21 points for the most prominent hand. The box is the min and max of those points.
- **Skin fallback:** a YCrCb Cr-channel Otsu threshold followed by the largest contour. Accepted only if its area lies within [2%, 65%] of the frame and it does not touch the border.
- **Border touch:** a box within 2 px of the frame edge. The hand is cut by the frame, so the margin cannot restore the missing part.

### Objective

Measure detector coverage and border clipping on a stratified sample, and identify frames where no hand is found.

### Methodology

720 images: 2 per volunteer per class, seed 42. Each image goes to MediaPipe, then to the skin fallback, then to `none`.

### Code output analysis [Measured]

| Method | Count | Rate |
|---|---|---|
| MediaPipe | 719 | 99.86% |
| Skin fallback | 0 | 0% |
| None | 1 | 0.14% (`H/P8_H_784`) |

Border touching among MediaPipe boxes: 0.83%. Box side (5th, median, 95th percentile): 111, 171, 219 px.

The one `none` frame is a blank frame, which Part B-1 handles. The skin fallback never triggered. In this dataset MediaPipe either finds the hand or the frame has no hand, so the fallback adds no coverage. It is kept in the code for transparency.

### Conclusion

MediaPipe is the detector. Single-hand mode is correct for 35 of 36 classes. Class 0 needs a rule (see Part B-2).

---

## Part B-1: Blank frame scan

### Relevance (deep learning)

A blank frame carries no sign. If it is cropped, the crop is an empty background with a random scale, which teaches the network that some background patches are valid samples. The blank frames must be removed before training.

### Definitions and formulas

For each image, convert to grayscale, apply a 5×5 Gaussian blur, and compute the Sobel gradient magnitude $|\nabla I| = \sqrt{G_x^2 + G_y^2}$. The edge fraction at magnitude $t$ is

$$
e_t = \frac{1}{HW}\sum_{i,j} \mathbb{1}\left[\,|\nabla I|_{ij} > t\,\right]
$$

A plain background has $e_{40} \approx 0$. A hand, even at the frame edge, produces many edges.

### Objective

Find the frames with no hand, using a threshold chosen from the data rather than guessed.

### Methodology

All 36,000 images are scored with $e_{40}$. The scores are ranked, the gap structure is inspected, and a threshold is chosen. The chosen threshold is then checked against MediaPipe on the candidates.

### Code output analysis [Measured]

The lowest 11 frames have $e_{40} \le 0.000033$. The 12th is at $0.0123$, a ratio of about 369× to the next value. The first 12 frames are all class H, volunteer P8. The threshold $T = 0.001$ selects exactly these 12.

| Frame set | Count | Contents |
|---|---|---|
| Blank ($e_{40} \le 0.001$) | 12 | `H/P8_H_778`, `780`–`789`, `791` |
| Partial entry (next 10 ranks) | 10 shown | fingertip or knuckles at the bottom edge |

MediaPipe was run on the 13 frames with $e_{40} \le 0.02$. It found a hand in 0 of them. The 13th frame, `P8_H_792`, is a partial entry and is kept.

The 12 blank frames match the sequence visible in the file browser. Partial-entry frames (for example `P8_H_777`, `P8_H_793`–`800`) are not blank, so they are not excluded.

### Conclusion

12 blank frames are excluded (D18). The gap in the ranking is large, so the threshold is not sensitive to its exact value between 0.00004 and 0.012.

---

## Part B-2: Crop, margin, letterbox

### Relevance (deep learning)

The network needs a consistent input format. A tight crop around the hand keeps the sign and removes background, but changes the scale of the hand in the input. The margin protects fingertips and wrist boundaries. Letterboxing keeps the aspect ratio, so a wide hand is not stretched.

### Definitions and formulas

Margin: $m = \lfloor 0.20 \cdot s + 0.5 \rfloor$, where $s$ is the longer side of the detected box. The margin box is the detected box expanded by $m$ on each side and clamped to the frame.

Letterbox scale: $\sigma = 224 / \max(w, h)$, where $w, h$ are the margin box dimensions. The content is resized by $\sigma$ and centred on a 224×224 canvas filled with $(128, 128, 128)$.

Scale $\sigma$ is the resampling factor. A value below 1 means the hand is downsampled. A value above 1 means it is upsampled, which softens the image.

### Objective

Produce a 224×224 PNG for every non-blank image, with no squashing and exact neutral padding, and record the method, box, and scale for every image.

### Methodology

Each image is assigned one method label:

| Label | Meaning | Count |
|---|---|---|
| `mediapipe` | single-hand box with 20% margin | 34,980 |
| `class_rule_full` | class 0 only: full frame by fixed rule (D19) | 1,000 |
| `undetected_full` | non-zero class, detector failed: full frame kept (D20) | 8 |
| `blank` | excluded (Part B-1) | 12 |

The labels separate a fixed rule from a detection failure, so the report does not conflate them.

The crops are written with bilinear resampling and saved as lossless PNG. The stale-output guard refuses to run if old files remain.

### Code output analysis [Measured]

| Quantity | Value |
|---|---|
| Images processed | 36,000 |
| Written PNGs | 35,988 |
| Errors | 0 |
| Border-touching rate among MediaPipe boxes | 0.77% (268 of 34,980) |
| MediaPipe crops enlarged more than 2× | 1 (class H) |

**Undetected non-zero frames (8):** `H/P8_H_792, 794, 796, 797, 798` and `R/P3_R_255, 256, 283`. The R/P3 frames show a clear hand in backlit light, which MediaPipe missed. These are detection failures, not blanks.

**Scale by class [Measured]** (median of $\sigma$ over the class's written images; smaller = hand drawn smaller):

| Range | Classes |
|---|---|
| Smallest (hand appears smaller) | 0 (0.747, rule), B (0.830), D (0.830), F (0.839), U (0.849), W (0.852) |
| Largest (hand appears larger) | S (1.306), M (1.325), T (1.295), X (1.266), N (1.224), Q (1.208), A (1.191), E (1.185) |

Overall: class 0 median 0.747, other classes median 0.941, ratio 0.793. Across all 36 classes the spread is 0.830 to 1.325, a factor of about 1.6.

### Conclusion

The crop geometry is correct. The output is ready for Task 2.3. The scale spread is a real property of the crop. It is not a bug, but it must be handled as a training issue (D21).

---

## Part C: Verification

### Relevance (deep learning)

A silent error at the crop stage, such as a wrong label, a missing file, or a padding error, would corrupt every later result. Part C checks the output independently of the code that wrote it.

### Definitions

- **Letterbox geometry check:** every pixel outside the placed content rectangle must equal $(128, 128, 128)$ exactly, and the content must not be uniform padding.
- **Method rule check:** `class_rule_full` rows must be exactly the class-0 rows.

### Objective

Confirm that every written file is correct, that the method labels follow the rules, and that the scale is measured for every class.

### Code output analysis [Measured]

| Gate | Result |
|---|---|
| 36,000 log rows | PASS |
| No error rows | PASS |
| Written files equal PNGs on disk (35,988) | PASS |
| No missing or extra files | PASS |
| No blank frame on disk | PASS |
| Per-class counts equal 1,000 minus blanks | PASS (H = 988) |
| All files 224×224 uint8 RGB | PASS (0 failures) |
| All padding exact | PASS (0 non-exact, 0 uniform) |
| Rule applies only to class 0 | PASS |
| Every class 0 row uses the rule | PASS |

Verdict: **PASSED**.

Visual review (`docs/figures/`):
- `crop_review_class0.png`: both hands complete in all 10 volunteer samples.
- `crop_review_full_frames.png`: 8 undetected frames (5 H/P8, 3 R/P3) and one class-0 crop per volunteer.
- `crop_review_wide_poses.png`: fingertips of 5 and W intact; class 0 P9 complete.
- `crop_review_lowest_edge_nonblank.png`: partial-entry frames are fingertips at the edge. Several MediaPipe crops are upscaled and soft.

### Conclusion

The output is structurally and geometrically correct. The visual review confirms the class-0 rule and the wide poses. The partial-entry frames are weak samples and are recorded below.

---

## Limitations

1. **Scale confound across signs (main).** Letterbox scale varies 0.83 to 1.33 across classes (about 1.6×). The crop removes absolute hand size, but the hand's size relative to the crop still varies by sign. A CNN can exploit this. Mitigation belongs in Task 2.3 (widen scale augmentation) and Task 3.5 (ablation). Task 2.3 must not use the roadmap's scale range of 0.94–1.06, which covers only about 12% of the spread.
2. **Class 0 uses the full frame (D19).** Its scale is 0.747, the lowest of any class, and its hands are drawn smaller than most other signs. The full frame keeps both hands, so the rule is needed.
3. **Undetected non-zero frames** (8 total) use the full frame and therefore share the class-0 scale. The count is negligible, but they are labelled `undetected_full` so they can be excluded in an ablation.
4. **Partial-entry frames** (about 13 non-blank H/P8 frames, one with an upscale over 2×). They are kept, flagged in the log, and documented. They are weak samples, and training results on class H for volunteer P8 should be read with this in mind.
5. **Single-hand MediaPipe.** Verified visually for class 0 and on the wide-pose set. The single-hand assumption is not verified on all 36,000 images. Only class 0 is known to be two-handed.
6. **Single detector.** The skin fallback never triggered, so its code path is untested on this dataset.
7. **Backlit frames** (R/P3) are missed by MediaPipe. Only 3 frames are affected.

## Decisions made in this task

| # | Decision | Reason | Status |
|---|---|---|---|
| D15 | Hand-box margin: 20% of the box's longer side | User decision; the range of the roadmap (15–20%) covers it, and the visual check shows no clipped fingertips | Active |
| D17 | Crops are lossless PNG, original stem | JPEG re-encoding breaks exact padding checks | Active |
| D18 | 12 blank frames excluded (`H/P8`, threshold $e_{40} \le 0.001$) | Clear gap in the ranking; visually blank | Active |
| D19 | Class 0 uses the full frame by fixed rule | Single-hand box clips the second hand; no two-hand logic | Active; scale limitation recorded |
| D20 | No centre-crop fallback. Undetected non-zero frames keep the full frame (`undetected_full`) | A centre crop invents a box with no hand | Active |
| D21 | Scale is a nuisance variable across signs. Task 2.3 widens the scale augmentation; Task 3.5 ablates it. | Measured spread 0.83–1.33 (about 1.6×); roadmap range covers only about 12% of it | Active; **Task 2.3 requirement** |

## Files produced

| File | Purpose |
|---|---|
| `src/roi_detector.py` | Shared single-hand detector (MediaPipe, skin fallback) |
| `src/hand_roi_audit.py` | Part A detection audit |
| `src/blank_scan.py` | Part B-1 blank scan |
| `src/preprocess_roi.py` | Part B-2 crop, margin, letterbox |
| `src/verify_crops.py` | Part C verification and review figures |
| `data/processed/blank_scan.csv` | Per-image edge fractions (cache) |
| `data/processed/blank_frames.csv` | 12 blank frames |
| `data/processed/hand_roi_sample.csv` | Part A sample results |
| `data/processed/roi_crop_log.csv` | Per-image method, box, margin, scale |
| `data/processed_crops/` | 35,988 PNG crops (not committed) |
| `docs/task2_2_partA_detection.json` | Part A report |
| `docs/task2_2_partB1_blank_scan.json` | Part B-1 report |
| `docs/task2_2_partB_crop.json` | Part B-2 report |
| `docs/task2_2_partC_verify.json` | Part C report |
| `docs/figures/diagnostics/blank_scan_*.png` | Blank scan figures |
| `docs/figures/dataset/roi_crop_inspection.png` | Crop overview |
| `docs/figures/diagnostics/crop_review_*.png` | Part C review figures |

Reproduction, from the repository root, in this order:
```
uv run python src/hand_roi_audit.py
uv run python src/blank_scan.py --threshold 0.001 --mediapipe-check
uv run python src/preprocess_roi.py      (after Remove-Item -Recurse -Force data\processed_crops)
uv run python src/verify_crops.py
```

## References

[1] R. Geirhos et al., "Shortcut learning in deep neural networks," *Nature Machine Intelligence*, 2(11), 665–673, 2020.