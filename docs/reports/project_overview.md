# Project Executive Summary & Technical Synthesis

## 1. Problem Description (Course Project Section I)

American Sign Language (ASL) fingerspelling is a core component of manual sign communication, translating orthographic characters into standardized handshapes. Building an automated, vision-based recognition system for ASL fingerspelling involves several technical challenges:

![Vision Challenge Pipeline](../figures/architecture/vision_challenge_pipeline.png)
*Figure 1: Core vision pipeline challenges in real-time ASL fingerspelling recognition.*

1. **High Inter-Class Similarity**: Many alphanumeric signs share nearly identical overall hand contours, differing only in subtle finger contacts or thumb placements (e.g., '6' vs. 'W', 'K' vs. 'V', 'N' vs. 'M').
2. **Loss of 3D Information**: A single 2D camera projects complex 3D hand shapes onto a flat plane, causing depth ambiguities (homographs) and self-occlusions.
3. **Anatomical and Signer Variations**: Hand sizes, skin tones, joint flexibility, and personal signing habits vary widely across individuals. Models frequently latch onto signer-specific shortcuts (backgrounds, skin tone) rather than true sign geometry.
4. **Dominant Hand Invariance**: Natural signers use either their left or right hand. A practical recognition system must handle both seamlessly without requiring separate models or duplicated training sets.
5. **Real-Time Efficiency**: Downstream assistive communication tools require low-latency inference on standard consumer hardware.

This project designs, trains, and evaluates an end-to-end deep learning pipeline that addresses these challenges through disciplined preprocessing, subject-independent validation, and progressive neural network architectures.

---

## 2. Dataset Description & Partitioning (Course Project Section II)

### Primary Training Corpus: ASL-HG

The primary corpus is the American Sign Language Hand Gesture (ASL-HG) dataset, hosted on Mendeley Data:
- **Scope**: 36 alphanumeric classes, including digits `0`–`9` and English alphabet letters `A`–`Z`.
- **Raw Volume**: 36,000 images (exactly 1,000 images per class across 10 human signers, `P1` through `P10`).
- **Nominal Resolution**: Uniform $300 \times 300$ RGB color images.
- **Recording Context**: Captured as continuous video bursts where each signer performs a sign under fixed camera angles and lighting.

### Data Cleaning and Partitioning Summary

To prevent burst-frame data leakage, the dataset is split strictly by signer identity:

| Split Partition | Signer Assignment | Raw Nominal Count | Cleaned Manifest Count | Rationale & Processing |
| :--- | :--- | :--- | :--- | :--- |
| **Train (70%)** | Signers P1–P7 | 25,200 | **24,777** | 423 exact byte-duplicate burst copies removed to avoid loss distortion. |
| **Validation (10%)** | Signer P8 | 3,600 | **3,588** | 12 corrupted blank camera frames in class 'H' pruned. |
| **Test (20%)** | Signers P9, P10 | 7,200 | **7,200** | Kept intact; exactly 200 images per class across two unseen signers. |
| **Total** | P1–P10 | **36,000** | **35,565** | Cleaned dataset universe used for model training and evaluation. |

### Preprocessing and Augmentation Standards

1. **MediaPipe Hand Localization**: 21-landmark hand detection with an adaptive 20% margin expansion achieves 99.86% coverage on non-blank frames, removing background clutter while preserving fingertips. Full-frame preservation is applied to Class '0' to avoid cropping the two-handed gesture.
2. **Aspect-Ratio Letterboxing**: Hand crops are scaled along their longest dimension to 224 pixels and padded with neutral gray `(128, 128, 128)`, avoiding the spatial distortion of naive squashing.
3. **Runtime Ambidexterity**: Training uses `RandomHorizontalFlip(p=0.5)` to reflect images on the fly, conferring left- and right-hand invariance dynamically without duplicating 36,000 files on disk.

---

## 3. Progressive Neural Architectures

We designed and evaluated three architectures representing progressive model complexity:

![Model Progression](../figures/architecture/model_progression.png)
*Figure 2: Three-tier model progression from baseline CNN to pretrained transfer backbone.*

### Comparative Architecture and Benchmark Results

All models were evaluated on the 7,200-image unseen test set (Signers P9 and P10) across all 36 classes:

| Metric / Dimension | Model 1: Baseline Simple CNN | Model 2: Deep Residual CNN | Model 3: Fine-Tuned ResNet-18 |
| :--- | :--- | :--- | :--- |
| **Model Type** | Sequential feed-forward CNN | Deep CNN with residual skip blocks | Transfer learning with pretrained vision backbone |
| **Trainable Parameters** | 463,940 (~464K) | 2,907,588 (~2.91M) | 11,194,980 (~11.19M) |
| **Conv Layers** | 4 convolutional layers | 16 convolutional layers | 17 convolutional layers + stem |
| **Receptive Field** | 46 px (spatial) | 216 px (spatial) | 435 px (spatial) |
| **Optimization** | AdamW (`weight_decay=5e-4`) | AdamW (`weight_decay=1.5e-4`) | AdamW (`weight_decay=1e-4`, differential LR) |
| **Loss Formulation** | Cross-Entropy ($\alpha = 0.10$ smoothing) | Cross-Entropy ($\alpha = 0.05$ smoothing) | Cross-Entropy ($\alpha = 0.10$ smoothing) |
| **Top-1 Test Accuracy** | **52.71%** (3,795 / 7,200) | **82.90%** (5,969 / 7,200) | **91.86%** (6,614 / 7,200) |
| **Macro Precision** | 54.47% | 87.34% | **93.12%** |
| **Macro Recall** | 52.71% | 82.90% | **91.86%** |
| **Macro F1-Score** | 49.20% | 81.78% | **91.10%** |
| **DirectML Throughput** | ~360 images/second | ~280 images/second | ~205 images/second |

---

## 4. Key Experimental Findings

### 1. Controlled Ablations ([`ablation_experiments.md`](ablation_experiments.md))
- **Data Split Leakage**: Uniform random splitting artificially inflates test accuracy from 52.71% to 90.72% (+38.01%) on Model 1 and to 100.00% on Model 3, caused by near-identical burst frames leaking across splits. Subject-independent splitting is necessary for reliable evaluation.
- **Aspect Ratio Preservation**: Letterboxing delivers a **+29.67% accuracy preservation** over naive resizing on Model 1 (52.71% vs. 23.04%) and **+6.01%** on Model 3 (91.86% vs. 85.85%), confirming that preserving finger aspect ratios prevents artificial distortion.
- **Label Smoothing**: Soft-target regularization ($\alpha = 0.10$) improves Top-1 accuracy by +0.68% while reducing overconfident softmax outputs from 93.70% down to a well-calibrated 70.54%.

### 2. Error Diagnostics ([`error_diagnostics.md`](error_diagnostics.md))
- On Model 3, 23 of 36 classes achieve 100% precision and recall.
- Remaining errors concentrate in five classes (`6`, `1`, `I`, `N`, `R`), accounting for 82.9% of misclassifications.
- Failures trace to physical causes: static 2D planar homographs (e.g., $6 \leftrightarrow W$), motion loss in dynamic signs ($1 \leftrightarrow Z$, $I \leftrightarrow J$), and inter-finger occlusions ($N \leftrightarrow M$).

### 3. Real-World External Benchmark ([`external_benchmark.md`](external_benchmark.md))
- Evaluated on 180 out-of-distribution smartphone and webcam photos across 4 unseen participants.
- Model 3 achieved **80.00% zero-shot accuracy** (vs. 28.33% for Model 1 and 61.11% for Model 2).
- Paired left-versus-right testing confirmed handedness parity deltas between **0.00% and 8.33%**, verifying that runtime flipping provides effective ambidexterity in practice.

### 4. Interpretability with Grad-CAM ([`model_interpretability.md`](model_interpretability.md))
- Gradient-weighted class activation mapping confirms that Model 2 and Model 3 focus tightly on active fingertips, knuckle folds, and thumb positions.
- In contrast, Model 1 spreads activation diffusely across the palm, explaining its vulnerability to handshape confusion.

---

## 5. Hardware Acceleration

All models were trained and benchmarked locally on an AMD Radeon RX 6700 XT (12GB VRAM) paired with an Intel Core i5-12600K and 32GB RAM on Windows 11. 

By leveraging PyTorch with the Microsoft DirectML backend (`torch-directml`), training ran at sustained speeds of 205-360 images per second, completing full training runs in minutes without GPU memory bottlenecks.