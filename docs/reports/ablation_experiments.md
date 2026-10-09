# Controlled Ablation Study: Data Partitioning, Spatial Preprocessing, and Loss Regularization

## 1. Executive Summary

This study investigates three controlled ablation experiments designed to isolate the empirical impact of critical architectural and data pipeline decisions on American Sign Language (ASL) fingerspelling recognition:
1. **Ablation 1 (Data Split Leakage)**: Quantifies the optimistic bias induced by naive uniform random splitting versus subject-independent leave-one-signer-out (LOSO) partitioning.
2. **Ablation 2 (Aspect Ratio Preservation)**: Isolates the effect of neutral gray letterbox padding against direct spatial squashing on anatomical finger geometry.
3. **Ablation 3 (Label Smoothing Regularization)**: Evaluates soft target regularization ($\alpha=0.10$) against standard hard-target cross-entropy ($\alpha=0.0$) to measure generalization gain and logit calibration.

All experiments were executed on an AMD Radeon RX 6700 XT accelerator using the DirectML execution backend under matched random seeds, learning rates, and optimizer configurations.

---

## 2. Ablation 1: Data Split Leakage (Random Split vs. Subject-Independent Split)

### Experimental Protocol
- **Subject-Independent Split (LOSO)**: Signers P1–P7 for training (24,777 images), Signer P8 for validation (3,588 images), and Signers P9–P10 for testing (7,200 images). Zero signer identity or recording session overlap crosses partitions.
- **Random Split (Uniform Burst Leakage)**: Stratified 70/15/15 partition across all 35,988 images (25,190 train, 5,399 validation, 5,399 test). Identical video capture bursts from the same signers and sessions are uniformly scattered across train, val, and test partitions, with 118 exact duplicate frame hashes leaking directly across the split boundary.
- **Models Evaluated**: Model 1 (Baseline Simple CNN, 464K params) and Model 3 (Fine-Tuned ResNet-18, 11.19M params). Both models were trained for up to 15 epochs with early stopping patience of 8.

### Empirical Results

| Architecture | Subject-Independent Test Acc | Random Split Test Acc | Performance Delta (Leakage Gain) | Subject Macro-F1 | Random Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1: Baseline Simple CNN** | 52.71% (3,795 / 7,200) | **90.72%** (4,898 / 5,399) | **+38.01%** | 49.20% | **90.21%** |
| **Model 3: Fine-Tuned ResNet-18** | 91.86% (6,614 / 7,200) | **100.00%** (5,399 / 5,399) | **+8.14%** | 91.10% | **100.00%** |

### Findings and Mechanistic Analysis
1. **Pervasive Overfitting Under Random Splitting**: In Model 1, random splitting artificially inflates test accuracy from **52.71% to 90.72% (+38.01% gain)**. Without cross-signer partitioning, shallow convolutional filters memorize static signer background cues, forearm skin tone, and clothing textures that remain constant within a video burst session, masking severe architectural incapacity.
2. **Artificial Saturation Ceiling**: In Model 3, random splitting produces a deceptive **100.00% test accuracy across all 5,399 test images**. Because adjacent frames in the same video session differ only by sub-pixel camera jitter, the network performs temporal interpolation rather than cross-subject sign recognition.
3. **Scientific Validity Verdict**: Evaluating sign language models using naive random splitting creates a false sense of production readiness. True cross-subject generalization drops to 52.71% for Model 1 and 91.86% for Model 3 when evaluated on unseen human anatomies. Subject-independent splitting is essential for scientific integrity.

---

## 3. Ablation 2: Aspect Ratio Preservation (Letterbox Padding vs. Naive Squashing)

### Experimental Protocol
Hand bounding boxes isolated by the Region-of-Interest (ROI) detector have variable rectangular aspect ratios depending on whether a sign extends vertically (e.g., `B`, `D`, `F`, `1`) or horizontally (e.g., `G`, `H`, `C`).
- **Letterbox Padding (Canonical Pipeline)**: Hand bounding box is centered within a square canvas using neutral gray `(128, 128, 128)` padding before bilinear resizing to $224 \times 224$, preserving the natural aspect ratio of the fingers.
- **Naive Spatial Squashing (Ablated Pipeline)**: Bounding box is resized directly to $224 \times 224$ pixels, distorting the width-to-height ratio of the hand.
- **Evaluation Set**: The complete 7,200-image unseen test set (Signers P9 and P10) evaluated across both Model 1 and Model 3.

### Empirical Results

| Architecture | Letterbox Padding Accuracy | Naive Squashing Accuracy | Accuracy Degradation | Letterbox Macro-F1 | Naive Squashing Macro-F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1: Baseline Simple CNN** | 52.71% (3,795 / 7,200) | **23.04%** (1,659 / 7,200) | **-29.67%** | 49.20% | **20.84%** (-28.36%) |
| **Model 3: Fine-Tuned ResNet-18** | 91.86% (6,614 / 7,200) | **85.85%** (6,181 / 7,200) | **-6.01%** | 91.10% | **84.14%** (-6.96%) |

### Findings and Mechanistic Analysis
1. **Catastrophic Degradation in Shallow CNNs**: In Model 1, naive squashing destroys more than half of the model's predictive capability (**-29.67% accuracy drop**, plummeting to 23.04%). Without multi-scale residual features, shallow filters rely heavily on rigid spatial proportions; squashing tall signs into squares widens fingers and compresses vertical knuckles, rendering morphological differentiation impossible.
2. **Residual Feature Resilience**: Model 3 demonstrates stronger resilience, dropping only **-6.01% (91.86% $\to$ 85.85%)**. Deep residual hierarchies and pre-trained texture filters partially compensate for aspect distortion by recognizing local joint configurations and edge gradients.
3. **Operational Recommendation**: Letterbox padding with neutral gray fill is non-negotiable for sign language recognition. Preserving natural hand proportions prevents artificial morphological distortion and secures a +6.01% to +29.67% accuracy advantage across model families.

---

## 4. Ablation 3: Loss Function Regularization (Label Smoothing vs. Standard Cross-Entropy)

### Experimental Protocol
To evaluate target distribution regularization, Model 3 (ResNet-18) was trained under identical random seeds, learning rates ($1.0\times 10^{-4}$ backbone, $5.0\times 10^{-4}$ head), optimizer schedules, and early stopping patience (8 epochs):
- **Condition A (Standard Cross-Entropy, $\alpha = 0.0$)**: Hard one-hot ground-truth target vectors $q_k = \mathbb{I}(y = k)$.
- **Condition B (Label Smoothing, $\alpha = 0.10$)**: Soft ground-truth target distributions $q_k = (1 - \alpha) \cdot \mathbb{I}(y = k) + \frac{\alpha}{36}$.
- **Metrics Evaluated**: Top-1 Accuracy, Macro-Precision, Macro-Recall, Macro-F1, and Mean Softmax Confidence ($\frac{1}{N} \sum_{i=1}^N \max_k p_k$).

### Empirical Results

| Loss Configuration | Unseen Test Top-1 Acc | Macro-Precision | Macro-Recall | Macro-F1 | Mean Softmax Confidence | Correct / Total |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Standard Cross-Entropy ($\alpha = 0.00$)** | 91.18% | 92.63% | 91.18% | 90.77% | **93.70%** (Overconfident) | 6,565 / 7,200 |
| **Label Smoothing ($\alpha = 0.10$)** | **91.86%** | **93.12%** | **91.86%** | **91.10%** | **70.54%** (Calibrated) | **6,614 / 7,200** |
| **Regularization Delta** | **+0.68%** | **+0.49%** | **+0.68%** | **+0.33%** | **-23.15% confidence drop** | **+49 samples** |

### Findings and Mechanistic Analysis
1. **Marginal Accuracy Improvement**: Label smoothing provides a modest gain of **+0.68% in Top-1 Accuracy** (49 additional test samples classified correctly) and +0.33% in Macro-F1.
2. **Mitigation of Logit Overconfidence**: Under standard cross-entropy, the network forces logits toward infinite values, producing an average maximum softmax confidence of **93.70%**. When making errors on ambiguous Category A homographs (`W` vs `6`), hard-target cross-entropy outputs overconfident probabilities exceeding 99%.
3. **Probabilistic Calibration**: Label smoothing prevents the largest logit from dominating all other classes, lowering the average maximum softmax confidence to **70.54% (-23.15% drop)**. The resulting probability distributions reflect genuine morphological uncertainty on fine-grained knuckle curl boundaries, providing calibrated confidence scores suitable for downstream uncertainty thresholding in production.

---

## 5. Synthesis and Consolidated Recommendations

| Decision Area | Investigated Options | Selected Standard | Justification / Impact |
| :--- | :--- | :--- | :--- |
| **Dataset Partitioning** | Random Split vs Subject Split | **Subject-Independent Split (LOSO)** | Random splitting inflates accuracy by +8.14% to +38.01% through burst leakage; subject split measures true cross-signer generalization. |
| **Image Preprocessing** | Letterbox Pad vs Naive Squash | **Letterbox Padding (Neutral Gray)** | Prevents artificial finger aspect distortion; delivers +6.01% (ResNet-18) to +29.67% (Simple CNN) accuracy preservation. |
| **Loss Formulation** | Hard CE vs Label Smoothing | **Label Smoothing ($\alpha = 0.10$)** | Improves accuracy by +0.68% and suppresses logit overconfidence by -23.15%, aligning output probabilities with morphological ambiguity. |
