# Transfer Learning Model Architecture: Pretrained Vision Backbone Adaptation

## 1. Overview and Design Motivation

While custom convolutional architectures trained from scratch learn feature hierarchies directly from the target dataset, they are fundamentally bounded by the volume and visual diversity of the training signers. In subject-independent American Sign Language (ASL) fingerspelling recognition, cross-signer generalization requires robustness against anatomical variations (finger thickness, palm geometry), skin tone differences, and ambient illumination.

Model 3 leverages **Transfer Learning with Differential Fine-Tuning** based on a deep vision backbone (**ResNet-18**) pretrained on ImageNet-1k (1.28 million natural images). 

Key design advantages:
1. **Pretrained Visual Inductive Bias**: Early and intermediate convolutional filters transfer universal edge, contour, gradient, and surface texture detectors, providing rich geometric priors before training on hand crops.
2. **Convex Projection Head**: Directly mapping the 512-dimensional pooled feature embedding to 36 fingerspelling classes via a linear projection with dropout regularization preserves the global geometry of ImageNet feature manifolds without introducing uncalibrated intermediate layers.
3. **Differential Learning Rate Optimization**: Training the pretrained backbone at a conservative learning rate ($\eta_{backbone} = 1.0 \times 10^{-4}$) while updating the classification head at a higher rate ($\eta_{head} = 5.0 \times 10^{-4}$) prevents catastrophic forgetting of generic visual features while accelerating task-specific decision boundary convergence.

---

## 2. Mathematical Formulation

Let $f_{\theta}(x)$ denote the pretrained backbone feature extractor parameterized by weights $\theta \in \mathbb{R}^P$, mapping an input image $x \in \mathbb{R}^{3 \times 224 \times 224}$ to a pooled latent embedding vector:
$$h = \text{AdaptiveAvgPool2d}(f_{\theta}(x)) \in \mathbb{R}^{512}$$

The adaptation classification head computes class logits $z \in \mathbb{R}^{36}$ through stochastic dropout and affine transformation:
$$z = W \cdot \text{Dropout}(h, p=0.3) + b$$
where $W \in \mathbb{R}^{36 \times 512}$ and $b \in \mathbb{R}^{36}$ represent the linear projection weights and bias.

During optimization, the parameter update vector $\Delta \Theta$ is partitioned across parameter groups:
$$\theta^{(t+1)} = \theta^{(t)} - \eta_{backbone}^{(t)} \cdot \left(\nabla_{\theta} \mathcal{L} + \lambda \theta^{(t)}\right)$$
$$W^{(t+1)} = W^{(t)} - \eta_{head}^{(t)} \cdot \left(\nabla_{W} \mathcal{L} + \lambda W^{(t)}\right)$$
with cosine decay schedules:
$$\eta_g^{(t)} = \eta_{min} + \frac{1}{2}(\eta_{g}^{(0)} - \eta_{min})\left(1 + \cos\left(\frac{t}{T_{max}}\pi\right)\right)$$

---

## 3. Detailed Architecture Specifications

| Stage / Component | Sub-Modules / Layers | Kernel / Stride | Input Dimensions | Output Dimensions | Parameters | Pretrained Source |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | RGB Tensor | — | — | $3 \times 224 \times 224$ | 0 | Input Pipeline |
| **Stem** | `Conv2d` + `BN` + `ReLU` + `MaxPool` | $7\times 7, s=2, p=3$ / $3\times 3, s=2, p=1$ | $3 \times 224 \times 224$ | $64 \times 56 \times 56$ | 9,920 | ImageNet-1k |
| **Stage 1 (Layer 1)** | 2 Residual Blocks (4 $\times$ Conv3x3) | $3\times 3, s=1, p=1$ | $64 \times 56 \times 56$ | $64 \times 56 \times 56$ | 147,968 | ImageNet-1k |
| **Stage 2 (Layer 2)** | 2 Residual Blocks (4 $\times$ Conv3x3 + 1x1 Proj) | $3\times 3, s=2, p=1$ | $64 \times 56 \times 56$ | $128 \times 28 \times 28$ | 525,568 | ImageNet-1k |
| **Stage 3 (Layer 3)** | 2 Residual Blocks (4 $\times$ Conv3x3 + 1x1 Proj) | $3\times 3, s=2, p=1$ | $128 \times 28 \times 28$ | $256 \times 14 \times 14$ | 2,100,224 | ImageNet-1k |
| **Stage 4 (Layer 4)** | 2 Residual Blocks (4 $\times$ Conv3x3 + 1x1 Proj) | $3\times 3, s=2, p=1$ | $256 \times 14 \times 14$ | $512 \times 7 \times 7$ | 8,394,752 | ImageNet-1k |
| **Global Pool** | `AdaptiveAvgPool2d` | target $(1 \times 1)$ | $512 \times 7 \times 7$ | $512 \times 1 \times 1$ | 0 | Deterministic |
| **Classifier Head** | Flatten + Dropout(0.3) + Linear(512, 36) | — | 512 | 36 | 18,468 | Initialized Scratch |

- **Pretrained Feature Extractor Parameters**: 11,176,512
- **Adaptation Head Parameters**: 18,468
- **Total Trainable Parameters**: **11,194,980** (~11.19M parameters)

---

## 4. Training Protocol and Hyperparameters

- **Data Split**: Subject-Independent Split (Train: Signers P1–P7, Val: Signer P8, Test: Signers P9–P10).
- **Batch Size**: 64 samples per step.
- **Head Learning Rate ($\eta_{head}$)**: $5.0 \times 10^{-4}$.
- **Backbone Learning Rate ($\eta_{backbone}$)**: $1.0 \times 10^{-4}$.
- **Optimizer**: AdamW (`weight_decay=1e-4`, `foreach=False`).
- **Warmup Schedule**: 2 epochs linear warmup to initial LR, followed by Cosine Annealing to $\eta_{min} = 1.0 \times 10^{-6}$.
- **Loss Function**: Label Smoothing Cross Entropy ($\alpha = 0.10$).
- **Regularization**: Spatial affine jitter, colour jitter, random horizontal flip ($p=0.5$), and classification dropout ($p=0.3$).
- **Convergence**: Trained up to 20 epochs; early stopping triggered after Epoch 18 with best validation performance at Epoch 10.
- **Execution Hardware**: DirectML backend (`privateuseone:0`, AMD Radeon RX 6700 XT), training throughput ~205 samples/sec (~115s/epoch).

---

## 5. Test Set Evaluation on Unseen Signers (P9 & P10)

Evaluated strictly on the held-out subject test set manifest (`data/splits/subject_test.csv`, 7,200 images):

| Metric | Model 1 (Simple CNN Scratch) | Model 2 (Complex Block CNN Scratch) | Model 3 (Transfer ResNet-18 Fine-Tuning) | Delta vs Model 2 | Delta vs Model 1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top-1 Accuracy** | 52.71% | 82.90% | **91.86%** | **+8.96%** | **+39.15%** |
| **Macro-Precision**| 54.47% | 87.34% | **93.12%** | **+5.78%** | **+38.65%** |
| **Macro-Recall**   | 52.71% | 82.90% | **91.86%** | **+8.96%** | **+39.15%** |
| **Macro-F1 Score** | 49.20% | 81.78% | **91.10%** | **+9.32%** | **+41.90%** |
| **Test Loss**      | 1.9444 | 0.8387 | **0.9257** | +0.0870 | -1.0187 |

---

## 6. Error and Confusion Analysis

Out of 7,200 unseen test samples, the model produced only 586 classification errors (8.14% error rate):

- **24 out of 36 classes** achieved **100.0% precision, 100.0% recall, and 1.00 F1 score** across all test images (`0, 3, 4, 5, 7, 8, 9, A, B, C, D, E, F, G, H, K, L, O, P, Q, S, U, X, Y`).
- **536 of the 586 total errors (91.5%)** stemmed exclusively from 5 static homograph pairs:
  1. **Class 6 $\to$ W** (182 errors): Both signs exhibit 3 vertical fingers with thumb-pinky contact vs spread; without temporal movement, boundary contact is ambiguous under low-contrast gray backgrounds.
  2. **Class 1 $\to$ Z** (108 errors): 'Z' is articulated by tracing a static index finger gesture through the air; isolated static camera frames are visually identical to the index-up '1' posture.
  3. **Class I $\to$ J** (100 errors): 'J' is a dynamic motion sign formed from the pinky-up 'I' posture; single frames lack rotational velocity information.
  4. **Class N $\to$ M** (81 errors): Differ solely by whether the thumb is tucked between ring/pinky or middle/ring fingers.
  5. **Class R $\to$ U** (65 errors): Differ solely by crossed fingers vs parallel aligned fingers.

These error patterns validate that Model 3 has converged near the empirical Bayes error rate for static 2D vision on ASL fingerspelling.
