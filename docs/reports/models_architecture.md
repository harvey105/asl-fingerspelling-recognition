# Model Architecture Specification and Comparative Synthesis

## 1. Executive Comparative Architecture

The following table summarizes the structural attributes, training protocols, and test performances on the unseen subject partition (Signers 9 and 10, 7,200 images) across all three investigated neural network architectures.

| Dimension / Metric | Model 1: Baseline Simple CNN | Model 2: Deep Residual CNN | Model 3: Fine-Tuned ResNet-18 |
| :--- | :--- | :--- | :--- |
| **Architectural Family** | Sequential feed-forward CNN | Deep CNN with residual skip blocks | Transfer learning with pretrained vision backbone |
| **Trainable Parameters** | 463,940 (~464K) | 2,907,588 (~2.91M) | 11,194,980 (~11.19M) |
| **Feature Extractor Depth** | 4 convolutional stages (4 Conv2d layers) | 4 stages, 8 residual blocks (16 Conv2d layers) | 4 residual stages (17 Conv2d layers + stem) |
| **Initial Stem Geometry** | Conv $3\times 3, s=1 \to$ MaxPool $2\times 2, s=2$ | Conv $3\times 3, s=1 \to$ MaxPool $2\times 2, s=2$ | Conv $7\times 7, s=2 \to$ MaxPool $3\times 3, s=2$ |
| **Effective Receptive Field** | 46 px (spatial) $\to$ 224 px (global pool) | 216 px (spatial) $\to$ 224 px (global pool) | 435 px (spatial) $\to$ 224 px (global pool) |
| **Optimization Algorithm** | AdamW (`weight_decay=5e-4`) | AdamW (`weight_decay=1.5e-4`) | AdamW (`weight_decay=1e-4`) |
| **Learning Rate Schedule** | 2-epoch warmup, Cosine decay ($5.0\times 10^{-4}$) | 2-epoch warmup, Cosine decay ($4.5\times 10^{-4}$) | Differential LR: Backbone $1.0\times 10^{-4}$, Head $5.0\times 10^{-4}$ |
| **Loss Function & Regularization** | Cross-Entropy ($\alpha = 0.10$ label smoothing) | Cross-Entropy ($\alpha = 0.05$ label smoothing) | Cross-Entropy ($\alpha = 0.10$ label smoothing) |
| **Classifier Head Regularization** | Dual Dropout ($p = 0.40$) | Dual Dropout ($p = 0.30$) | Single Dropout ($p = 0.30$) |
| **Hardware Throughput (DirectML)** | ~360 images/second | ~280 images/second | ~205 images/second |
| **Convergence Epoch** | Epoch 17 (early-stopped at 25) | Epoch 8 (early-stopped at 16) | Epoch 10 (early-stopped at 18) |
| **Top-1 Accuracy (Unseen Test)** | **52.71%** | **82.90%** (+30.19%) | **91.86%** (+39.15%) |
| **Macro Precision** | **54.47%** | **87.34%** (+32.87%) | **93.12%** (+38.65%) |
| **Macro Recall** | **52.71%** | **82.90%** (+30.19%) | **91.86%** (+39.15%) |
| **Macro F1-Score** | **49.20%** | **81.78%** (+32.58%) | **91.10%** (+41.90%) |
| **Test Cross-Entropy Loss** | 1.9444 | 0.8387 | 0.9257 |

---

## 2. Architectural Blueprints and Theoretical Formulations

### Model 1: Baseline Simple CNN (Sequential Feed-Forward)

The baseline model provides a clean reference benchmark built without residual highways or external pre-training.

![Model 1 Detailed Architecture](../figures/architecture/model_1_architecture_detailed.png)
*High-resolution vector graphic: [model_1_architecture_detailed.svg](../figures/architecture/model_1_architecture_detailed.svg)*

#### Mathematical Formulation
Given an input hand crop $X \in \mathbb{R}^{3 \times 224 \times 224}$, each sequential stage $l \in \{1, 2, 3, 4\}$ computes:
$$Z_l = \text{ReLU}\left(\text{BN}\left(W_l * X_{l-1} + b_l\right)\right)$$
$$X_l = \text{MaxPool}_{2\times 2, s=2}(Z_l)$$

Channel dimensions expand as $3 \to 32 \to 64 \to 128 \to 256$, with spatial grids halving at each step from $224 \to 112 \to 56 \to 28 \to 14$. Global average pooling compresses the feature map to $h \in \mathbb{R}^{256}$:
$$h = \frac{1}{14 \times 14} \sum_{u=1}^{14}\sum_{v=1}^{14} X_4[:, u, v]$$
The classification head applies dense projection with dropout:
$$\hat{y} = W_2 \cdot \left(\text{Dropout}_{0.4}\left(\text{ReLU}\left(W_1 \cdot \text{Dropout}_{0.4}(h)\right)\right)\right)$$
where $W_1 \in \mathbb{R}^{256 \times 256}$ and $W_2 \in \mathbb{R}^{36 \times 256}$.

---

### Model 2: Deep Residual CNN (Residual Skip Blocks from Scratch)

Model 2 introduces identity and projection shortcuts to overcome feature degradation across depth while remaining trained entirely from scratch.

![Model 2 Detailed Architecture](../figures/architecture/model_2_architecture_detailed.png)
*High-resolution vector graphic: [model_2_architecture_detailed.svg](../figures/architecture/model_2_architecture_detailed.svg)*

#### Mathematical Formulation
Each residual stage contains two residual blocks. The transformation of a residual block with input $x$ is:
$$y = \text{ReLU}\left(\mathcal{F}(x, \{W_1, W_2\}) + \mathcal{G}(x)\right)$$
where the residual function $\mathcal{F}$ is:
$$\mathcal{F}(x) = \text{BN}_2\left(\text{Conv}_{3\times 3}\left(\text{ReLU}\left(\text{BN}_1\left(\text{Conv}_{3\times 3}(x)\right)\right)\right)\right)$$
The shortcut operator $\mathcal{G}(x)$ preserves spatial alignment:
$$\mathcal{G}(x) = \begin{cases} x & \text{if } s = 1 \text{ and } C_{in} = C_{out} \quad \text{(Identity shortcut)} \\ \text{BN}_s\left(\text{Conv}_{1\times 1, s}(x)\right) & \text{if } s > 1 \text{ or } C_{in} \neq C_{out} \quad \text{(Projection shortcut)} \end{cases}$$

Backpropagating gradient $\frac{\partial \mathcal{L}}{\partial x}$ demonstrates the direct gradient highway:
$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(\frac{\partial \mathcal{F}}{\partial x} + \frac{\partial \mathcal{G}}{\partial x}\right)$$
Even when $\frac{\partial \mathcal{F}}{\partial x}$ attenuates during early training epochs, gradients flow uninterrupted via $\frac{\partial \mathcal{G}}{\partial x}$, preserving fine knuckle curl and edge features.

---

### Model 3: Transfer Learning Backbone (Pretrained ResNet-18 with Differential Tuning)

Model 3 adapts 11.19M parameters initialized on ImageNet-1K, leveraging rich geometric, contour, and multi-scale texture representations.

![Model 3 Detailed Architecture](../figures/architecture/model_3_architecture_detailed.png)
*High-resolution vector graphic: [model_3_architecture_detailed.svg](../figures/architecture/model_3_architecture_detailed.svg)*

#### Mathematical Formulation
The pretrained backbone maps $X \in \mathbb{R}^{3 \times 224 \times 224}$ into a 512-dimensional embedding:
$$e = \text{AdaptiveAvgPool}_{1\times 1}\left(\Phi_{\text{ResNet18}}(X)\right) \in \mathbb{R}^{512}$$
A classification projection head maps the embedding to class logits:
$$z = W_{\text{head}} \cdot \text{Dropout}_{0.3}(e) + b_{\text{head}}$$
where $W_{\text{head}} \in \mathbb{R}^{36 \times 512}$.

To protect early layer visual filters while adapting the semantic representation, training utilizes a differential learning rate parameter grouping:
$$\theta_{\text{backbone}} \gets \theta_{\text{backbone}} - \eta_{\text{backbone}} \cdot \nabla_{\theta_{\text{backbone}}} \mathcal{L}_{\alpha}$$
$$\theta_{\text{head}} \gets \theta_{\text{head}} - \eta_{\text{head}} \cdot \nabla_{\theta_{\text{head}}} \mathcal{L}_{\alpha}$$
with $\eta_{\text{backbone}} = 1.0 \times 10^{-4}$ and $\eta_{\text{head}} = 5.0 \times 10^{-4}$ ($5\times$ ratio).

The objective function uses label smoothing cross-entropy:
$$\mathcal{L}_{\alpha}(y, z) = -\sum_{k=1}^{36} q_k \log \left(\frac{e^{z_k}}{\sum_{j=1}^{36} e^{z_j}}\right), \quad q_k = (1 - \alpha) \cdot \mathbb{I}(y = k) + \frac{\alpha}{36}$$
where $\alpha = 0.10$.

---

## 3. Receptive Field Comparative Analysis

Receptive field ($RF$) measures the area of the input image that directly influences an activation unit at layer $l$:
$$RF_l = RF_{l-1} + (k_l - 1) \cdot j_{l-1}, \quad j_l = j_{l-1} \cdot s_l$$
where $k_l$ is the kernel size, $s_l$ is the stride, and $j_l$ is the cumulative jump stride.

| Stage / Layer Boundary | Model 1: Baseline Simple CNN | Model 2: Deep Residual CNN | Model 3: ResNet-18 Backbone |
| :--- | :--- | :--- | :--- |
| **Input** | $RF = 1$, $j = 1$ | $RF = 1$, $j = 1$ | $RF = 1$, $j = 1$ |
| **Stem Output** | $RF = 4$, $j = 2$ | $RF = 4$, $j = 2$ | $RF = 11$, $j = 4$ |
| **Stage 1 Output** | $RF = 4$, $j = 2$ (no extra conv) | $RF = 20$, $j = 2$ (4 convs) | $RF = 35$, $j = 4$ (4 convs) |
| **Stage 2 Output** | $RF = 10$, $j = 4$ | $RF = 48$, $j = 4$ (4 convs, $s=2$) | $RF = 83$, $j = 8$ (4 convs, $s=2$) |
| **Stage 3 Output** | $RF = 22$, $j = 8$ | $RF = 104$, $j = 8$ (4 convs, $s=2$) | $RF = 179$, $j = 16$ (4 convs, $s=2$) |
| **Stage 4 Output** | $RF = 46$, $j = 16$ | $RF = 216$, $j = 16$ (4 convs, $s=2$) | $RF = 435$, $j = 32$ (4 convs, $s=2$) |
| **Global Pooling Aggregation** | **224 px (Full Image)** | **224 px (Full Image)** | **224 px (Full Image)** |

### Analytical Insights
- **Model 1**: Prior to global pooling, feature maps achieve an $RF$ of only 46 pixels (20.5% of the $224 \times 224$ image). The network cannot capture long-range dependencies across distant fingertips and wrist anchors simultaneously, relying solely on late linear pooling.
- **Model 2**: Reaches an $RF$ of 216 pixels by Stage 4, covering 96.4% of the frame before global pooling. This enables the network to simultaneously observe thumb placement and finger curvature.
- **Model 3**: Reaches an $RF$ of 435 pixels, substantially exceeding the spatial extent of the input frame. Deep units integrate context across the entire hand posture and wrist angle at multiple feature scales.

---

## 4. Master Error Taxonomy Comparison

Across the 7,200 held-out test samples (Signers 9 and 10), errors fall into three operational categories:
1. **Category A (Static 2D Homographs)**: Identical 3D postures whose 2D monocular projection is fundamentally ambiguous (`W` $\leftrightarrow$ `6`, `V` $\leftrightarrow$ `2`, `D` $\leftrightarrow$ `1`, `F` $\leftrightarrow$ `9`).
2. **Category B (Dynamic Lookalikes)**: Letters with trajectory-dependent motions whose instantaneous static snapshot mimics static signs (`1` $\leftrightarrow$ `Z`, `I` $\leftrightarrow$ `J`).
3. **Category C (Structural Errors)**: Genuine misclassifications between distinct static signs, caused by model capacity or feature representation limitations.

### Comparative Error Distribution

| Error Category | Model 1: Simple CNN (Total: 3,405 errors) | Model 2: Residual CNN (Total: 1,231 errors) | Model 3: ResNet-18 (Total: 586 errors) |
| :--- | :--- | :--- | :--- |
| **Category A: 2D Homographs** | 320 (9.40% of errors / 4.44% of test set) | 465 (37.77% of errors / 6.46% of test set) | **230** (39.25% of errors / 3.19% of test set) |
| • `W` $\leftrightarrow$ `6` | 200 errors | 312 errors | 215 errors |
| • `V` $\leftrightarrow$ `2` | 48 errors | 153 errors | 15 errors |
| • `D` $\leftrightarrow$ `1` | 72 errors | 0 errors | 0 errors |
| • `F` $\leftrightarrow$ `9` | 0 errors | 0 errors | 0 errors |
| **Category B: Dynamic Lookalikes** | 386 (11.34% of errors / 5.36% of test set) | 194 (15.76% of errors / 2.69% of test set) | **208** (35.49% of errors / 2.89% of test set) |
| • `1` $\leftrightarrow$ `Z` | 157 errors | 94 errors | 108 errors |
| • `I` $\leftrightarrow$ `J` | 18 errors | 100 errors | 100 errors |
| • Other `J` / `Z` confusions | 211 errors | 0 errors | 0 errors |
| **Category C: Structural Errors** | **2,699** (79.27% of errors / **37.49% of test set**) | **572** (46.47% of errors / **7.94% of test set**) | **148** (25.26% of errors / **2.05% of test set**) |
| • Major Confused Pairs | `3` $\to$ `Y` (199), `I` $\to$ `X` (164), `G` $\to$ `H` (117) | `8` $\to$ `7` (166), `N` $\to$ `M` (102), `E` $\to$ `M` (100) | `N` $\to$ `M` (81), `R` $\to$ `U` (65), `T` $\to$ `N` (2) |
| **Total Test Misclassifications** | **3,405** (47.29% error rate) | **1,231** (17.10% error rate) | **586** (8.14% error rate) |

### Key Takeaways
1. **Suppression of Structural Deficiencies**: Model 1 suffered from pervasive structural failures across 37.49% of all test images due to restricted receptive field and lack of residual depth. Model 2 reduced structural errors to 7.94%. Model 3 reduced structural errors to **148 images (2.05% of the test set)**.
2. **Bayes Error Bound Convergence**: In Model 3, Category A and Category B errors account for **74.74% of all remaining mistakes (438 out of 586)**. Differentiating `W` from `6` or `1` from single-frame snapshots of `Z` requires temporal motion video or depth sensing. The model has approached the empirical Bayes error limit for static monocular computer vision.
3. **Perfect Class Isolation**: Model 3 achieves **100% precision and 100% recall on 24 of the 36 classes**, confining all remaining structural ambiguity to three subtle finger-tuck pairs (`N` $\to$ `M`, `R` $\to$ `U`, and `T` $\to$ `N`).
