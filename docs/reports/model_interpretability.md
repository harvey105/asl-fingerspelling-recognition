# Interpretability and Spatial Attribution Analysis via Grad-CAM

## 1. Overview and Scientific Motivation

A central question in visual fingerspelling recognition is whether deep convolutional neural networks learn biologically grounded hand anatomy or overfit to spurious dataset artifacts. Monocular vision benchmarks frequently induce shortcut learning, where models exploit constant letterbox padding boundaries, wall shadows, or signer skin tones rather than discriminative hand kinematics.

To establish causal interpretability across all three project architectures, we developed a native PyTorch Gradient-weighted Class Activation Mapping (Grad-CAM) engine. We instrument the terminal convolutional layers of each model to extract activation maps and their associated class-specific gradient flows:
- **Model 1 (Baseline Simple CNN)**: `features[12]` (Stage 4 Conv2d layer, 256 channels).
- **Model 2 (Deep Residual CNN)**: `stages[7].conv2` (Stage 4 Block 2 Conv2d layer, 256 channels).
- **Model 3 (Fine-Tuned ResNet-18)**: `model.layer4[1].conv2` (Stage 4 Block 2 Conv2d layer, 512 channels).

This investigation evaluates:
1. **Architectural Attention Evolution**: How receptive field depth transitions attribution from diffuse edge-bleeding patterns (Model 1) to tight anatomical joint pinpoints (Model 3).
2. **Counterfactual Diagnostic Attribution**: What spatial features trigger mistakes on top failure modes (e.g., `6` mispredicted as `W`, `1` as `Z`, `R` as `U`) by contrasting predicted vs ground-truth logit gradients.
3. **Letterbox and Context Invariance**: Whether the models maintain attribution neutrality over constant gray `(128, 128, 128)` letterbox borders across signers with disparate skin pigmentation and ambient illumination.

---

## 2. Mathematical Derivations and Gradient Flow Mechanics

### 2.1 The Failure Mode of Standard Global Average Pooling (GAP)
Standard Global Average Pooling condenses 2D spatial feature representations $A^k \in \mathbb{R}^{H \times W}$ for channel $k \in \{1, \dots, C\}$ into a 1D channel vector $a \in \mathbb{R}^C$:

$$a_k = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W A_{i, j}^k$$

where $Z = H \times W$ denotes the spatial resolution of the feature grid.

In classical Class Activation Mapping (CAM; Zhou et al., 2016), the pooled vector $a$ must feed directly into a single linear projection matrix $W \in \mathbb{R}^{K \times C}$ without intermediate layers:

$$y^c = \sum_{k=1}^C w_k^c a_k = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \sum_{k=1}^C w_k^c A_{i, j}^k$$

This formulation yields spatial attribution map $L_{\text{CAM}}^c = \sum_k w_k^c A^k$. However, CAM breaks down when the classification head includes non-linear mappings:
1. **Multilayer Perceptrons (MLPs)**: Models 1 and 2 utilize intermediate dense projections (`Linear(256, 256)` and `Linear(256, 384)`) interleaved with ReLU activations and dropout layers.
2. **Batch Normalization**: ResNet-18 heads frequently incorporate running affine normalizations.

Under nonlinear classifiers, the relationship between channel activation $A^k$ and class logit $y^c$ becomes non-linear and state-dependent ($y^c = f_{\text{head}}(a)$). GAP strips the $(i, j)$ spatial coordinates, preventing direct projection of linear weights back to convolutional maps without backpropagation.

---

### 2.2 Derivation of Grad-CAM Neuron Importance Weights
Grad-CAM (Selvaraju et al., 2017) resolves this by using the backward gradient $\frac{\partial y^c}{\partial A^k}$ to calculate first-order Taylor expansion importance weights.

Let $y^c$ represent the unnormalized class logit score (prior to the softmax normalization $\sigma(y)_c = \frac{e^{y^c}}{\sum_k e^{y^k}}$) for target class $c$. The change in logit $y^c$ resulting from an infinitesimal spatial disturbance $dA_{i, j}^k$ in feature map $k$ is given by:

$$dy^c \approx \sum_{k=1}^C \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k} dA_{i, j}^k$$

To compute a scalar importance coefficient $\alpha_k^c$ capturing the total contribution of activation slice $A^k$ to target class $c$, we integrate spatial partial derivatives via global average pooling:

$$\alpha_k^c = \frac{1}{Z} \sum_{i=1}^H \sum_{j=1}^W \frac{\partial y^c}{\partial A_{i, j}^k}$$

This formulation guarantees that $\alpha_k^c$ reflects the exact first-order sensitivity of the target prediction $y^c$ to channel $k$, regardless of the number of nonlinear layers in the classifier head.

---

### 2.3 Rectified Linear Combination and Heatmap Blending
The coarse 2D localization map $L_{\text{Grad-CAM}}^c \in \mathbb{R}^{H \times W}$ is formed by a linear combination of activation maps weighted by their importance coefficients:

$$L_{\text{Grad-CAM}}^c = \mathrm{ReLU}\left( \sum_{k=1}^C \alpha_k^c A^k \right)$$

#### The Role of the Rectified Linear Operator
The $\mathrm{ReLU}$ operator removes negative values ($\max(0, x)$). In visual fingerspelling, features that suppress a class score (such as an extended thumb during an `S` fist pose) should not contribute positive intensity to the spatial attribution map. ReLU guarantees that only positive evidence for class $c$ is retained.

#### Spatial Upsampling and Normalization Pipeline
1. **Bilinear Upsampling**: The low-resolution map $L^c \in \mathbb{R}^{H \times W}$ ($14 \times 14$ for Models 1 and 2; $7 \times 7$ for Model 3) is bilinearly interpolated to the original input resolution:
   $$\tilde{L}^c = \mathrm{Interpolate}_{224 \times 224}(L_{\text{Grad-CAM}}^c)$$
2. **Min-Max Dynamic Scaling**: To prevent scale distortion across varying gradient magnitudes, the interpolated map is normalized per-sample with numerical stabilization $\epsilon = 10^{-8}$:
   $$\hat{L}^c(u, v) = \frac{\tilde{L}^c(u, v) - \min_{(u, v)} \tilde{L}^c(u, v)}{\max_{(u, v)} \tilde{L}^c(u, v) - \min_{(u, v)} \tilde{L}^c(u, v) + \epsilon}$$
3. **Jet Colormap and Alpha Blending**: The normalized intensity $\hat{L}^c \in [0, 1]$ is mapped to pseudocolor RGB values ($H_{\text{Jet}} = \mathcal{C}_{\text{Jet}}(\hat{L}^c)$) and overlaid on the letterboxed RGB image ($I_{\text{RGB}}$):
   $$I_{\text{overlay}} = (1 - \alpha) \cdot I_{\text{RGB}} + \alpha \cdot H_{\text{Jet}}, \quad \text{where } \alpha = 0.50$$

---

## 3. Theoretical Receptive Field Propagation and Spatial Coverage

Grad-CAM spatial localization is bounded by the theoretical Receptive Field ($RF$) of the target convolutional layer. Receptive field growth across layers follows the recurrence relation:

$$RF_l = RF_{l-1} + (k_l - 1) \cdot j_{l-1}$$

where $j_{l} = j_{l-1} \cdot s_l$ represents the cumulative layer stride jump.

| Structural Dimension                   | Model 1: Baseline Simple CNN      | Model 2: Deep Residual CNN          | Model 3: ResNet-18 Backbone               |
| :------------------------------------- | :-------------------------------- | :---------------------------------- | :---------------------------------------- |
| **Instrumented Layer**                 | `features[12]` (Stage 4 Conv)     | `stages[7].conv2` (Stage 4 Block 2) | `model.layer4[1].conv2` (Layer 4 Block 2) |
| **Channel Capacity ($C$)**             | 256 channels                      | 256 channels                        | 512 channels                              |
| **Spatial Output Grid ($H \times W$)** | $14 \times 14$ ($s=16$)           | $14 \times 14$ ($s=16$)             | $7 \times 7$ ($s=32$)                     |
| **Receptive Field ($RF$)**             | **46 pixels**                     | **216 pixels**                      | **435 pixels**                            |
| **Frame Coverage Ratio ($RF / 224$)**  | **20.5% (Severe under-coverage)** | **96.4% (Global hand coverage)**    | **194.2% (Multi-scale context)**          |
| **Attribution Geometry**               | Fragmented, border-leaking        | Hand bounding envelope              | Localized joint and knuckle peaks         |

![Receptive Field Coverage Across Architectures](../figures/interpretability/receptive_field_coverage.png)

*Figure 1. Spatial receptive field extent prior to global pooling across models relative to the $224 \times 224$ input window. High-resolution vector graphic: [receptive_field_coverage.svg](../figures/interpretability/receptive_field_coverage.svg); source diagram: [receptive_field_coverage.drawio](../figures/interpretability/receptive_field_coverage.drawio).*

### Analytical Insights
1. **Model 1 ($RF = 46\text{ px}$)**: Individual convolutional neurons in Stage 4 observe only small local image regions. A neuron reacting to an extended finger cannot simultaneously observe wrist anchors or contralateral thumb tucks. As a result, Model 1 cannot form unified hand topology representations in its convolutional backbone, relying on the classifier head to fuse disconnected spatial patches. This leads to fragmented saliency maps with pronounced sensitivity to high-contrast letterbox boundaries.
2. **Model 2 ($RF = 216\text{ px}$)**: Sixteen stacked $3 \times 3$ convolutions across four residual stages expand the receptive field to 216 pixels, encompassing almost the entire hand. The network constructs cohesive attribution envelopes that bound the hand silhouette, eliminating border edge artifacts.
3. **Model 3 ($RF = 435\text{ px}$)**: With an initial $7 \times 7$ strided convolution and deep residual stages, neurons in `layer4[1].conv2` receive visual inputs spanning the full input frame. Combined with ImageNet low-level edge and texture representations, Model 3 focuses attribution on key anatomical markers (distal phalanges, interphalangeal joints, and finger contact loops).

---

## 4. Empirical Evaluation Panels and Visual Evidence

### 4.1 Architectural Attribution Evolution
To examine how attribution matures across model architectures, we evaluate identical test inputs from unseen Signer P9 across four canonical hand configurations:
- **`B`**: Flat palm blade (four extended parallel fingers, thumb folded across lower palm).
- **`O`**: Ovoid ring enclosure (fingertips flexed to meet thumb pad, forming an open circular aperture).
- **`Y`**: Radial/ulnar spread (thumb and pinky extended, middle three fingers folded).
- **`L`**: Orthogonal index and thumb (vertical index, horizontal thumb, remaining digits curled).

![Architectural Attribution Evolution](../figures/interpretability/architectural_attribution_evolution.png)

*Figure 2. Architectural attribution progression across Model 1, Model 2, and Model 3 on representative ASL gestures from unseen Signer P9. Columns show: (1) Input letterboxed RGB crop, (2) Model 1 (Simple CNN), (3) Model 2 (Deep Residual CNN), and (4) Model 3 (Transfer Learning ResNet-18).*

#### Detailed Anatomical Breakdown
1. **Gesture B (Flat Palm Blade)**:
   - *Model 1*: Shows vertical activation bands that leak onto both lateral letterbox seams ($85.8\%$ leakage ratio). Saliency is split across the palm with no focus on the folded thumb.
   - *Model 2*: Restricts attribution to the hand silhouette, with moderate activation across the palm body and proximal phalanges.
   - *Model 3*: Pinpoints the folded thumb base and the central metacarpal knuckle—the critical anatomical signal distinguishing `B` from open palm gestures `4` or `5`.
2. **Gesture O (Ovoid Ring Enclosure)**:
   - *Model 1*: Yields disconnected activation peaks scattered across the wrist and palm center.
   - *Model 2*: Forms a broad activation envelope covering the palm.
   - *Model 3*: Focuses on the curved loop closure where the thumb pad contacts the index and middle fingertips, demonstrating sensitivity to the circular topological aperture.
3. **Gesture Y (Radial/Ulnar Spread)**:
   - *Model 1*: Highlights outer finger silhouettes with border edge bleeding.
   - *Model 2*: Concentrates heavily on the central curled fingers.
   - *Model 3*: Balances saliency between the retracted central knuckles and the extended thumb/pinky terminal anchors.
4. **Gesture L (Orthogonal Index and Thumb)**:
   - *Model 1*: Displays low activation on the horizontal thumb and background border leakage.
   - *Model 2*: Covers both extended digits in a unified L-shaped footprint.
   - *Model 3*: Concentrates saliency on the 90-degree radial notch between the thumb base and index metacarpal.

---

### 4.2 Counterfactual Diagnostic Attribution on Failure Modes
Evaluating the model against its top failure modes clarifies whether misclassifications stem from genuine anatomical ambiguity or feature representation errors.

For each confusion case, we compute dual Grad-CAM attribution maps using Model 3:
1. **Targeting the Predicted Class Logit**: Isolates the spatial cues that triggered the false prediction.
2. **Targeting the Ground-Truth Class Logit**: Identifies where the model looks for the correct gesture's anatomical markers.

![Counterfactual Diagnostic Attribution](../figures/interpretability/counterfactual_diagnostic_attribution.png)

*Figure 2. Dual counterfactual attribution maps for top failure modes in Model 3 on unseen test data. Left: Input letterboxed crop. Middle: Grad-CAM targeting the predicted mistake logit. Right: Grad-CAM targeting the ground-truth class logit.*

#### In-Depth Case Analysis
1. **True '6' Misclassified as 'W' (Confidence: 0.270, Signer P9)**:
   - *Anatomical Context*: Both `6` and `W` display three extended upright digits (index, middle, ring). In `W`, the thumb folds over the pinky fingernail; in `6`, the thumb tip contacts the pinky tip, forming a closed ring while leaving the other three fingers upright.
   - *Predicted Logit (W)*: Saliency concentrates on the three upright vertical finger shafts, which closely match the signature shape of `W`.
   - *Ground-Truth Logit (6)*: Saliency shifts toward the palm center, targeting the thumb-pinky touchpoint. Because the pinky is partially occluded from a monocular front view, the contact signal is weak, causing the classifier to default to `W`.
2. **True '1' Misclassified as 'Z' (Confidence: 0.427, Signer P9)**:
   - *Anatomical Context*: Sign `1` is a static vertical index finger with remaining digits curled into a fist. Sign `Z` is a dynamic sign that begins with an extended index finger before tracing a zigzag path through the air.
   - *Predicted Logit (Z)*: Attribution concentrates at the distal index fingertip, which initiates the spatial trace for dynamic sign `Z`.
   - *Ground-Truth Logit (1)*: Attribution expands downward to include the curled thumb tucked across the middle finger knuckle, the static marker of numeral `1`.
3. **True 'R' Misclassified as 'U' (Confidence: 0.300, Signer P10)**:
   - *Anatomical Context*: Sign `U` features parallel index and middle fingers held together. Sign `R` features intertwined fingers, with the index crossed over the middle finger.
   - *Predicted Logit (U)*: Saliency covers both finger shafts symmetrically, treating them as adjacent parallel digits.
   - *Ground-Truth Logit (R)*: Saliency shifts to the crossing intersection where the index overlaps the middle finger. In monocular 2D images with low local contrast, overlapping digits can be difficult to resolve from parallel fingers.

---

### 4.3 Letterbox Invariance and Neutral Gray Background Suppression
To ensure that preprocessing steps do not introduce non-causal confounding features, we evaluate gradient attribution on the neutral gray `(128, 128, 128)` letterbox padding across diverse signers:
- **Signer P9**: Fair skin tone, cool ambient lighting.
- **Signer P10**: Deep skin tone, high-contrast ambient lighting.

![Letterbox Invariance Verification](../figures/interpretability/letterbox_invariance_verification.png)

*Figure 3. Letterbox invariance and artifact suppression across diverse skin tones and lighting conditions. Values indicate the padding leakage ratio (mean attribution in padding vs mean attribution on active hand ROI).*

#### Quantitative Attribution Leakage Ratios
The padding leakage ratio is defined as:

$$\mathcal{R}_{\text{padding}} = \frac{\frac{1}{|\Omega_{\text{pad}}|} \sum_{(u, v) \in \Omega_{\text{pad}}} \hat{L}(u, v)}{\frac{1}{|\Omega_{\text{hand}}|} \sum_{(u, v) \in \Omega_{\text{hand}}} \hat{L}(u, v)}$$

where $\Omega_{\text{pad}} = \{ (u, v) \mid I(u, v) \approx (128, 128, 128) \}$ and $\Omega_{\text{hand}} = \Omega \setminus \Omega_{\text{pad}}$.

| Signer & Setting                 | Gesture Case   | Model 1 (Simple CNN) | Model 2 (Complex CNN) | Model 3 (Transfer ResNet-18) | Invariance Status       |
| :------------------------------- | :------------- | :------------------- | :-------------------- | :--------------------------- | :---------------------- |
| **Signer P9 (Fair / Cool)**      | **B (Blade)**  | **85.8%**            | 13.1%                 | **6.5%**                     | **Invariance Verified** |
| **Signer P10 (Deep / Contrast)** | **B (Blade)**  | **60.7%**            | 11.3%                 | **9.0%**                     | **Invariance Verified** |
| **Signer P9 (Fair / Cool)**      | **Y (Spread)** | **10.0%**            | 0.1%                  | **2.0%**                     | **Invariance Verified** |
| **Signer P10 (Deep / Contrast)** | **Y (Spread)** | **44.5%**            | 0.5%                  | **4.1%**                     | **Invariance Verified** |

#### Findings
- **Model 1**: Exhibits severe edge leakage ($44.5\% - 85.8\%$). Because its receptive field is small ($46\text{ px}$), early convolutional filters react strongly to step discontinuities at the letterbox borders ($x \approx 35\text{ px}$ and $x \approx 189\text{ px}$), treating the gray boundary as a discriminative shape feature.
- **Models 2 and 3**: Larger receptive fields ($216\text{ px}$ and $435\text{ px}$) allow these models to contextualize the padding borders as invariant boundaries. In Model 3, padding leakage remains under $10\%$ across all signers and lighting conditions, confirming that its predictions rely on anatomical hand features rather than background artifacts.

---

## 5. Software Architecture

The interpretability pipeline is organized as modular, standalone components:

```
src/
├── utils/
│   └── gradcam.py                 <-- Native Grad-CAM engine (<105 lines, strict type hints)
└── experiments/
    └── generate_gradcam_panels.py <-- High-resolution panel runner (<220 lines)
tests/
└── test_gradcam.py                <-- Comprehensive unit test suite (5 passing test cases)
docs/
└── figures/interpretability/      <-- Publication figures (3 multi-panel PNGs + receptive field diagram)
```
