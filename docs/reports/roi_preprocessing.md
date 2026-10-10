# Hand ROI Detection, Cropping, and Letterboxing

## 1. Overview

In raw camera captures, the human hand occupies only a small portion of the 300x300 image. Background walls, desk surfaces, and lighting variations dominate the outer boundary. Feeding full uncropped frames directly to a convolutional network forces early filters to process irrelevant background information, slowing down training and hurting accuracy.

To focus the models on the sign itself, our preprocessing pipeline isolates the hand Region of Interest (ROI) and standardizes the image geometry:
1. **Detection**: MediaPipe Hands detects the hand and produces a tight bounding box.
2. **Margin expansion**: Adds a 20% proportional margin around the box to avoid clipping extended fingertips and knuckles.
3. **Aspect-ratio letterboxing**: Scales the longest dimension to 224 pixels and adds neutral gray `(128, 128, 128)` padding. This brings every crop to a uniform 224x224 square without distorting finger shapes.

Processed crops are stored in `data/processed_crops/`.

---

## 2. MediaPipe Landmark Localization

MediaPipe Hands is used to localize the active signing hand across the dataset. The detector estimates 21 three-dimensional landmarks on the hand, tracing key joints from the wrist to the distal tips of all five digits.

![ROI Letterbox Workflow](../figures/dataset/roi_letterbox_workflow.png)
*Figure 1: Hand localization and aspect-preserving letterbox workflow.*

### Detection Performance and Bounding Box Extraction

Given the detected 21 landmark points $(x_i, y_i)$ normalized to $[0, 1]$, the hand bounding box is extracted as:

$$x_{\min} = \min_{i} x_i, \quad x_{\max} = \max_{i} x_i, \quad y_{\min} = \min_{i} y_i, \quad y_{\max} = \max_{i} y_i$$

A 20% margin is added proportionally to each side of the box:

$$\Delta w = 0.20 \cdot (x_{\max} - x_{\min}), \quad \Delta h = 0.20 \cdot (y_{\max} - y_{\min})$$

$$x_{\min}' = \max(0, x_{\min} - \Delta w), \quad x_{\max}' = \min(W, x_{\max} + \Delta w)$$
$$y_{\min}' = \max(0, y_{\min} - \Delta h), \quad y_{\max}' = \min(H, y_{\max} + \Delta h)$$

This margin protects against accidental fingertip clipping in extended poses (such as the index finger in '1' and 'D', or the pinky in 'I').

On the ASL-HG dataset, single-hand MediaPipe detection achieved **99.86% coverage** across non-blank frames. Out of 36,000 raw frames:
- **34,980 frames (97.2%)**: Localized directly via MediaPipe single-hand landmark bounding boxes.
- **1,000 frames (2.8%)**: Retained as full frames using the special-class rule for Class 0.
- **8 frames (<0.1%)**: MediaPipe missed the hand in difficult poses; these were safely preserved using full-frame fallback.
- **12 frames**: Identified as corrupted blank frames (with no hand visible) and excluded.

---

## 3. Special Class Rules

### Class 0: Two-Handed Sign Preservation

ASL fingerspelling predominantly uses one hand. However, Class '0' in the ASL-HG dataset is articulated as a two-handed gesture, where both hands form a connected circular shape.

Running a single-hand detector on Class '0' produces an unpredictable result: the detector locks onto whichever hand has higher initial confidence, cropping away the other hand entirely. Removing the second hand destroys the visual identity of the sign.

To handle this cleanly, all 1,000 frames in Class '0' bypass single-hand cropping and are kept as full frames. This preserves both hands and maintains the complete circular gesture.

### Uncropped Fallback for Extreme Poses

In 8 non-zero frames where hand detection failed, the pipeline falls back to keeping the full frame rather than discarding data or failing silently. The full frame is letterboxed directly to 224x224, ensuring downstream loaders receive valid tensors without pipeline crashes.

---

## 4. Aspect-Ratio Letterboxing vs. Naive Squashing

### Why Letterboxing Matters

Hand crops have naturally variable aspect ratios. Vertical gestures (like 'B', 'D', and 'F') produce tall, narrow bounding boxes with aspect ratios around 1:2. Lateral gestures (such as 'G', 'H', and 'C') produce wide, horizontal boxes with ratios around 2:1.

Standard computer vision pipelines often resize images directly to a square canvas ($224 \times 224$), squashing or stretching non-square regions. In hand gesture recognition, this spatial distortion changes finger thickness, joint angles, and palm proportions.

![Letterbox vs Squash](../figures/dataset/letterbox_vs_squash.png)
*Figure 2: Geometric preservation of letterboxing versus morphological distortion of naive squashing.*

### The Letterboxing Transformation

Our pipeline uses an aspect-preserving letterbox transformation:
1. The longest dimension of the crop is scaled to 224 pixels:
   $$s = \frac{224}{\max(w, h)}, \quad w' = \text{round}(w \cdot s), \quad h' = \text{round}(h \cdot s)$$
2. The resized crop is centered within a $224 \times 224$ square canvas.
3. The remaining borders are padded symmetrically using neutral gray `(128, 128, 128)`:
   $$\text{pad}_{\text{left}} = \left\lfloor\frac{224 - w'}{2}\right\rfloor, \quad \text{pad}_{\text{right}} = 224 - w' - \text{pad}_{\text{left}}$$
   $$\text{pad}_{\text{top}} = \left\lfloor\frac{224 - h'}{2}\right\rfloor, \quad \text{pad}_{\text{bottom}} = 224 - h' - \text{pad}_{\text{top}}$$

Neutral gray corresponds to $(0.5, 0.5, 0.5)$ in normalized RGB space. After ImageNet standardization, this shifts close to zero, preventing the padding borders from triggering strong activation in convolutional filters.

### Empirical Validation from Controlled Ablation

The impact of letterboxing versus naive squashing was evaluated in a controlled experiment on the unseen test set (7,200 images, Signers P9 and P10), as detailed in [`ablation_experiments.md`](ablation_experiments.md):

| Architecture | Letterbox Accuracy | Naive Squashing Accuracy | Performance Delta | Letterbox Macro F1 | Naive Squashing Macro F1 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model 1: Baseline Simple CNN** | **52.71%** (3,795 / 7,200) | 23.04% (1,659 / 7,200) | **-29.67%** | **49.20%** | 20.84% (-28.36%) |
| **Model 3: Fine-Tuned ResNet-18** | **91.86%** (6,614 / 7,200) | 85.85% (6,181 / 7,200) | **-6.01%** | **91.10%** | 84.14% (-6.96%) |

In Model 1, squashing causes a severe drop (**-29.67%**), cutting accuracy in half. Without residual connections or pretrained features, shallow convolutional layers depend heavily on rigid spatial proportions; squashing tall signs into squares widens fingers and distorts joint angles, making it difficult to distinguish signs like 'B' from 'C'.

Even in Model 3, which benefits from deep multi-scale features, squashing causes an unnecessary **6.01% drop** (433 additional misclassifications). Letterboxing with neutral gray padding avoids this spatial distortion entirely.