# Task 2.2: Hand ROI Cropping & Standardization Pipeline

## Part 1: Core Localization Engine & Mask Generation

### Description
Standard CNN architectures are highly susceptible to spurious spatial correlations. Our Exploratory Data Analysis (EDA) in Task 2.1 proved that the background environment in the dataset remains static across frames. To prevent the neural network from minimizing loss by memorizing background wall textures, we must physically decouple the semantic foreground (the hand) from the static background. 

### Objective
Build a robust offline preprocessing engine that tightly localizes the hand region, crops it with a safe margin, and computes a binary foreground alpha mask. This ensures:
1. **Background Noise Elimination:** The convolutional kernels are forced to learn anatomical features (knuckle flexion, finger spacing).
2. **Standardization:** All cropped hands are warped to a uniform $200 \times 200$ spatial resolution, ensuring consistent tensor dimensions for mini-batching.
3. **Task 2.3 Preparation:** The extracted alpha mask is saved offline so the training DataLoader can instantly perform alpha-blending with novel backgrounds at zero computational latency.

### Method
Our `HandROIExtractor` pipeline utilizes a two-tier architectural design:
1. **Primary Pass (MediaPipe Hands):** We utilize MediaPipe's BlazePalm and Hand Landmark models to regress 21 2D anatomical keypoints. A bounding box is calculated from the extrema of these landmarks. To prevent spatial information loss (e.g., fingertips being sliced off), we expand the bounding box by a $15\%$ spatial margin.
2. **Alpha Mask Generation (YCrCb + Otsu):** Inside the tightly cropped bounding box, we apply a color-space transformation from RGB to YCrCb. Skin tones cluster densely in the Cr and Cb channels. We isolate the skin cluster and apply Otsu's adaptive thresholding to generate a crisp, 1-bit binary alpha mask of the hand silhouette.
3. **Fallback Mechanism:** For extreme angles or control classes (like `nothing` or `space`) where the MediaPipe regressor fails to detect a topological hand, the pipeline automatically falls back to applying the YCrCb + Otsu thresholding over the entire frame, calculating the spatial bounding box from the largest connected skin contour.

### Outputs/Results
- **Deliverable:** `src/preprocess_roi.py` containing the `HandROIExtractor` class.
- **Visual Proof:** A generated test grid (`docs/figures/roi_test_grid.png`) demonstrating the Original Image $\rightarrow$ Bounding Box Crop $\rightarrow$ Binary Alpha Mask pipeline.

### Analysis (Deep Learning Defense)
By cropping the image strictly to the hand ROI, we artificially induce **Translation Invariance** prior to the network's forward pass. Instead of requiring the CNN's early Conv2D layers to waste parameter capacity acting as spatial attention mechanisms to find the hand, we center the semantic target. Furthermore, separating mask generation into an offline preprocessing phase avoids catastrophic GPU starvation; running MediaPipe on-the-fly during training would consume ~15ms per image, bottlenecking DataLoader throughput and leaving the GPU idle.