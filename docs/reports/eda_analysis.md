# Exploratory Data Analysis & Dataset Integrity Report

## 1. Executive Summary

This report analyzes the structure, metadata, and pixel characteristics of the American Sign Language Hand Gesture (ASL-HG) dataset before preprocessing.

Our audit verified that the raw dataset contains 36,000 images across 36 classes (0-9, A-Z), contributed by 10 signers (P1-P10). Every image decodes successfully as a 300x300 RGB file.

The primary finding is high frame-to-frame redundancy: consecutive images from the same signer's video burst have an average perceptual hash distance of less than 1, meaning they are near-identical copies. Conversely, images of the same sign across different signers show an average distance of 23-34. This measurement provides the empirical justification for subject-independent partitioning.

---

## 2. Dataset Properties & Metadata Verification

The ASL-HG dataset is stored in 36 class folders under `data/raw/`:

| Dimension | Measured Property |
| :--- | :--- |
| **Total Images** | 36,000 image files |
| **Class Count** | 36 classes: Digits `0`–`9` and Letters `A`–`Z` |
| **Nominal Samples per Class** | Exactly 1,000 images per class folder |
| **Human Signers** | 10 volunteers (`P1` through `P10`) |
| **Captures per Signer** | Exactly 3,600 images per signer (100 per class per signer) |
| **Spatial Dimensions** | Uniform $300 \times 300$ pixels |
| **Color Channels** | 3-channel RGB (8 bits per channel) |
| **File Format & Integrity** | JPEG format; 36,000 of 36,000 files decode completely |
| **Filename Schema** | `P<signer_id>_<class_name>_<index>.jpg` |

![Dataset Structure Overview](../figures/dataset/dataset_structure_overview.png)
*Figure 1: Organization of classes, signers, and images in ASL-HG.*

### Signer-to-Index Correspondence

Within each class folder, images follow an index range tied directly to the signer:
- `P1`: 1–100
- `P2`: 101–200
- `P3`: 201–300
- `P4`: 301–400
- `P5`: 401–500
- `P6`: 501–600
- `P7`: 601–700
- `P8`: 701–800
- `P9`: 801–900
- `P10`: 901–1,000

This deterministic filename schema allows splitting data cleanly by signer identity without relying on error-prone heuristics.

---

## 3. Pixel Distributions & Illumination Variance

Analysis of RGB channel intensity distributions reveals significant illumination differences across signers:

![Signer Luminance Spread](../figures/dataset/signer_luminance_spread.png)
*Figure 2: Systematic background and lighting shifts across signers.*

- **Luminance range**: Mean pixel brightness varies from 160 (Signers P4 and P7) to 202 (Signer P3).
- **Within-signer consistency**: For 9 of the 10 signers, the internal standard deviation of brightness is narrow (5–8 intensity units), indicating fixed camera exposure during recording.
- **Lighting shifts**: Signer P3 exhibits an internal standard deviation of 27 units, reflecting moving ambient sunlight or varying automatic gain during the capture session.

These systematic lighting differences mean that models trained on unaugmented data can easily latch onto background brightness as a shortcut for identifying specific signers.

---

## 4. Perceptual Hashing & Burst-Frame Redundancy

### Perceptual Difference Hash (dHash) Formulation

To quantify visual similarity independent of minor sensor noise, we compute a 64-bit difference hash (dHash) for each image:
1. Convert the image to grayscale and resize to a $9 \times 8$ grid.
2. Compare adjacent horizontal pixels. Each bit is set based on whether the right pixel is brighter than the left:
   $$b_{r,c} = \mathbb{1}\left[p_{r, c+1} > p_{r, c}\right], \quad r \in \{1,\dots,8\}, \; c \in \{1,\dots,8\}$$
3. Compute the Hamming distance $D_H(h_1, h_2)$ between hashes by counting differing bit positions:
   $$D_H(h_1, h_2) = \sum_{k=1}^{64} \left(h_{1,k} \oplus h_{2,k}\right)$$

![dHash Computation Workflow](../figures/dataset/dhash_computation_workflow.png)
*Figure 3: Perceptual difference hashing for quantifying burst-frame redundancy.*

### Measured Hamming Distance Distributions

Comparing hashes across the dataset reveals two distinct similarity patterns:

| Comparison Scope | Median Hamming Distance | Mean Hamming Distance | Interpretation |
| :--- | :--- | :--- | :--- |
| **Consecutive frames (same signer, same sign)** | **1** | **< 1.0** (0.83) | Near-identical duplicates from continuous video bursts. |
| **Non-consecutive frames (same signer, same sign)** | **2** | **2.8** | Highly redundant static poses with minimal hand movement. |
| **Cross-signer frames (different signer, same sign)** | **30** | **27.4** (range 23–34) | Distinct anatomical hand shapes, finger lengths, and backgrounds. |

The average distance between adjacent frames within a signer's burst is less than 1, meaning consecutive frames differ by at most a single bit. In contrast, the distance between different signers performing the same sign is 23 to 34 bits.

```
Hamming Distance Distribution:
Consecutive burst frames: [0-2 bits]  (Near-identical copies)
Cross-signer comparisons: [23-34 bits] (Genuine anatomical & spatial variation)
```

---

## 5. Exact File Duplicates and Blank Frames

### Cross-Signer File Duplication in Classes 'I' and 'V'

A cryptographic MD5 audit identified 435 duplicate groups (994 files) across the dataset. Most notably:
- **Classes I and V**: 199 groups of byte-identical files (400 files total) are shared between Signer P5 and Signer P6.
- Due to these identical copies, Class I contains 893 unique images and Class V contains 894 unique images, rather than 1,000 distinct images.
- **Handling**: These duplicates were cataloged. In the subject-independent training split, exact duplicates are filtered to prevent over-representing identical frames in the training loss.

### Corrupted Blank Frames in Class 'H'

In Class 'H', Signer P8 includes 12 corrupted frames (`P8_H_778`, `P8_H_780`–`789`, `P8_H_791`). A Sobel gradient magnitude audit shows an edge fraction of $e_{40} \le 0.001$ for these frames, compared to $e_{40} \ge 0.012$ for valid hand frames (a 369x separation). Inspection confirmed these frames are empty backgrounds where the signer's hand had dropped below the camera frame before recording stopped. These 12 frames are excluded during preprocessing.

---

## 6. Scientific Implications for Dataset Splitting

The combination of high burst redundancy and signer-specific lighting leads directly to our partitioning protocol:

1. **Why random splitting fails**: If images are split randomly, adjacent frames from the same video burst land in both training and test sets. The model simply interpolates between neighboring frames, memorizing backgrounds and clothing instead of learning generalizable sign shapes.
2. **Why subject-independent splitting is necessary**: By isolating Signers P1–P7 for training, P8 for validation, and P9–P10 for testing, the test set contains zero signers, backgrounds, or burst frames from training. This provides an honest test of whether the models can recognize signs on new, unseen hands.