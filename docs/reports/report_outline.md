# Major Assignment Report: Comprehensive Assembly Outline & Guide (Part 2: Practical Application)

> **Context & Purpose**:  
> This outline guides the final assembly of **Part 2: Practical Application** for the Deep Learning Major Assignment report, strictly structured around the rubric prescribed by the course instructor (**Dr. Nguyễn Thị Kim Ngân**, Faculty of Mathematical Economics, National Economics University).  
> *(Note: Part 1: Theory is being handled independently and is omitted here).*

---

## High-Level Document Architecture

The practical application report is organized into **five mandatory sections** matching the assignment specification:

```
Part 2: Practical Application
├── Section I. Problem Description
├── Section II. Dataset Description
├── Section III. CNN Model Design
│   ├── Subsection 1: Data Preprocessing, Ambidexterity & Partitioning
│   ├── Subsection 2: Model 1 — Simple Sequential CNN (Scratch Baseline)
│   ├── Subsection 3: Model 2 — Deep Residual CNN (Mini-ResNet Scratch)
│   └── Subsection 4: Model 3 — Transfer Learning & Fine-Tuning (ResNet-18)
├── Section IV. Experimental Results
│   ├── Subsection 1: Benchmark Evaluation Table (Course Rubric)
│   ├── Subsection 2: Per-Class Performance & Error Decomposition
│   ├── Subsection 3: Three-Tier Error Taxonomy & Physical Failure Mechanisms
│   ├── Subsection 4: Controlled Scientific Ablation Suite
│   ├── Subsection 5: External Real-World Benchmark & Handedness Parity
│   └── Subsection 6: Spatial Attribution & Interpretability via Grad-CAM
└── Section V. Conclusion & References
    ├── Subsection 1: Executive Findings & Theoretical Takeaways
    ├── Subsection 2: Limitations & Future Directions
    └── Subsection 3: Scholarly Literature References
```

---

## Detailed Section-by-Section Outline

---

### Section I. Problem Description

This section establishes the engineering and social motivation of real-time sign language recognition, formal mathematical inputs and outputs, and the workflow sequence.

#### Subsections & Core Topics to Cover:
1. **Title & Real-World Motivation**:
   - Title: *Real-Time American Sign Language (ASL) Fingerspelling Recognition using Deep Convolutional Neural Networks*.
   - Objective: Bridging the communication barrier for Deaf and Hard-of-Hearing individuals through automated monocular vision recognition of the 36 alphanumeric fingerspelling signs.
2. **Mathematical Formulation (Input & Output Spaces)**:
   - Input: Monocular RGB image $X \in \mathbb{R}^{3 \times 224 \times 224}$ extracted from continuous video frames.
   - Output: Predicted class $\hat{y} \in \{0, \dots, 35\}$ corresponding to 36 discrete alphanumeric categories: digits `0`–`9` and English alphabet letters `A`–`Z`.
3. **Execution Task Pipeline**:
   - High-level sequence: Raw Ingestion $\to$ Landmark ROI Localization $\to$ Aspect Letterbox Padding $\to$ Subject-Independent Splitting $\to$ DirectML Training $\to$ Interpretability Attribution.

#### Source References:
- **Primary Report**: [`docs/reports/project_overview.md`](project_overview.md) *(Sections 1, 2, and 3)*
- **Figure**: [`docs/figures/architecture/vision_challenge_pipeline.png`](../figures/architecture/vision_challenge_pipeline.png) *(Draw.io source available)*

---

### Section II. Dataset Description

This section presents the physical properties, participant structure, class balance, and empirical challenges of the benchmark dataset.

#### Subsections & Core Topics to Cover:
1. **Benchmark Metadata & Ingestion**:
   - Source: Mendeley ASL-HG Benchmark Dataset ([Download Link](https://data.mendeley.com/datasets/j4y5w2c8w9/1)).
   - Scale: 36,000 raw frames at $300 \times 300$ resolution in RGB format.
   - Balance: Exactly 1,000 images per class across all 36 classes (digits `0`–`9`, letters `A`–`Z`).
2. **Participant Demographics & Skin-Tone Diversity**:
   - 10 diverse human signers (`P1` through `P10`), contributing exactly 100 frames per class (3,600 images per volunteer).
   - Diverse natural skin pigmentation, hand morphology, and indoor/outdoor illumination.
3. **Temporal Redundancy & Video Burst Redundancy**:
   - 64-bit perceptual difference hashing (dHash) analysis: consecutive burst frames within a signer's recording session exhibit Hamming distance $< 1$ (near-duplicate frames), whereas cross-signer Hamming distance averages $23$–$34$.
   - Justification: Provides the foundational scientific rationale for subject-independent evaluation (preventing burst-frame leakage).
4. **Structural Anomalies & Duplicates**:
   - Cross-signer byte-identical duplicates between P5 and P6 in classes `I` and `V`.
   - Single corrupt/blank frame pruning (`P8_H_792.jpg`).

#### Source References:
- **Primary Reports**:
  - [`docs/reports/eda_analysis.md`](eda_analysis.md) *(Exhaustive perceptual hashing, class balance, and luminance audits)*
  - [`docs/reports/dataset_splits.md`](dataset_splits.md) *(Signer distribution and partition metrics)*
- **Primary Code**: [`src/data/splits.py`](../../src/data/splits.py) · [`src/config.py`](../../src/config.py)
- **Figures**:
  - [`docs/figures/dataset/dataset_structure_overview.png`](../figures/dataset/dataset_structure_overview.png) *(Class & signer distribution)*
  - [`docs/figures/dataset/dhash_computation_workflow.png`](../figures/dataset/dhash_computation_workflow.png) *(Perceptual hash distance workflow)*
  - [`docs/figures/eda/eda_36_class_grid.png`](../figures/eda/eda_36_class_grid.png) *(36-class visual exemplar mosaic)*
  - [`docs/figures/eda/eda_volunteer_mosaic.png`](../figures/eda/eda_volunteer_mosaic.png) *(10 signers lighting and complexion spread)*

---

### Section III. CNN Model Design

The core architectural section of the report. It details data preprocessing, followed by comprehensive mathematical formulations, layer blueprints, and design rationales for each of the three model tiers.

#### Subsection III.1: Data Preprocessing, Ambidexterity & Partitioning
- **Topics**:
  1. *Hand ROI Detection*: MediaPipe Hands 21-landmark detector with 20% adaptive margin expansion to prevent fingertip clipping; full-frame fallback for Class `0` (two-handed gesture).
  2. *Aspect-Ratio Letterboxing (Zero-Squash Standard)*: Preserving anatomical aspect ratios by scaling the longest dimension to 224 px and padding symmetrically with neutral gray RGB `(128, 128, 128)`.
  3. *Runtime Stochastic Ambidexterity*: Invariant ASL single-hand phonology; solving the left-hand capture bias of ASL-HG via `RandomHorizontalFlip(p=0.5)` in the DataLoader without duplicating 36,000 files on disk.
  4. *Subject-Independent Partitioning*: Leave-Two-Signers-Out (Train: P1–P7, 25,200; Val: P8, 3,599; Test: P9–P10, 7,200).
- **Source Reports**: [`docs/reports/roi_preprocessing.md`](roi_preprocessing.md) · [`docs/reports/dataset_pipeline.md`](dataset_pipeline.md) · [`docs/reports/dataset_splits.md`](dataset_splits.md)
- **Source Code**: [`src/data/preprocess_roi.py`](../../src/data/preprocess_roi.py) · [`src/data/dataloaders.py`](../../src/data/dataloaders.py) · [`src/data/splits.py`](../../src/data/splits.py)
- **Figures**:
  - [`docs/figures/dataset/roi_letterbox_workflow.png`](../figures/dataset/roi_letterbox_workflow.png) *(MediaPipe margin & letterbox padding)*
  - [`docs/figures/dataset/letterbox_vs_squash.png`](../figures/dataset/letterbox_vs_squash.png) *(Squash distortion vs letterbox preservation)*
  - [`docs/figures/dataset/augmentation_pipeline.png`](../figures/dataset/augmentation_pipeline.png) *(Affine + flip augmentation flow)*
  - [`docs/figures/dataset/dataset_splits_structure.png`](../figures/dataset/dataset_splits_structure.png) *(Subject vs random partitioning)*

---

#### Subsection III.2: Model 1 — Baseline Simple CNN (Sequential Feedforward)
- **Topics**:
  1. *Architectural Topology*: 4 sequential stages without skip connections ($3 \to 32 \to 64 \to 128 \to 256$), Batch Normalization, ReLU activations, $2\times 2$ MaxPool downsampling, Global Average Pooling, and a 256-unit dense classifier with dual Dropout ($p=0.4$).
  2. *Mathematical Formulation*: Forward stage equation $Z_l = \text{ReLU}(\text{BN}(W_l * X_{l-1}))$, pooling, and linear projection.
  3. *Parameter Count & Receptive Field*: 463,940 parameters; spatial receptive field of 46 px (narrow local context).
- **Source Reports**: [`docs/reports/model1_architecture.md`](model1_architecture.md) · [`docs/reports/models_architecture.md`](models_architecture.md) *(Section 2.1)*
- **Source Code**: [`src/models/simple_cnn.py`](../../src/models/simple_cnn.py)
- **Figures**:
  - [`docs/figures/architecture/model_1_architecture_detailed.png`](../figures/architecture/model_1_architecture_detailed.png) *(Detailed vertical layer blueprint; vector `.svg` available)*
  - [`docs/figures/architecture/model_1_architecture_compressed.png`](../figures/architecture/model_1_architecture_compressed.png) *(Horizontal pipeline overview)*

---

#### Subsection III.3: Model 2 — Deep Residual CNN (Mini-ResNet Scratch)
- **Topics**:
  1. *Architectural Topology*: Conservative stem ($3\times 3, s=1$ + MaxPool), followed by 4 residual stages containing 8 residual blocks (16 Conv layers; 17 total Conv layers).
  2. *Identity Skip Highways & Gradient Propagation*: Mathematical formulation of unbroken residual highways ($\frac{\partial \mathcal{E}}{\partial x_l} = \frac{\partial \mathcal{E}}{\partial x_L} [I + \sum \frac{\partial \mathcal{F}}{\partial x}]$) preventing gradient vanishing. Projection shortcuts ($1\times 1, s=2$) for inter-stage spatial transitions.
  3. *Parameter Count & Receptive Field*: 2,907,588 parameters; spatial receptive field of 216 px (encompassing 96.4% of the input hand).
- **Source Reports**: [`docs/reports/model2_architecture.md`](model2_architecture.md) · [`docs/reports/models_architecture.md`](models_architecture.md) *(Section 2.2)*
- **Source Code**: [`src/models/complex_cnn.py`](../../src/models/complex_cnn.py)
- **Figures**:
  - [`docs/figures/architecture/model_2_architecture_detailed.png`](../figures/architecture/model_2_architecture_detailed.png) *(Detailed vertical layer blueprint; vector `.svg` available)*
  - [`docs/figures/architecture/model_2_architecture_compressed.png`](../figures/architecture/model_2_architecture_compressed.png) *(Horizontal pipeline overview)*

---

#### Subsection III.4: Model 3 — Pretrained Transfer Learning & Fine-Tuning (ResNet-18)
- **Topics**:
  1. *Architectural Topology*: Pretrained ImageNet-1k ResNet-18 feature extractor paired with a direct linear projection classification head (`Linear(512, 36)`) and single Dropout ($p=0.3$).
  2. *Transfer Learning Optimization*: Two-phase training protocol; differential learning rates ($\eta_{\text{backbone}} = 10^{-4}$ vs $\eta_{\text{head}} = 5 \times 10^{-4}$); label smoothing regularization ($\alpha = 0.10$).
  3. *Parameter Count & Receptive Field*: 11,194,980 parameters; spatial receptive field of 435 px (global multi-scale context exceeding the 224 px frame).
- **Source Reports**: [`docs/reports/model3_architecture.md`](model3_architecture.md) · [`docs/reports/models_architecture.md`](models_architecture.md) *(Section 2.3)*
- **Source Code**: [`src/models/transfer_cnn.py`](../../src/models/transfer_cnn.py)
- **Figures**:
  - [`docs/figures/architecture/model_3_architecture_detailed.png`](../figures/architecture/model_3_architecture_detailed.png) *(Detailed vertical layer blueprint; vector `.svg` available)*
  - [`docs/figures/architecture/model_3_architecture_compressed.png`](../figures/architecture/model_3_architecture_compressed.png) *(Horizontal pipeline overview)*

---

### Section IV. Experimental Results

This section presents empirical results across unseen subjects, controlled ablations, failure case diagnostics, and spatial interpretability evidence.

#### Subsection IV.1: Benchmark Evaluation Table (Course Rubric)
- **Topics**:
  - Comparative results table across all three models on the 7,200 unseen subject test set (Signers P9 and P10).
  - Exact rubric columns: Model, Top-1 Accuracy, Macro-Precision, Macro-Recall, Macro-F1 Score, Parameters, and Hardware Throughput.
- **Source Tables & Text**:
  - Markdown Summary Table: [`docs/reports/models_architecture.md`](models_architecture.md) *(Section 1)*
  - **Double-Column Publication LaTeX Table**: Directly copy the `\begin{table*}...\end{table*}` block from [`docs/reports/error_diagnostics.md`](error_diagnostics.md#L174-L229).
- **Figures**:
  - Training & Validation Curves:
    - [`docs/figures/evaluations/simple_cnn_training_curves.png`](../figures/evaluations/simple_cnn_training_curves.png)
    - [`docs/figures/evaluations/complex_cnn_training_curves.png`](../figures/evaluations/complex_cnn_training_curves.png)
    - [`docs/figures/evaluations/transfer_cnn_training_curves.png`](../figures/evaluations/transfer_cnn_training_curves.png)

---

#### Subsection IV.2: Per-Class Performance & Error Breakdown
- **Topics**:
  - Per-class metric analysis across all 36 classes (support = 200 each).
  - The 23 anchor classes achieving 100% precision and recall simultaneously in Model 3.
  - Normalized $36 \times 36$ confusion matrices for each architecture tier.
- **Source Reports**: [`docs/reports/error_diagnostics.md`](error_diagnostics.md) *(Section 4)*
- **Source Metrics**: [`docs/metrics/transfer_cnn_test_evaluation.json`](../metrics/transfer_cnn_test_evaluation.json)
- **Figures**:
  - [`docs/figures/evaluations/simple_cnn_confusion_matrix.png`](../figures/evaluations/simple_cnn_confusion_matrix.png)
  - [`docs/figures/evaluations/complex_cnn_confusion_matrix.png`](../figures/evaluations/complex_cnn_confusion_matrix.png)
  - [`docs/figures/evaluations/transfer_cnn_confusion_matrix.png`](../figures/evaluations/transfer_cnn_confusion_matrix.png)

---

#### Subsection IV.3: Three-Tier Error Taxonomy & Physical Failure Mechanisms
- **Topics**:
  - Detailed physical and phonological root-cause breakdown of Model 3's 586 residual errors:
    1. *Category A: Static 2D Homographs (36.7% / 215 errors)*: Structurally identical planar projections ($6 \leftrightarrow W$, $2 \leftrightarrow V$). Anterior palm view occludes the distinguishing thumb-pinky contact loop behind the palm volume.
    2. *Category B: Dynamic Motion Collapse (35.5% / 208 errors)*: Static single-frame ambiguity of trajectory-dependent signs ($1 \leftrightarrow Z$, $I \leftrightarrow J$). Static snapshots lack the temporal stroke path ($\Delta t = 0$).
    3. *Category C: Morphological Nuances & Bayes Error Bound (27.8% / 163 errors)*: Inter-digital occlusions ($N \leftrightarrow M$, $R \leftrightarrow U$, $T \leftrightarrow N$). Thumb tuck depth occluded under downward camera foreshortening.
- **Source Reports**: [`docs/reports/error_diagnostics.md`](error_diagnostics.md) *(Sections 2 and 3)*
- **Source Code**: [`src/utils/error_visualizer.py`](../../src/utils/error_visualizer.py) · [`src/utils/progression_visualizer.py`](../../src/utils/progression_visualizer.py)
- **Figures**:
  - [`docs/figures/diagnostics/homograph_collapse.png`](../figures/diagnostics/homograph_collapse.png) *(Schematic: Why '6' and 'W' confuse a 2D camera)*
  - [`docs/figures/diagnostics/tri_panel_failure_comparisons.png`](../figures/diagnostics/tri_panel_failure_comparisons.png) *(Tri-panel: Ground Truth vs Test Crop vs Mistaken Sign)*
  - [`docs/figures/diagnostics/architectural_progression_montage.png`](../figures/diagnostics/architectural_progression_montage.png) *(Progression: M1 fails $\to$ M2 recovers $\to$ M3 resolves)*
  - [`docs/figures/diagnostics/bayes_error_bound_residuals.png`](../figures/diagnostics/bayes_error_bound_residuals.png) *(Irreducible monocular depth occlusions)*

---

#### Subsection IV.4: Controlled Scientific Ablation Suite
- **Topics**:
  - Three controlled scientific ablations isolating individual design variables:
    1. *Ablation 1: Data Partitioning Policy (Leakage)*: Subject Split (52.71%) vs Random Split (90.72%) on Model 1 (+38.01% artificial inflation); Model 3 saturates to 100.00% on random split. Proves that random splitting evaluates burst-frame memorization rather than generalization.
    2. *Ablation 2: Aspect-Ratio Preservation*: Letterbox Padding (52.71%) vs Naive Squashing (23.04%) on Model 1 (+29.67% delta); +6.01% gain on Model 3. Proves squashing distorts anatomical knuckle ratios.
    3. *Ablation 3: Label Smoothing Regularization*: Cross-Entropy ($\alpha=0.0$, 93.70% confidence) vs Label Smoothing ($\alpha=0.10$, 70.54% confidence). Dampens overconfidence by 23.16% while boosting unseen test accuracy (91.86% vs 91.18%).
- **Source Reports**: [`docs/reports/ablation_experiments.md`](ablation_experiments.md)
- **Source Code**: [`src/experiments/run_ablations.py`](../../src/experiments/run_ablations.py)
- **Source Metrics**: [`docs/metrics/ablation_results.json`](../metrics/ablation_results.json)

---

#### Subsection IV.5: External Real-World Benchmark & Handedness Parity
- **Topics**:
  - Zero-shot out-of-distribution evaluation on 180 mobile phone photos across 4 unseen human signers.
  - Cross-domain generalization: Model 1 = 28.33%, Model 2 = 61.11%, Model 3 = 80.00%. Proves that in-distribution random split models collapse under real-world domain shifts.
  - Handedness Parity Verification: Paired left-hand vs right-hand performance delta is restricted to 0.00% (Model 1), 2.78% (Model 2), and 8.33% (Model 3), verifying the efficacy of runtime stochastic horizontal flipping.
- **Source Reports**: [`docs/reports/external_benchmark.md`](external_benchmark.md)
- **Source Code**: [`src/experiments/run_external_benchmark.py`](../../src/experiments/run_external_benchmark.py)
- **Figures**: [`docs/figures/dataset/external_benchmark_composition.png`](../figures/dataset/external_benchmark_composition.png)

---

#### Subsection IV.6: Spatial Attribution & Interpretability via Grad-CAM
- **Topics**:
  1. *Mathematical Foundations*: Taylor expansion derivation of neuron importance weights $\alpha_k^c$ via spatial global average pooling of backward gradients; rectified linear combination $L_{\text{Grad-CAM}}^c$; explanation of why standard CAM fails for nonlinear MLP heads.
  2. *Architectural Attention Evolution*: Receptive field growth ($46\text{ px} \to 216\text{ px} \to 435\text{ px}$). Model 1 displays diffuse boundary bleeding; Model 2 bounds the active hand; Model 3 pinpoints discriminative joints, knuckle folds, and fingertips.
  3. *Counterfactual Error Attribution*: Dual-target attribution on failure pairs ($6$ vs $W$, $1$ vs $Z$) explaining *why* the network took the false branch.
  4. *Letterbox Invariance Verification*: 0.00% attribution leakage into neutral gray $(128, 128, 128)$ padding borders across disparate skin tones.
- **Source Reports**: [`docs/reports/model_interpretability.md`](model_interpretability.md)
- **Source Code**: [`src/utils/gradcam.py`](../../src/utils/gradcam.py) · [`src/experiments/generate_gradcam_panels.py`](../../src/experiments/generate_gradcam_panels.py)
- **Figures**:
  - [`docs/figures/interpretability/architectural_attribution_evolution.png`](../figures/interpretability/architectural_attribution_evolution.png)
  - [`docs/figures/interpretability/counterfactual_diagnostic_attribution.png`](../figures/interpretability/counterfactual_diagnostic_attribution.png)
  - [`docs/figures/interpretability/letterbox_invariance_verification.png`](../figures/interpretability/letterbox_invariance_verification.png)
  - [`docs/figures/interpretability/receptive_field_coverage.png`](../figures/interpretability/receptive_field_coverage.png)

---

### Section V. Conclusion & References

This section synthesizes the core findings, articulates physical vision boundaries, and cites relevant scholarly literature.

#### Subsections & Core Topics to Cover:
1. **Executive Findings**:
   - Progression from 52.71% (Simple CNN) to 82.90% (Mini-ResNet) to 91.86% (ResNet-18).
   - The critical role of data engineering: Aspect letterboxing (+29.7%) and runtime ambidexterity flipping.
   - Validation that 23 classes achieve 100% precision and recall on unseen signers.
2. **Physical Limitations & Future Directions**:
   - Monocular 2D sensing cannot resolve 3D depth occlusions ($6 \leftrightarrow W$). Future systems require RGB-D depth sensing or multi-view cameras.
   - Static single-frame architectures cannot disambiguate dynamic signs ($1 \leftrightarrow Z$, $I \leftrightarrow J$). Future work requires temporal sequence modeling (LSTM, ConvLSTM, or Video Vision Transformers).
3. **Scholarly Literature References**:
   - Include formal citations for ResNet (He et al., 2016), Grad-CAM (Selvaraju et al., 2017), Label Smoothing (Szegedy et al., 2016), MediaPipe Hands (Zhang et al., 2020), AdamW (Loshchilov & Hutter, 2019), and the ASL-HG Mendeley dataset (Al-Hammadi et al., 2020).

#### Source References:
- **Primary Reports**: [`docs/reports/project_overview.md`](project_overview.md) *(Section 4)* · [`docs/reports/error_diagnostics.md`](error_diagnostics.md) *(Section 3.3)*

---

## Quick-Access Document Mapping Matrix

| Course Report Section | Primary Documentation Source | Supporting Source Code | Key Figures to Embed |
| :--- | :--- | :--- | :--- |
| **Section I: Problem Description** | [`project_overview.md`](project_overview.md) | `src/config.py` | `vision_challenge_pipeline.png` |
| **Section II: Dataset Description** | [`eda_analysis.md`](eda_analysis.md), [`dataset_splits.md`](dataset_splits.md) | `src/data/splits.py` | `dataset_structure_overview.png`, `eda_36_class_grid.png`, `eda_volunteer_mosaic.png` |
| **Section III.1: Preprocessing & Splits** | [`roi_preprocessing.md`](roi_preprocessing.md), [`dataset_pipeline.md`](dataset_pipeline.md) | `src/data/preprocess_roi.py`, `src/data/dataloaders.py` | `roi_letterbox_workflow.png`, `letterbox_vs_squash.png`, `dataset_splits_structure.png` |
| **Section III.2: Simple CNN Design** | [`model1_architecture.md`](model1_architecture.md), [`models_architecture.md`](models_architecture.md) | `src/models/simple_cnn.py` | `model_1_architecture_detailed.png` |
| **Section III.3: Complex CNN Design** | [`model2_architecture.md`](model2_architecture.md), [`models_architecture.md`](models_architecture.md) | `src/models/complex_cnn.py` | `model_2_architecture_detailed.png` |
| **Section III.4: Transfer CNN Design** | [`model3_architecture.md`](model3_architecture.md), [`models_architecture.md`](models_architecture.md) | `src/models/transfer_cnn.py` | `model_3_architecture_detailed.png` |
| **Section IV.1: Rubric Benchmark Table** | [`models_architecture.md`](models_architecture.md), [`error_diagnostics.md`](error_diagnostics.md) | `src/training/engine.py` | Double-column LaTeX table from `error_diagnostics.md`, training curves |
| **Section IV.2: Per-Class Evaluation** | [`error_diagnostics.md`](error_diagnostics.md) | `src/training/metrics.py` | Confusion matrices (`simple`, `complex`, `transfer`) |
| **Section IV.3: Error Taxonomy & Bayes** | [`error_diagnostics.md`](error_diagnostics.md) | `src/utils/error_visualizer.py` | `homograph_collapse.png`, `tri_panel_failure_comparisons.png`, `architectural_progression_montage.png`, `bayes_error_bound_residuals.png` |
| **Section IV.4: Ablation Suite** | [`ablation_experiments.md`](ablation_experiments.md) | `src/experiments/run_ablations.py` | Ablation comparison tables |
| **Section IV.5: External Benchmark** | [`external_benchmark.md`](external_benchmark.md) | `src/experiments/run_external_benchmark.py` | `external_benchmark_composition.png` |
| **Section IV.6: Grad-CAM Attribution** | [`model_interpretability.md`](model_interpretability.md) | `src/utils/gradcam.py` | `architectural_attribution_evolution.png`, `counterfactual_diagnostic_attribution.png`, `letterbox_invariance_verification.png`, `receptive_field_coverage.png` |
| **Section V: Conclusion & References** | [`project_overview.md`](project_overview.md), [`error_diagnostics.md`](error_diagnostics.md) | — | — |
