# AVLT: Paper vs. Code Analysis

This document outlines the discrepancies between the research paper "Adaptive Vision-Language Transformer for Multimodal CNS Tumor Detection" and its accompanying source code.

## 1. Data Requirements & Preprocessing

### Paper Description
*   **Datasets**: BraTS (285 subjects), TCGA-GBM/LGG (600 subjects), REMBRANDT (500 subjects), GLASS (300 subjects).
*   **Input Modalities**:
    *   **MRI**: Multi-sequence 3D volumes ($T_1, T_{1c}, T_2, FLAIR$).
    *   **Clinical Text**: Pathology reports, clinical notes (tokenized with BioBERT).
    *   **Metadata**: Age, tumor grade (normalized).
*   **Preprocessing Pipeline**:
    1.  Co-registration to common anatomical space.
    2.  Skull-stripping to remove non-brain tissue.
    3.  N4 Bias Field Correction.
    4.  Intensity Normalization: Zero mean, unit variance.
    5.  Rescaling: Min-max normalization to $[0, 1]$.
    6.  **Slicing**: 3D volumes are sliced into 2D images ($224 \times 224$). The paper mentions "fused volumetric representation" via channel-wise concatenation.
*   **3D Context**: The paper claims the vision branch uses a backbone to describe "local and global spatial interdependence across 3D MRI volumes".

### Code Implementation
*   **Dataset Implementation**: `src/avlt/data/dataset.py` contains **only** a `SyntheticMultimodalDataset`. There is **no code** to load real MRI files (e.g., NIfTI), parse clinical text, or handle metadata.
*   **Preprocessing**: **Completely missing**. There are no scripts for skull-stripping, bias correction, or slicing 3D volumes.
*   **Input Format**: The `VisionEncoder` in `src/avlt/models/encoders.py` expects **2D inputs** of shape $(B, 4, 224, 224)$. The 4 channels likely correspond to the 4 MRI sequences ($T_1, T_{1c}, T_2, FLAIR$).
*   **Discrepancy**: The code treats the problem as a **2D classification task** (likely slice-based), whereas the paper implies a more complex 3D-aware architecture. The "3D dependence" claimed in the paper is not explicitly modeled in the provided ViT implementation, derived from standard 2D `timm` models.

## 2. Vision Encoder Architecture

### Paper Description
*   **Backbone**: Explicitly states "hierarchical **Swin-Transformer** backbone" (Section 2.3, verified via OCR).
*   **Diagrams**: Figure 2 and Equation 4 describe a standard **ViT** (Vision Transformer).

### Code Implementation
*   **Backbone**: Uses **Standard ViT** (`vit_base_patch16_224` from `timm`).
*   **Status**: The code matches the diagrams/equations but contradicts the text.

## 3. Adaptive Normalization Module (ANM)

### Paper Description
*   **Goal**: Domain generalization by re-centering/scaling features using domain-specific statistics ($\mu_d, \sigma_d$).
*   **Location**: Placed after each transformer block (Section 2.3).
*   **Status**: **CRITICAL FEATURE**.

### Code Implementation
*   **Status**: **MISSING**. There is no implementation of ANM in `src/avlt/models/` or `src/avlt/modules/`. The code uses standard LayerNorm.

## 4. Loss Functions

### Paper Description
*   **Optimization**: $L_{total} = L_{cls} + \lambda_1 L_{mlm} + \lambda_2 L_{align}$
    *   $L_{mlm}$: Masked Language Modeling loss (Section 2.6).
    *   $L_{align}$: Contrastive alignment loss.
    *   $L_{cls}$: Cross-entropy classification loss.

### Code Implementation
*   **Losses**: Implements `Losses.total` in `src/avlt/train/losses.py`.
    *   Includes: Classification Loss, Self-Distillation Loss (not in paper's primary eq).
    *   **MISSING**: **MLM Loss** ($L_{mlm}$) is not implemented.

## 5. Token Length Constraint

### Paper Description
*   **text**: "The max token length was set to **256**" (Section 2.4).
*   **Reasoning**: Likely a trade-off for GPU memory when training with 3D volumes.

### Code Implementation
*   **Synthetic Data**: `src/avlt/data/dataset.py` sets `max_length=128`.
*   **Discrepancy**: Code is even more restrictive (128 vs 256). Both are short for full clinical reports, necessitating aggressive truncation or summarization.

## 6. Fusion Mechanism

### Paper Description
*   **Eq 14**: $z_{fused} = \alpha z^*_v + (1 - \alpha) z^*_l$ (Simple convex combination).
*   **Figure 3**: $f_{fused} = \alpha f_v + \beta f_t$ (Weighted sum).

### Code Implementation
*   **Logic**: `f_fused = alpha * f_v + beta * f_t + fused`
    *   Includes the direct attention output (`fused`) in addition to the weighted branches.
    *   **Status**: More complex than Eq 14, essentially a superset of Figure 3's logic.

## Recommendation for Next Steps
1.  **Data Ingestion**: A data loading pipeline is required to handle NIfTI files and preprocess them (slice extraction) to match the $(4, 224, 224)$ expected input.
2.  **Implementation**: To match the paper, the **ANM** module and **MLM Loss** need to be implemented.
