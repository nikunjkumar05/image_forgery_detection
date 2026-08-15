# LMS-FNet Architecture Guide

> *"Professor, while modern State-of-the-Art models like ForMa (2025) rely on heavy Vision Mamba architectures to implicitly learn manipulations, our model—**LMS-FNet**—takes an explicit, physics-informed approach. We extract multi-spectral compression errors (MQ-ELA) at three distinct JPEG quality levels, pass them through a lightweight dual-stream DenseNet121 backbone, and fuse the spatial and error features using Bidirectional Cross-Attention. This allows our model to achieve competitive accuracy on CASIA v2 using only ~16 Million parameters—a significant reduction compared to 2025 literature."*

---

## Architecture Overview

```
Input: Raw Image (224×224×3) + ELA Image (224×224×3)
                    │                          │
        ┌───────────┴───────────┐    ┌─────────┴─────────┐
        │    DenseNet121        │    │   DenseNet121     │
        │  (Feature Extractor)  │    │ (Error Extractor) │
        └───────────┬───────────┘    └─────────┬─────────┘
                    │                          │
            GlobalAvgPool2D             GlobalAvgPool2D
                    │                          │
            Dense(256, ReLU)           Dense(256, ReLU)
                    │                          │
                    └──────────┬───────────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Cross-Attention    │
                    │  (RGB ↔ ELA)        │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │  Product + Diff     │
                    │  (Fusion)           │
                    └──────────┬──────────┘
                               │
                    Concatenate → 768-dim
                               │
                    Dense(256, GELU) → Dropout(0.3)
                               │
                    Dense(128, GELU) → Dropout(0.2)
                               │
                    Dense(1, Sigmoid) → Forgery Probability
```

---

## 1. Symmetrical Dual-Stream Backbone (`DenseNet121`)

* **What it does:** We use two separate `DenseNet121` networks (without the top classification layers). 

* **Why DenseNet121:**
  - **Parameter Efficiency:** ~8M params per backbone vs ~12M for EfficientNetB3
  - **Feature Reuse:** Dense connections enable feature reuse across layers
  - **Gradient Flow:** Dense connections alleviate vanishing gradient problems
  - **Total params:** ~16M (36% reduction from EfficientNetB3 version)

* **Stream 1 (Raw):** Takes the original RGB image and learns spatial textures, color patterns, and semantic content.

* **Stream 2 (ELA):** Takes the MQ-ELA error image and learns compression artifacts, quantization differences, and manipulation signatures.

---

## 2. Multi-Spectral Error Level Analysis (MQ-ELA)

Instead of traditional single-quality ELA, we compute ELA at three JPEG quality levels:

| Channel | Quality | What it captures |
|---------|---------|------------------|
| R | Q=75 | High compression artifacts |
| G | Q=85 | Medium compression artifacts |
| B | Q=95 | Low compression artifacts |

**Why this works:** Different JPEG quality levels reveal different manipulation patterns. Splicing often leaves artifacts at specific quality levels, while copy-move forgery shows different error patterns.

---

## 3. Bidirectional Cross-Attention Fusion

```
RGB Features ──→ Q_r ──┐
ELA Features ──→ K_e ──┼──→ Cross-Attention (r→e)
ELA Features ──→ V_e ──┘

ELA Features ──→ Q_e ──┐
RGB Features ──→ K_r ──┼──→ Cross-Attention (e→r)
RGB Features ──→ V_r ──┘

Output = Concat(CrossAttn(r→e), CrossAttn(e→r))
```

* **RGB→ELA Attention:** "What ELA features should I attend to given these RGB features?"
* **ELA→RGB Attention:** "What RGB features should I attend to given these ELA features?"
* **Bidirectional:** Both streams inform each other, creating a rich fused representation.

---

## 4. Additional Fusion Mechanisms

After Cross-Attention, we add:

1. **Element-wise Product:** Captures similarity between streams
2. **Element-wise Difference:** Captures contrast between streams
3. **Concatenate:** Combines all three signals → 768-dim vector

---

## 5. Classification Head

```
768-dim fused features
    ↓
Dense(256, GELU) → Dropout(0.3)    # Prevent overfitting
    ↓
Dense(128, GELU) → Dropout(0.2)    # Further compression
    ↓
Dense(1, Sigmoid)                   # Binary output
```

---

## Key Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Input Size | 224×224×3 | Standard CNN input |
| Backbone | DenseNet121 | Pre-trained on ImageNet |
| Embed Dim | 256 | Cross-attention dimension |
| Attention Heads | 4 | Multi-head attention |
| Dropout | 0.3, 0.2 | Regularization |
| Loss | OHEM Focal Loss | Hard sample mining |

---

## Q&A

**Q1: "Why DenseNet121 and not a Vision Transformer (ViT) or Mamba (like ForMa)?"**

> *"Parameter efficiency. ViTs require 86M+ parameters and scale quadratically, making them computationally heavy for large 2D images. ForMa uses 37M parameters. By using DenseNet121 combined with our specialized Cross-Attention Fusion, we achieve a global receptive field while keeping the parameter count at just ~16M—a significant reduction with competitive ROC-AUC."*

**Q2: "Why use ELA as a separate input instead of letting the network learn compression artifacts from RGB?"**

> *"ELA acts as an explicit inductive bias. Forcing the network to learn compression artifacts solely from RGB requires massive capacity (e.g., 50M+ parameters). Providing ELA as a secondary modality allows our 16M parameter model to match heavier models. Our ablation study confirms this: RGB-only achieves 52.35% accuracy, while adding ELA boosts it to 88.45%."*

**Q3: "Why bidirectional cross-attention instead of simple concatenation?"**

> *"Simple concatenation treats both streams equally. Cross-attention allows each stream to selectively attend to the most relevant features from the other stream. RGB can focus on ELA regions with high compression differences, while ELA can focus on RGB regions with suspicious textures."*

---

## Model Summary

```
Total Parameters: ~16M
├── DenseNet121 (Raw):     ~8M
├── DenseNet121 (ELA):     ~8M
├── Cross-Attention:       ~0.5M
├── Fusion Layers:         ~0.2M
└── Classification Head:   ~0.1M
```

**Input:** Raw Image (224×224×3) + ELA Image (224×224×3)  
**Output:** Forgery Probability (0-1)  
**Backbone:** DenseNet121 (dual-stream)  
**Fusion:** Bidirectional Cross-Attention + Product + Difference
