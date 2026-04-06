# Fake Dataset Factory
## Synthetic Medical Chest X-Ray Generation using Generative AI

---

**Course:** CSET419 — Introduction to Generative AI
**Institution:** Bennett University
**Author:** Lakshya Rathi
**Date:** March 2026

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Introduction](#2-introduction)
3. [Literature Review](#3-literature-review)
4. [Methodology](#4-methodology)
5. [Implementation](#5-implementation)
6. [Results & Analysis](#6-results--analysis)
7. [Discussion](#7-discussion)
8. [Conclusion](#8-conclusion)
9. [Future Work](#9-future-work)
10. [References](#10-references)
11. [Appendix](#11-appendix)

---

## 1. Abstract

Medical imaging datasets are scarce, expensive to acquire, and subject to strict privacy regulations. This project addresses these challenges by implementing and comparing six state-of-the-art generative AI architectures for synthetic chest X-ray generation. We evaluate Stable Diffusion, DCGAN, WGAN-GP, VQ-VAE, DDPM, and Flow Matching on the Kaggle Chest X-Ray Pneumonia dataset using domain-adapted FID scores computed with torchxrayvision DenseNet121 features instead of standard Inception V3.

**Key Result:** Flow Matching v2 with attention layers and EMA achieves the best FID score of **6.20**, outperforming DDPM (8.96) and WGAN-GP (11.10). This demonstrates that modern flow-based generative models with proper architectural improvements can produce high-quality synthetic medical images suitable for data augmentation.

**Keywords:** Generative AI, Medical Imaging, Chest X-Ray, GAN, Diffusion Models, Flow Matching, Synthetic Data

---

## 2. Introduction

### 2.1 Background

Medical imaging plays a crucial role in disease diagnosis and treatment planning. However, creating labeled medical imaging datasets faces several challenges:

- **Data Scarcity:** Medical images require expensive equipment and expert radiologists for annotation
- **Privacy Concerns:** Patient data is protected by regulations like HIPAA, limiting data sharing
- **Class Imbalance:** Rare conditions are underrepresented in available datasets

Synthetic data generation offers a promising solution by creating unlimited labeled training data that preserves statistical properties of real medical images without privacy concerns.

### 2.2 Problem Statement

This project aims to:
1. Implement and compare 6 generative AI architectures for chest X-ray synthesis
2. Develop a domain-adapted evaluation methodology using medical imaging features
3. Create an exportable synthetic dataset with labels for downstream classification tasks
4. Build a user-friendly interface for generating synthetic medical images

### 2.3 Scope

- **Dataset:** Kaggle Chest X-Ray Pneumonia dataset (5,216 training images)
- **Resolution:** 64×64 grayscale (computational efficiency)
- **Models:** SD (img2img), DCGAN, WGAN-GP, VQ-VAE, DDPM, Flow Matching v2
- **Evaluation:** Domain-adapted FID + Proxy TSTR using torchxrayvision

---

## 3. Literature Review

### 3.1 Generative Adversarial Networks

**DCGAN (Radford et al., 2015)** introduced architectural guidelines for stable GAN training using strided convolutions, batch normalization, and LeakyReLU activations.

**WGAN-GP (Gulrajani et al., 2017)** improved training stability by replacing the Jensen-Shannon divergence with Wasserstein distance and enforcing the Lipschitz constraint through gradient penalty.

### 3.2 Variational Autoencoders

**VQ-VAE (van den Oord et al., 2017)** combines variational inference with discrete latent representations through vector quantization, achieving sharper reconstructions than continuous VAEs.

### 3.3 Diffusion Models

**DDPM (Ho et al., 2020)** formulates generation as iterative denoising, achieving state-of-the-art image quality. The model learns to reverse a gradual noising process over 1000 timesteps.

### 3.4 Flow Matching

**Conditional Flow Matching (Lipman et al., 2022)** provides a simulation-free training objective for continuous normalizing flows. The optimal transport formulation creates straighter generative paths, enabling faster sampling.

**MOTFM (2025)** demonstrated that flow matching with attention layers achieves excellent results on medical imaging tasks, inspiring our v2 architecture improvements.

### 3.5 Medical Image Generation

**torchxrayvision (Cohen et al., 2022)** provides pretrained models on multiple chest X-ray datasets (NIH, CheXpert, MIMIC-CXR), enabling domain-specific feature extraction and evaluation.

---

## 4. Methodology

### 4.1 Dataset

**Source:** Kaggle Chest X-Ray Pneumonia Dataset

| Split | Normal | Pneumonia | Total |
|-------|--------|-----------|-------|
| Train | 1,341 | 3,875 | 5,216 |
| Test | 234 | 390 | 624 |

**Preprocessing:**
- Resize to 64×64 pixels
- Convert to grayscale (1 channel)
- Normalize to [-1, 1] for GANs/diffusion or [0, 1] for VQ-VAE

### 4.2 Model Architectures

#### 4.2.1 DCGAN / WGAN-GP
- Generator: 5 ConvTranspose2d layers (100 → 64×64×1)
- Discriminator/Critic: 5 Conv2d layers with LeakyReLU
- WGAN-GP adds gradient penalty (λ=10) and removes batch norm from critic

#### 4.2.2 VQ-VAE
- Encoder: Conv layers (64×64 → 16×16 latent)
- Vector Quantizer: 512 codebook vectors, 64-dim embeddings
- Decoder: ConvTranspose layers (16×16 → 64×64)
- Loss: Reconstruction + Codebook + Commitment (β=0.25)

#### 4.2.3 DDPM
- UNet2DModel from HuggingFace diffusers
- Block channels: [64, 128, 256, 256]
- Attention at 16×16 resolution
- 1000 diffusion timesteps, linear schedule

#### 4.2.4 Flow Matching v2 (Best Model)
- Custom UNet with self-attention at 16×16 and 8×8 resolutions
- Sinusoidal time embeddings
- ResBlocks with GroupNorm + SiLU
- EMA (decay=0.999) for stable generation
- Adaptive ODE solver (dopri5) for sampling

### 4.3 Evaluation Metrics

#### 4.3.1 Domain-Adapted FID

Standard FID uses Inception V3 features trained on ImageNet, which is suboptimal for medical images. We extract features from **torchxrayvision DenseNet121** pretrained on:
- NIH ChestX-ray14
- CheXpert
- MIMIC-CXR
- Kaggle Pneumonia

This provides medically meaningful feature representations.

#### 4.3.2 Proxy TSTR

True TSTR (Train on Synthetic, Test on Real) requires training a classifier. We use a proxy:
1. Label synthetic images using pretrained torchxrayvision
2. Check if "Lung Opacity" score > 0.5
3. Report percentage classified as positive

---

## 5. Implementation

### 5.1 Training Configuration

| Model | Epochs | Batch Size | LR | Special |
|-------|--------|------------|-----|---------|
| DCGAN | 50 | 32 | 2e-4 | β1=0.5 |
| WGAN-GP | 50 | 32 | 1e-4 | λ=10, n_critic=5 |
| VQ-VAE | 50 | 32 | 1e-3 | β=0.25, K=512 |
| DDPM | 50 | 32 | 1e-4 | T=1000 |
| Flow Matching v2 | 100 | 32 | 1e-4 | EMA=0.999, attention |

### 5.2 Code Structure

```
fake-dataset-factory/
├── notebooks/
│   ├── 01_stable_diffusion.ipynb
│   ├── 02_dcgan.ipynb
│   ├── 03_wgan_gp.ipynb
│   ├── 04_vqvae.ipynb
│   ├── 05_ddpm.ipynb
│   ├── 06_flow_matching.ipynb
│   ├── 06_flow_matching_v2.ipynb   ← Best model
│   └── 07_comparison.ipynb
├── app.py                          ← Gradio UI
└── data/
    ├── normal/
    └── pneumonia/
```

### 5.3 Compute Resources

- **Platform:** Kaggle Notebooks
- **GPU:** NVIDIA Tesla T4 (16GB VRAM)
- **Training time:** 10-30 minutes per model (100 epochs for Flow Matching v2)

---

## 6. Results & Analysis

### 6.1 Quantitative Results

| Rank | Model | FID ↓ | TSTR | Training Time |
|------|-------|-------|------|---------------|
| 🥇 | **Flow Matching v2** | **6.20** | 100% | ~25 min |
| 🥈 | DDPM | 8.96 | 100% | ~15 min |
| 🥉 | WGAN-GP | 11.10 | 100% | ~10 min |
| 4 | DCGAN | 15.24 | 100% | ~8 min |
| 5 | VQ-VAE | 46.59 | 100% | ~10 min |
| 6 | Stable Diffusion | 94.71 | 100% | ~20 min |

### 6.2 FID Comparison

![FID Comparison](../notebooks/outputs/fid_comparison.png)

### 6.3 Visual Comparison

![Samples Comparison](../notebooks/outputs/samples_comparison.png)

### 6.4 Analysis

**Flow Matching v2** achieves the best results due to:
1. **Attention layers** capture long-range anatomical dependencies
2. **EMA** stabilizes generation quality
3. **Adaptive ODE solver** provides accurate trajectory integration
4. **100 epochs** allows full convergence

**DDPM** performs well but requires many denoising steps (1000).

**GANs** (WGAN-GP, DCGAN) offer a good quality-to-speed tradeoff.

**VQ-VAE** suffers from random codebook sampling without a prior model.

**Stable Diffusion** (img2img) has high FID because it uses real images as input.

---

## 7. Discussion

### 7.1 Why Flow Matching Outperforms Diffusion

1. **Straighter paths:** Optimal transport creates more direct generative trajectories
2. **Fewer steps needed:** Achieves quality with ~10 ODE steps vs 1000 diffusion steps
3. **Attention benefits:** Self-attention captures rib structures and lung boundaries better

### 7.2 Limitations of TSTR Metric

All models achieve 100% TSTR because torchxrayvision is sensitive to X-ray-like textures, not diagnostic quality. This metric does not discriminate between models for this task.

### 7.3 Practical Considerations

For real-world deployment:
- **GANs** are fastest for inference (~10ms per image)
- **Flow Matching v2** provides best quality (~100ms per image)
- **DDPM** is slowest (~2s per image) but robust

---

## 8. Conclusion

This project successfully demonstrates that:

1. **Flow Matching v2 achieves state-of-the-art FID of 6.20** for synthetic chest X-ray generation
2. **Architectural improvements** (attention, EMA) matter more than raw training time
3. **Domain-adapted evaluation** using torchxrayvision features provides meaningful medical image comparisons
4. **GANs remain competitive** with faster training and inference times

The generated synthetic images can augment real datasets for training downstream classifiers, addressing data scarcity in medical imaging.

---

## 9. Future Work

### 9.1 Immediate Improvements
- Scale to 256×256 or 512×512 resolution
- Add PixelCNN prior for VQ-VAE generation
- Implement class-conditional generation
- True TSTR evaluation with trained classifier

### 9.2 Extended Research
- Multi-class generation (Normal, Bacterial Pneumonia, Viral Pneumonia)
- Fine-tune Stable Diffusion on chest X-ray data
- Cross-dataset evaluation (CheXpert, MIMIC-CXR)
- Clinical expert evaluation

### 9.3 Deployment
- HuggingFace Spaces deployment
- Model quantization for CPU inference
- API for integration with medical imaging pipelines

---

## 10. References

1. Goodfellow, I., et al. (2014). Generative Adversarial Networks. NeurIPS.
2. Radford, A., et al. (2015). Unsupervised Representation Learning with DCGANs. ICLR.
3. Gulrajani, I., et al. (2017). Improved Training of Wasserstein GANs. NeurIPS.
4. van den Oord, A., et al. (2017). Neural Discrete Representation Learning. NeurIPS.
5. Ho, J., et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
6. Lipman, Y., et al. (2022). Flow Matching for Generative Modeling. ICLR.
7. Cohen, J., et al. (2022). torchxrayvision: A library of chest X-ray datasets and models.
8. Rombach, R., et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.
9. Medical Optimal Transport Flow Matching (2025). arXiv:2503.00266.

---

## 11. Appendix

### A. Hyperparameter Details

| Parameter | DCGAN | WGAN-GP | VQ-VAE | DDPM | FM v2 |
|-----------|-------|---------|--------|------|-------|
| Optimizer | Adam | Adam | Adam | Adam | AdamW |
| β1 | 0.5 | 0.0 | 0.9 | 0.9 | 0.9 |
| β2 | 0.999 | 0.9 | 0.999 | 0.999 | 0.999 |
| Weight Decay | 0 | 0 | 0 | 0 | 1e-4 |
| LR Schedule | - | - | - | - | Cosine |

### B. Model Parameters

| Model | Parameters |
|-------|------------|
| DCGAN Generator | ~3.5M |
| WGAN-GP Generator | ~3.5M |
| VQ-VAE | ~5.2M |
| DDPM UNet | ~14M |
| Flow Matching v2 UNet | ~18M |

### C. Sample Outputs

Generated samples from each model are saved in `notebooks/outputs/<model>/images/`.

---

*Submitted as part of CSET419 coursework, Bennett University, March 2026*
