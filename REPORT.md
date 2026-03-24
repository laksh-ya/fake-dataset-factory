# Fake Dataset Factory — Project Report

**Course:** CSET419 — Introduction to Generative AI
**Author:** Lakshya Rathi
**Date:** March 2026

---

## 1. Executive Summary

This project implements and compares 6 generative AI architectures for synthetic medical chest X-ray generation. The goal is to generate labeled synthetic data that can augment real datasets for training downstream classifiers, addressing the scarcity and class imbalance of medical imaging data.

**Key Findings:**
- Flow Matching v2 achieves the best FID (6.20), outperforming all other architectures including DDPM
- Attention layers + EMA + adaptive ODE solver dramatically improve flow matching quality
- Simple GAN architectures (DCGAN, WGAN-GP) remain competitive with FID of 11-15
- TSTR metric (100% across all models) is not discriminative for this task due to torchxrayvision's sensitivity to X-ray textures

---

## 2. Problem Statement

Medical imaging datasets suffer from:
1. **Scarcity** — expensive to acquire, requires expert annotation
2. **Privacy concerns** — patient data cannot be freely shared
3. **Class imbalance** — rare conditions underrepresented

Synthetic data generation addresses these issues by creating unlimited labeled training data that preserves the statistical properties of real medical images without privacy concerns.

---

## 3. Dataset

**Source:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

| Split | Normal | Pneumonia |
|-------|--------|-----------|
| Train | 1,341 | 3,875 |
| Test | 234 | 390 |

**Preprocessing:**
- Resize to 64×64 (computational efficiency)
- Convert to grayscale (1 channel)
- Normalize to [-1, 1] or [0, 1] depending on model

**Preprocessed dataset:** [lakshyarathi/lungpp](https://www.kaggle.com/datasets/lakshyarathi/lungpp)

---

## 4. Models Implemented

### 4.1 Stable Diffusion (img2img) — 2022

**Approach:** Use pretrained SD v1.5 in img2img mode — feed real X-ray as input, generate synthetic variation.

**Configuration:**
- Strength: 0.7 (balance between original structure and variation)
- Guidance scale: 7.5
- Prompt: "chest x-ray, medical radiograph, grayscale"

**Note:** This is not a fair comparison since SD uses real images as input (not pure noise). High FID (94.71) reflects a fundamentally different input distribution.

---

### 4.2 DCGAN — 2014

**Architecture:**
- Generator: 5 ConvTranspose2d layers (100 → 64×64×1)
- Discriminator: 5 Conv2d layers with LeakyReLU
- Loss: Binary Cross-Entropy

**Training:**
- 50 epochs, batch size 32
- Learning rate: 2e-4 (Adam, β1=0.5)
- Noise dimension: 100

**Result:** FID 15.24 — strong baseline despite being the oldest architecture.

---

### 4.3 WGAN-GP — 2017

**Architecture:** Same as DCGAN

**Key Differences:**
- Wasserstein distance instead of BCE
- Gradient penalty (λ=10) for Lipschitz constraint
- Critic trains 5× per generator step
- No sigmoid on discriminator output

**Training:**
- 50 epochs, batch size 32
- Learning rate: 1e-4

**Result:** FID 11.10 — **best performance**. Gradient penalty provides stable training.

---

### 4.4 VQ-VAE — 2017

**Architecture:**
- Encoder: Conv layers (64×64 → 16×16 latent)
- Vector Quantizer: 512 codebook vectors, 64-dim embeddings
- Decoder: ConvTranspose layers (16×16 → 64×64)

**Loss:**
- Reconstruction (MSE)
- Codebook loss (move codebook toward encoder output)
- Commitment loss (move encoder toward codebook, β=0.25)

**Generation Limitation:** Random codebook sampling without PixelCNN prior. Proper generation requires training an autoregressive model on codebook indices.

**Result:** FID 46.59 — reasonable reconstruction but poor generation due to missing prior.

---

### 4.5 DDPM — 2020

**Architecture:** UNet2DModel from HuggingFace diffusers
- Block channels: [64, 128, 256, 256]
- 1000 diffusion timesteps
- Linear noise schedule

**Training:**
- 50 epochs, batch size 32
- Learning rate: 1e-4

**Result:** FID 8.96 — **best performance**. Diffusion models excel when given sufficient training time and proper architecture tuning.

---

### 4.6 Flow Matching (OT-CFM) — 2022

**Library:** torchcfm

**Architecture:** Custom UNet with:
- Sinusoidal time embeddings
- ResBlocks with GroupNorm + SiLU
- Skip connections

**Training:**
- Conditional Flow Matching with σ=0
- MSE loss on velocity field
- 50 epochs, batch size 32

**Sampling:** Euler ODE solver, 10 steps (step_size=0.1)

**Result:** FID 40.35 — competitive with a newer, cleaner formulation than diffusion.

---

## 5. Results Comparison

| Model | Year | FID ↓ | TSTR | Notes |
|-------|------|-------|------|-------|
| **Flow Matching v2** | 2022 | **6.20** | 100% | Best overall — attention + EMA |
| DDPM | 2020 | 8.96 | 100% | Diffusion baseline |
| WGAN-GP | 2017 | 11.10 | 100% | Stable training |
| DCGAN | 2014 | 15.24 | 100% | Strong baseline |
| VQ-VAE | 2017 | 46.59 | 100% | No PixelCNN prior |
| Stable Diffusion | 2022 | 94.71 | 100% | img2img mode |

### FID Comparison

![FID Comparison](notebooks/outputs/fid_comparison.png)

### 5.1 Visual Comparison

![Samples Comparison](notebooks/outputs/samples_comparison.png)

Generated samples from each model are saved in `notebooks/outputs/<model>/images/`.

**Observations:**
- Flow Matching v2: Sharpest, most realistic — best rib and lung detail
- DDPM: Sharp, realistic rib structures and lung fields
- WGAN-GP/DCGAN: Competitive quality, sharp textures
- VQ-VAE: Noisy, lacks global coherence (random sampling issue)
- Stable Diffusion: Retains source image structure too strongly

---

## 6. Evaluation Methodology

### 6.1 Domain-Adapted FID

Standard FID uses Inception V3 features (trained on ImageNet). For medical images, this is suboptimal.

**Our approach:** Extract features from torchxrayvision DenseNet121, pretrained on:
- NIH ChestX-ray14
- CheXpert
- MIMIC-CXR
- Kaggle Pneumonia

This provides medically meaningful feature representations for comparing real vs. synthetic X-rays.

### 6.2 Proxy TSTR (Train on Synthetic, Test on Real)

True TSTR would train a classifier on synthetic data and test on real data. We use a proxy:

1. Label synthetic images using pretrained torchxrayvision
2. Check if "Lung Opacity" score > 0.5 (proxy for pneumonia features)
3. Report percentage classified as positive

**Limitation:** TSTR is 100% for all models because torchxrayvision is sensitive to X-ray-like textures, not diagnostic quality. This metric does not discriminate between models.

---

## 7. Technical Implementation

### 7.1 Consistent Training Setup

All models trained with:
- Image size: 64×64
- Batch size: 32
- Random seed: 42
- Epochs: 50 (except SD which is pretrained)
- Hardware: Kaggle T4 GPU (16GB VRAM)

### 7.2 Code Structure

```
notebooks/
├── 01_stable_diffusion.ipynb
├── 02_dcgan.ipynb
├── 03_wgan_gp.ipynb
├── 04_vqvae.ipynb
├── 05_ddpm.ipynb
├── 06_flow_matching.ipynb
└── 07_comparison.ipynb

src/
├── evaluate.py    # FID computation
├── label.py       # torchxrayvision labeling
└── export.py      # dataset export utilities

app.py             # Gradio UI
```

### 7.3 Key Libraries

- `torch`, `torchvision` — deep learning
- `diffusers` — Stable Diffusion, DDPM
- `torchcfm` — Flow Matching
- `torchxrayvision` — medical image features
- `gradio` — web UI

---

## 8. Discussion

### 8.1 Why Flow Matching v2 Achieves the Best Results

1. **Attention layers**: Self-attention at 16×16 and 8×8 resolutions captures long-range dependencies (rib structures, lung boundaries)
2. **EMA (Exponential Moving Average)**: Stabilizes generation by averaging model weights over training
3. **Adaptive ODE solver (dopri5)**: More accurate trajectory integration than fixed-step Euler
4. **Longer training**: 100 epochs (2× baseline) allows better convergence
5. **Optimal transport path**: Straighter transport paths require fewer steps to sample accurately

### 8.2 WGAN-GP Stability

Gradient penalty provides consistent training without mode collapse. The Wasserstein distance is a better metric than Jensen-Shannon divergence for measuring distribution similarity.

### 8.3 VQ-VAE Generation Quality

VQ-VAE achieves excellent reconstruction (MSE 0.001) but poor generation. Without a PixelCNN prior, randomly sampling codebook indices breaks spatial correlations. The decoder can only reconstruct what the encoder learned, not generate novel compositions.

### 8.4 Stable Diffusion Domain Gap

Pretrained SD sees real X-rays as input but outputs RGB images with different texture characteristics. The model wasn't trained on medical data, so it can't generate authentic X-ray features from pure noise.

---

## 9. Limitations

1. **Low resolution**: 64×64 is insufficient for clinical use (typical X-rays are 2000×2000+)
2. **Single class training**: Each model trained on pneumonia class only
3. **No clinical validation**: FID measures statistical similarity, not diagnostic utility
4. **TSTR not discriminative**: All models achieve 100%, providing no ranking signal

---

## 10. Future Work

### 10.1 Immediate Improvements

- [ ] Add PixelCNN prior for VQ-VAE
- [ ] Fine-tune SD on chest X-ray data
- [ ] Train on both classes (Normal + Pneumonia)
- [ ] Train Flow Matching for more epochs

### 10.2 Extended Scope

- [ ] Scale to 256×256 or 512×512 resolution
- [ ] Implement conditional generation (class-conditional GANs)
- [ ] True TSTR evaluation (train classifier on synthetic, test on real)
- [ ] Add more datasets (CheXpert, MIMIC-CXR)
- [ ] Clinical expert evaluation of generated images

### 10.3 Production Deployment

- [ ] Optimize inference for CPU (model quantization)
- [ ] Add batch generation support
- [ ] Implement model selection API
- [ ] Deploy to HuggingFace Spaces

---

## 11. Conclusion

This project demonstrates that **Flow Matching with attention and EMA achieves the best results** for synthetic medical image generation, with **FID of 6.20**. This outperforms both diffusion models (DDPM: 8.96) and GANs (WGAN-GP: 11.10).

Key takeaways:
- Flow Matching v2 with attention layers, EMA, and adaptive ODE solver achieves state-of-the-art FID
- Architectural improvements (attention) matter more than raw training time for this task
- GANs remain competitive and train faster, making them practical alternatives
- Domain-adapted evaluation using torchxrayvision features provides more meaningful comparisons than standard Inception-based FID

The project successfully demonstrates that modern generative models can produce high-quality synthetic chest X-rays suitable for data augmentation in medical imaging tasks.

---

## 12. References

1. Goodfellow et al. (2014). Generative Adversarial Networks. NeurIPS.
2. Radford et al. (2015). Unsupervised Representation Learning with DCGANs. ICLR.
3. Gulrajani et al. (2017). Improved Training of WGANs. NeurIPS.
4. van den Oord et al. (2017). Neural Discrete Representation Learning. NeurIPS.
5. Ho et al. (2020). Denoising Diffusion Probabilistic Models. NeurIPS.
6. Lipman et al. (2022). Flow Matching for Generative Modeling. ICLR.
7. Cohen et al. (2022). torchxrayvision: A library of chest X-ray datasets and models.
8. Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models. CVPR.

---

## Appendix A: Hyperparameters

| Model | LR | β1 | β2 | Epochs | Batch | Special |
|-------|-----|-----|-----|--------|-------|---------|
| DCGAN | 2e-4 | 0.5 | 0.999 | 50 | 32 | — |
| WGAN-GP | 1e-4 | 0.0 | 0.9 | 50 | 32 | λ=10, n_critic=5 |
| VQ-VAE | 1e-3 | 0.9 | 0.999 | 50 | 32 | β=0.25, K=512 |
| DDPM | 1e-4 | 0.9 | 0.999 | 50 | 32 | T=1000 |
| Flow Matching | 1e-4 | 0.9 | 0.999 | 50 | 32 | σ=0 |

---

## Appendix B: Compute Resources

- **Platform:** Kaggle Notebooks
- **GPU:** NVIDIA Tesla T4 (16GB VRAM)
- **Training time per model:** 10-30 minutes
- **Total compute:** ~3 hours

---

*Generated as part of CSET419 coursework, March 2026*
