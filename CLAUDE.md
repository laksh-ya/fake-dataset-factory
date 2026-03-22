# CLAUDE.md — Fake Dataset Factory

## what you are building
An end-to-end synthetic medical chest X-ray generation and evaluation tool.
6 generative AI models. Same dataset. Same evaluation. Exportable labeled dataset. Gradio UI.

Read arch.md for full architecture. Read rules.md before writing any code.

---

## dataset
- Local path: chest_xray/ (already in project folder)
- Kaggle path: /kaggle/input/chest-xray-pneumonia/chest_xray/
- Use these folders:
  - chest_xray/train/NORMAL      → 1341 images
  - chest_xray/train/PNEUMONIA   → 3875 images
  - chest_xray/test/NORMAL       → 234 images  (use for TSTR testing)
  - chest_xray/test/PNEUMONIA    → 390 images  (use for TSTR testing)
  - val/ has only 8 images — ignore completely
- Preprocess: resize all to 64x64 grayscale, save to data/normal/ and data/pneumonia/
- Start with data/prepare.ipynb

---

## build order (strict — do not skip)
1. data/prepare.ipynb
2. notebooks/01_stable_diffusion.ipynb
3. notebooks/02_dcgan.ipynb
4. notebooks/03_wgan_gp.ipynb
5. notebooks/04_vqvae.ipynb
6. notebooks/05_ddpm.ipynb
7. notebooks/06_flow_matching.ipynb
8. notebooks/07_comparison.ipynb
9. src/ — extract shared code from notebooks into modules
10. app.py — Gradio UI
11. Deploy to HuggingFace Spaces

---

## every notebook has the same structure (no exceptions)

```
## Setup
## Dataset Loading
## Model Architecture
## Training
## Generate 100 Images
## Evaluate — FID (domain-adapted, DenseNet121 features)
## Evaluate — Label + TSTR (torchxrayvision)
## Export (images/ + labels.csv + metrics.json)
## Results
```

---

## evaluation — critical rules
- FID always computed FIRST before labeling or TSTR
- FID uses torchxrayvision DenseNet121 features, NOT Inception V3
- TSTR uses torchxrayvision as the test classifier
- Call it "domain-adapted FID" and "proxy TSTR" in all output and comments
- 100 images per model per class — not more, not less

---

## the 6 models

### 01 — Stable Diffusion (img2img)
- model: runwayml/stable-diffusion-v1-5
- use img2img mode, feed real xray as input, generate synthetic variation
- strength=0.6 to 0.8 (keeps xray structure, adds variation)
- note in notebook: "FID comparison exploratory — different input distribution"

### 02 — DCGAN
- vanilla baseline, BCE loss
- generator: ConvTranspose2d layers, input noise dim=100
- discriminator: Conv2d + LeakyReLU
- train 50 epochs minimum
- reference: kaledhoshme123/Using-GAN-to-Generate-Chest-X-Ray-Images

### 03 — WGAN-GP
- same architecture as DCGAN
- loss: Wasserstein distance + gradient penalty (lambda=10)
- critic trains 5x per generator step
- no sigmoid on discriminator output
- reference: aladdinpersson/Machine-Learning-Collection WGAN-GP

### 04 — VQ-VAE
- encoder → vector quantizer (codebook size=512, embed_dim=64) → decoder
- loss: reconstruction + commitment (beta=0.25) + codebook loss
- generation: random codebook sampling (no PixelCNN prior — document this limitation)
- reference: kamenbliznashki/generative_models

### 05 — DDPM
- HuggingFace diffusers UNet2DModel
- train from scratch on chest xray data
- DDPMScheduler, 1000 timesteps
- ⚠️ takes 4-6 hours on Kaggle T4 — save checkpoint every 10 epochs

### 06 — Flow Matching
- library: torchcfm (pip install torchcfm)
- small UNet backbone
- loss: MSE between predicted and target velocity field
- reference: atong01/conditional-flow-matching mnist_example.ipynb (adapt to 64x64)

---

## shared src/ modules (build after notebooks work)

```
src/evaluate.py   — compute_fid(real_path, fake_path) + compute_tstr(fake_path)
src/label.py      — label_images(image_folder) using torchxrayvision
src/export.py     — save_dataset(images, labels, metrics, output_dir)
src/models/       — clean class versions of each model
```

---

## comment style rules (from rules.md)
- explain WHY not WHAT
- subheadings over inline comments in notebooks
- no AI-sounding comments ("leveraging state-of-the-art...")
- no useless comments ("# import libraries")
- if a block needs more than 2 lines to explain → write a markdown cell

---

## key libraries
```
torch torchvision
diffusers transformers accelerate
torchcfm
torchxrayvision
pytorch-fid
gradio
```

---

## what "done" means for each notebook
- [ ] model trains without crash
- [ ] 100 images generated and saved as PNG
- [ ] FID score printed
- [ ] torchxrayvision scores on generated images printed
- [ ] labels.csv saved
- [ ] metrics.json saved

FID printed = notebook is presentable. everything else is bonus.

---

## Gradio app (app.py)
- 6 tabs, one per model
- inputs: class (normal/pneumonia) + n_samples (default 100)
- outputs: image grid + FID score + TSTR accuracy + download ZIP button
- deploy to HuggingFace Spaces (free CPU tier)

---

## image settings (consistent across everything)
- size: 64x64
- channels: 1 (grayscale)
- format: PNG (no JPEG compression)
- random seed: 42 everywhere
- batch size: 32 for training
