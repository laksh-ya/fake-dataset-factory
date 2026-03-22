# Rules

## evaluation order — non-negotiable
always run in this exact sequence per model:

```
1. FID first          → before anything else, always
2. labeling           → torchxrayvision scores
3. TSTR               → last, needs labeled outputs
```

why: FID gives you a presentable number even mid-project.
if you only have FID, you have something to show.
if you skip to TSTR first, you have nothing until the end.

---

## comment style

### do this
```python
# load real images as reference distribution for FID
real_images = load_folder("data/pneumonia/train")

# quantize continuous encoder output to nearest codebook vector
z_q = codebook[torch.argmin(distances, dim=1)]
```

### not this
```python
# This function uses advanced AI techniques to process the data
# leveraging state-of-the-art deep learning methodologies
x = model(x)
```

**rules:**
- subheadings over inline comments where possible
- explain WHY, not WHAT (the code shows what, you explain why)
- no "as an AI" style comments
- no corporate-speak, no buzzword salad
- if a block needs more than 2 lines to explain, write a markdown cell instead (notebooks)

---

## notebook structure (every notebook follows this)

```
## Setup
## Dataset Loading
## Model Architecture
## Training
## Generate 100 Images
## Evaluate — FID
## Evaluate — Label + TSTR
## Export
## Results
```

same structure every time. sir can open any notebook and instantly know where to look.

---

## what counts as done for a notebook
- [ ] model trains without crashing
- [ ] 100 images generated and saved
- [ ] FID score printed
- [ ] torchxrayvision scores on generated images
- [ ] labels.csv exported
- [ ] metrics.json exported

if FID is printed = notebook is presentable. rest is bonus.

---

## misc
- all images saved as PNG not JPEG (no compression artifacts)
- all models save checkpoints every 10 epochs (so overnight runs dont die silently)
- random seed = 42 everywhere (reproducibility)
- image size = 64x64 everywhere, no exceptions (fair comparison)
- generate exactly 100 images per run (FID valid at 100, GPU friendly)
