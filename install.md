# Install

## Kaggle setup (do this first)
```bash
# in any kaggle notebook, dataset is already available at:
/kaggle/input/chest-xray-pneumonia/

# or download locally:
kaggle datasets download paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

---

## pip installs
```bash
pip install torch torchvision
pip install diffusers transformers accelerate   # for SD and DDPM
pip install torchcfm                            # for flow matching
pip install torchxrayvision                     # for evaluation
pip install pytorch-fid                         # for FID score
pip install gradio                              # for UI
pip install kaggle                              # for dataset download
```

one liner:
```bash
pip install torch torchvision diffusers transformers accelerate torchcfm torchxrayvision pytorch-fid gradio
```

---

## Kaggle T4 GPU tips
- enable GPU: settings → accelerator → GPU T4 x2
- 30 free hours per week, resets weekly
- save checkpoints every 10 epochs so a disconnect doesnt kill your run
- DDPM training: start it, set persistence on, close laptop, check in morning

---

## HuggingFace Spaces deploy
```bash
# create new Space at huggingface.co/spaces
# choose Gradio SDK
# upload app.py + requirements.txt
# requirements.txt:
torch
torchvision
diffusers
torchxrayvision
gradio
```

free CPU tier is enough for inference demo.

---

## verify everything works
```python
import torch
import torchxrayvision as xrv
import diffusers
import torchcfm

print("torch:", torch.__version__)
print("torchxrayvision:", xrv.__version__)
print("diffusers:", diffusers.__version__)
print("GPU:", torch.cuda.is_available())
```
