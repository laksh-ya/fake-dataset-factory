"""
Gradio app for synthetic chest X-ray generation.
Auto-detects available models and enables their tabs.
"""

import os
import json
import random
import shutil
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy import linalg

# conditionally import heavy libraries
try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False

try:
    from diffusers import StableDiffusionImg2ImgPipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "notebooks" / "outputs"

IMG_SIZE = 64
NOISE_DIM = 100
FEATURE_G = 64


# =============================================================================
# Model Definitions (same as notebooks)
# =============================================================================

class DCGANGenerator(nn.Module):
    """DCGAN Generator — noise to 64x64 grayscale image."""
    def __init__(self, noise_dim=100, channels=1, feature_g=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class WGANGPGenerator(nn.Module):
    """WGAN-GP Generator — same architecture as DCGAN."""
    def __init__(self, noise_dim=100, channels=1, feature_g=64):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim, feature_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 8, feature_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 4, feature_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g * 2, feature_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(feature_g, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# =============================================================================
# Model Registry
# =============================================================================

MODEL_CONFIGS = {
    "01_stable_diffusion": {
        "name": "Stable Diffusion (img2img)",
        "checkpoint": None,  # uses pretrained SD
        "check_path": OUTPUTS_DIR / "01_stable_diffusion" / "metrics.json",
        "type": "stable_diffusion"
    },
    "02_dcgan": {
        "name": "DCGAN",
        "checkpoint": OUTPUTS_DIR / "02_dcgan" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "dcgan"
    },
    "03_wgan_gp": {
        "name": "WGAN-GP",
        "checkpoint": OUTPUTS_DIR / "03_wgan_gp" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "wgan_gp"
    },
    "04_vqvae": {
        "name": "VQ-VAE",
        "checkpoint": OUTPUTS_DIR / "04_vqvae" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "vqvae"
    },
    "05_ddpm": {
        "name": "DDPM",
        "checkpoint": OUTPUTS_DIR / "05_ddpm" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "ddpm"
    },
    "06_flow_matching": {
        "name": "Flow Matching",
        "checkpoint": OUTPUTS_DIR / "06_flow_matching" / "checkpoints" / "checkpoint_epoch_50.pt",
        "type": "flow_matching"
    }
}

# cache loaded models
loaded_models = {}


def check_model_available(model_id: str) -> bool:
    """Check if model checkpoint exists."""
    config = MODEL_CONFIGS[model_id]

    if config["type"] == "stable_diffusion":
        check_path = config.get("check_path")
        return check_path and check_path.exists()

    checkpoint = config.get("checkpoint")
    return checkpoint and checkpoint.exists()


def get_available_models() -> dict:
    """Return dict of model_id -> availability status."""
    return {mid: check_model_available(mid) for mid in MODEL_CONFIGS}


def load_model(model_id: str):
    """Load model from checkpoint. Returns None if unavailable."""
    if model_id in loaded_models:
        return loaded_models[model_id]

    if not check_model_available(model_id):
        return None

    config = MODEL_CONFIGS[model_id]
    model_type = config["type"]

    if model_type == "stable_diffusion":
        if not DIFFUSERS_AVAILABLE:
            return None
        # SD pipeline loaded on demand during generation
        loaded_models[model_id] = "stable_diffusion"
        return "stable_diffusion"

    elif model_type == "dcgan":
        model = DCGANGenerator(NOISE_DIM, 1, FEATURE_G)
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["netG_state_dict"])
        model.to(device)
        model.eval()
        loaded_models[model_id] = model
        return model

    elif model_type == "wgan_gp":
        model = WGANGPGenerator(NOISE_DIM, 1, FEATURE_G)
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["netG_state_dict"])
        model.to(device)
        model.eval()
        loaded_models[model_id] = model
        return model

    # TODO: add VQ-VAE, DDPM, Flow Matching loaders when notebooks are ready

    return None


# =============================================================================
# Evaluation Functions
# =============================================================================

def load_xrv_model():
    """Load torchxrayvision model for FID and labeling."""
    if not XRV_AVAILABLE:
        return None, None

    xrv_model = xrv.models.DenseNet(weights="densenet121-res224-all")
    xrv_model.to(device)
    xrv_model.eval()

    feature_extractor = nn.Sequential(*list(xrv_model.features.children()))
    feature_extractor.to(device)
    feature_extractor.eval()

    return xrv_model, feature_extractor


def preprocess_for_xrv(img: Image.Image) -> torch.Tensor:
    """Preprocess PIL image for torchxrayvision."""
    img = img.convert('L').resize((224, 224), Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = (arr / 255.0) * 2048 - 1024
    return torch.tensor(arr[np.newaxis, ...], dtype=torch.float32)


def compute_fid(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Compute FID between two feature sets."""
    # need at least 2 samples for valid covariance
    if len(real_features) < 2 or len(fake_features) < 2:
        return None

    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    # add small epsilon to diagonal for numerical stability
    eps = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_fake += np.eye(sigma_fake.shape[0]) * eps

    diff = mu_real - mu_fake

    try:
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        # check for NaN/Inf
        if not np.isfinite(covmean).all():
            return None
        fid = float(diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean))
        return fid if np.isfinite(fid) else None
    except Exception:
        return None


# =============================================================================
# Generation Functions
# =============================================================================

def generate_gan_images(model, n_samples: int) -> list:
    """Generate images from GAN model."""
    images = []
    with torch.no_grad():
        for i in range(n_samples):
            noise = torch.randn(1, NOISE_DIM, 1, 1, device=device)
            fake = model(noise)
            arr = fake.squeeze().cpu().numpy()
            arr = ((arr + 1) / 2 * 255).astype(np.uint8)
            images.append(Image.fromarray(arr, mode='L'))
    return images


def generate_sd_images(n_samples: int, target_class: str) -> list:
    """Generate images using Stable Diffusion img2img."""
    if not DIFFUSERS_AVAILABLE:
        return []

    # load real images as input
    class_dir = DATA_DIR / target_class.lower()
    if not class_dir.exists():
        return []

    real_paths = sorted([p for p in class_dir.iterdir() if p.suffix == '.png'])[:n_samples]
    if not real_paths:
        return []

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None,
        requires_safety_checker=False
    )
    pipe = pipe.to(device)
    if torch.cuda.is_available():
        pipe.enable_attention_slicing()

    images = []
    prompt = "chest x-ray, medical radiograph, grayscale"
    neg_prompt = "color, artistic, painting, drawing, cartoon"

    for i, img_path in enumerate(real_paths):
        img = Image.open(img_path).convert('RGB').resize((512, 512), Image.LANCZOS)
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=neg_prompt,
                image=img,
                strength=0.7,
                guidance_scale=7.5,
                num_inference_steps=50,
                generator=torch.Generator(device=device).manual_seed(SEED + i)
            )
        gen_img = result.images[0].convert('L').resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        images.append(gen_img)

    # cleanup
    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return images


def generate_images(model_id: str, target_class: str, n_samples: int):
    """Main generation function. Returns (images, fid, tstr, zip_path)."""
    model = load_model(model_id)
    if model is None:
        return None, None, None, None

    config = MODEL_CONFIGS[model_id]

    # generate images
    if config["type"] == "stable_diffusion":
        images = generate_sd_images(n_samples, target_class)
    elif config["type"] in ["dcgan", "wgan_gp"]:
        images = generate_gan_images(model, n_samples)
    else:
        return None, None, None, None

    if not images:
        return None, None, None, None

    # compute FID and TSTR if torchxrayvision available
    fid_score = None
    tstr_accuracy = None

    if XRV_AVAILABLE:
        xrv_model, feature_extractor = load_xrv_model()

        # extract features from generated images
        fake_features = []
        pneumonia_scores = []

        pathology_names = xrv_model.pathologies
        pneumonia_idx = next(
            (i for i, name in enumerate(pathology_names) if 'lung opacity' in name.lower()),
            None
        )

        for img in images:
            inp = preprocess_for_xrv(img).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = feature_extractor(inp).mean(dim=[2, 3])
                fake_features.append(feat.cpu().numpy())

                pred = xrv_model(inp)
                if pneumonia_idx is not None:
                    pneumonia_scores.append(pred[0, pneumonia_idx].item())

        fake_features = np.concatenate(fake_features, axis=0)

        # load real images for FID comparison
        class_dir = DATA_DIR / target_class.lower()
        if class_dir.exists():
            real_paths = sorted([p for p in class_dir.iterdir() if p.suffix == '.png'])
            real_sample = random.sample(real_paths, min(n_samples, len(real_paths)))

            real_features = []
            for p in real_sample:
                img = Image.open(p)
                inp = preprocess_for_xrv(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feat = feature_extractor(inp).mean(dim=[2, 3])
                    real_features.append(feat.cpu().numpy())

            if real_features:
                real_features = np.concatenate(real_features, axis=0)
                fid_score = compute_fid(real_features, fake_features)

        # TSTR accuracy
        if pneumonia_scores:
            threshold = 0.5
            if target_class.lower() == "pneumonia":
                tstr_accuracy = sum(s > threshold for s in pneumonia_scores) / len(pneumonia_scores) * 100
            else:
                tstr_accuracy = sum(s <= threshold for s in pneumonia_scores) / len(pneumonia_scores) * 100

    # create ZIP file
    tmp_dir = tempfile.mkdtemp()
    zip_dir = Path(tmp_dir) / "generated"
    zip_dir.mkdir()

    for i, img in enumerate(images):
        img.save(zip_dir / f"{i:04d}.png", "PNG")

    zip_path = Path(tmp_dir) / f"{model_id}_{target_class}_generated.zip"
    shutil.make_archive(str(zip_path.with_suffix('')), 'zip', zip_dir)

    return images, fid_score, tstr_accuracy, str(zip_path)


# =============================================================================
# Gradio Interface
# =============================================================================

def create_model_tab(model_id: str, is_available: bool):
    """Create Gradio components for a model tab."""
    config = MODEL_CONFIGS[model_id]

    if not is_available:
        return gr.Markdown(f"""
        ## {config['name']}

        **Status:** Training in progress...

        This model will be available once training completes.
        Restart the app to check for newly trained models.
        """)

    with gr.Column():
        gr.Markdown(f"## {config['name']}")

        with gr.Row():
            class_dropdown = gr.Dropdown(
                choices=["Normal", "Pneumonia"],
                value="Pneumonia",
                label="Target Class"
            )
            n_samples = gr.Slider(
                minimum=1,
                maximum=100,
                value=100,
                step=1,
                label="Number of Samples"
            )

        generate_btn = gr.Button("Generate Images", variant="primary")

        with gr.Row():
            fid_display = gr.Textbox(label="Domain-adapted FID", interactive=False)
            tstr_display = gr.Textbox(label="Proxy TSTR Accuracy", interactive=False)

        gallery = gr.Gallery(
            label="Generated Images",
            columns=5,
            rows=4,
            height="auto"
        )

        download_btn = gr.File(label="Download ZIP")

        def on_generate(target_class, n):
            images, fid, tstr, zip_path = generate_images(model_id, target_class, int(n))

            if images is None:
                return [], "Error", "Error", None

            fid_str = f"{fid:.2f}" if fid is not None else "N/A (need more samples)"
            tstr_str = f"{tstr:.1f}%" if tstr is not None else "N/A"

            return images, fid_str, tstr_str, zip_path

        generate_btn.click(
            fn=on_generate,
            inputs=[class_dropdown, n_samples],
            outputs=[gallery, fid_display, tstr_display, download_btn]
        )


def create_app():
    """Create the Gradio app."""
    available = get_available_models()

    with gr.Blocks(title="Fake Dataset Factory", theme=gr.themes.Soft()) as app:
        gr.Markdown("""
        # Fake Dataset Factory

        Synthetic medical chest X-ray generation using 6 different generative models.

        **Evaluation:** Domain-adapted FID (DenseNet121 features) + Proxy TSTR (torchxrayvision)
        """)

        with gr.Row():
            status_md = gr.Markdown(f"""
            **Available Models:** {sum(available.values())}/6 |
            **Device:** {device} |
            **torchxrayvision:** {'Yes' if XRV_AVAILABLE else 'No'}
            """)
            refresh_btn = gr.Button("Refresh Models", size="sm")

        with gr.Tabs():
            for model_id, config in MODEL_CONFIGS.items():
                with gr.Tab(config["name"]):
                    create_model_tab(model_id, available[model_id])

        gr.Markdown("""
        ---
        **Note:** Click 'Refresh Models' after training new models, then restart the app to enable them.
        """)

        def on_refresh():
            new_available = get_available_models()
            count = sum(new_available.values())
            status = ", ".join([
                f"{MODEL_CONFIGS[mid]['name']}: {'Ready' if avail else 'Training'}"
                for mid, avail in new_available.items()
            ])
            return f"**Models ({count}/6 ready):** {status}\n\n*Restart app to load newly available models.*"

        refresh_btn.click(fn=on_refresh, outputs=[status_md])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(share=False)
