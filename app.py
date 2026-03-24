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
import torch.nn.functional as F
from PIL import Image
from scipy import linalg

# conditionally import heavy libraries
try:
    import torchxrayvision as xrv
    XRV_AVAILABLE = True
except ImportError:
    XRV_AVAILABLE = False

try:
    from diffusers import StableDiffusionImg2ImgPipeline, UNet2DModel, DDPMScheduler
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False

try:
    from torchdiffeq import odeint
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False


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
# VQ-VAE Model Definition
# =============================================================================

class VectorQuantizer(nn.Module):
    """Quantizes continuous latents to discrete codebook vectors."""
    def __init__(self, num_embeddings, embed_dim, beta=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embed_dim = embed_dim
        self.beta = beta
        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z):
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flat = z.view(-1, self.embed_dim)
        d = (z_flat ** 2).sum(dim=1, keepdim=True) + \
            (self.embedding.weight ** 2).sum(dim=1) - \
            2 * z_flat @ self.embedding.weight.t()
        indices = d.argmin(dim=1)
        z_q = self.embedding(indices).view(z.shape)
        codebook_loss = F.mse_loss(z_q, z.detach())
        commitment_loss = F.mse_loss(z_q.detach(), z)
        z_q = z + (z_q - z).detach()
        z_q = z_q.permute(0, 3, 1, 2).contiguous()
        return z_q, codebook_loss, commitment_loss, indices


class VQVAEEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, embed_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, hidden_dim, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, embed_dim, 1, 1, 0)
        )

    def forward(self, x):
        return self.net(x)


class VQVAEDecoder(nn.Module):
    def __init__(self, embed_dim, hidden_dim, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, 2, 1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim // 2, out_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


class VQVAE(nn.Module):
    def __init__(self, in_channels=1, hidden_dim=128, embed_dim=64, num_embeddings=512, beta=0.25):
        super().__init__()
        self.encoder = VQVAEEncoder(in_channels, hidden_dim, embed_dim)
        self.vq = VectorQuantizer(num_embeddings, embed_dim, beta)
        self.decoder = VQVAEDecoder(embed_dim, hidden_dim, in_channels)

    def forward(self, x):
        z = self.encoder(x)
        z_q, codebook_loss, commitment_loss, indices = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon, codebook_loss, commitment_loss, indices

    def decode(self, z_q):
        return self.decoder(z_q)


# =============================================================================
# Flow Matching Model Definition
# =============================================================================

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)
        self.block1 = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.residual_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.block1(x)
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.block2(h)
        return h + self.residual_conv(x)


class FlowMatchingUNet(nn.Module):
    def __init__(self, in_ch=1, base_ch=64, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        self.enc1 = ResBlock(in_ch, base_ch, time_emb_dim)
        self.enc2 = ResBlock(base_ch, base_ch * 2, time_emb_dim)
        self.enc3 = ResBlock(base_ch * 2, base_ch * 4, time_emb_dim)
        self.down1 = nn.Conv2d(base_ch, base_ch, 4, 2, 1)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 2, 4, 2, 1)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, 4, 2, 1)
        self.mid = ResBlock(base_ch * 4, base_ch * 4, time_emb_dim)
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, 4, 2, 1)
        self.dec3 = ResBlock(base_ch * 8, base_ch * 2, time_emb_dim)
        self.up2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, 4, 2, 1)
        self.dec2 = ResBlock(base_ch * 4, base_ch, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(base_ch, base_ch, 4, 2, 1)
        self.dec1 = ResBlock(base_ch * 2, base_ch, time_emb_dim)
        self.out = nn.Conv2d(base_ch, in_ch, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down1(e1), t_emb)
        e3 = self.enc3(self.down2(e2), t_emb)
        m = self.mid(self.down3(e3), t_emb)
        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1), t_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1), t_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1), t_emb)
        return self.out(d1)


class ODEFunc(nn.Module):
    """Wrapper for ODE solver."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x):
        t_batch = t.expand(x.shape[0])
        return self.model(x, t_batch)


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

    elif model_type == "vqvae":
        model = VQVAE(in_channels=1, hidden_dim=128, embed_dim=64, num_embeddings=512, beta=0.25)
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        loaded_models[model_id] = model
        return model

    elif model_type == "ddpm":
        if not DIFFUSERS_AVAILABLE:
            return None
        model = UNet2DModel(
            sample_size=IMG_SIZE,
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=(64, 128, 256, 256),
            down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
            up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        )
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        scheduler = DDPMScheduler(num_train_timesteps=1000)
        loaded_models[model_id] = {"model": model, "scheduler": scheduler}
        return loaded_models[model_id]

    elif model_type == "flow_matching":
        if not TORCHDIFFEQ_AVAILABLE:
            return None
        model = FlowMatchingUNet(in_ch=1, base_ch=64, time_emb_dim=128)
        checkpoint = torch.load(config["checkpoint"], map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        loaded_models[model_id] = model
        return model

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


def generate_vqvae_images(model, n_samples: int) -> list:
    """Generate images from VQ-VAE using random codebook sampling."""
    images = []
    latent_h, latent_w = 16, 16  # 64/4
    embed_dim = 64
    num_embeddings = 512

    with torch.no_grad():
        for i in range(n_samples):
            indices = torch.randint(0, num_embeddings, (latent_h * latent_w,), device=device)
            z_q = model.vq.embedding(indices)
            z_q = z_q.view(1, latent_h, latent_w, embed_dim)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            fake = model.decode(z_q)
            arr = fake.squeeze().cpu().numpy()
            arr = (arr * 255).astype(np.uint8)
            images.append(Image.fromarray(arr, mode='L'))
    return images


def generate_ddpm_images(model_dict, n_samples: int) -> list:
    """Generate images from DDPM."""
    model = model_dict["model"]
    scheduler = model_dict["scheduler"]
    images = []

    with torch.no_grad():
        for i in range(n_samples):
            torch.manual_seed(SEED + i)
            sample = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=device)

            for t in scheduler.timesteps:
                model_output = model(sample, t).sample
                sample = scheduler.step(model_output, t, sample).prev_sample

            arr = sample.squeeze().cpu().numpy()
            arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(arr, mode='L'))
    return images


def generate_flow_matching_images(model, n_samples: int) -> list:
    """Generate images from Flow Matching using ODE integration."""
    if not TORCHDIFFEQ_AVAILABLE:
        return []

    images = []
    ode_func = ODEFunc(model)

    with torch.no_grad():
        for i in range(n_samples):
            torch.manual_seed(SEED + i)
            x0 = torch.randn(1, 1, IMG_SIZE, IMG_SIZE, device=device)
            t_span = torch.tensor([0.0, 1.0], device=device)
            x1 = odeint(ode_func, x0, t_span, method='euler', options={'step_size': 0.1})[-1]
            arr = x1.squeeze().cpu().numpy()
            arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
            images.append(Image.fromarray(arr, mode='L'))
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
    elif config["type"] == "vqvae":
        images = generate_vqvae_images(model, n_samples)
    elif config["type"] == "ddpm":
        images = generate_ddpm_images(model, n_samples)
    elif config["type"] == "flow_matching":
        images = generate_flow_matching_images(model, n_samples)
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
