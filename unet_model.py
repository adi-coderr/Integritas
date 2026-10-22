"""
UNet-VGG16 Model Architecture for Crack Segmentation.

This module provides a standalone, clean PyTorch implementation of the UNet-VGG16
convolutional neural network architecture for concrete/surface crack segmentation.
It does not require external model downloads or internet access.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

# Auto-environment switcher: re-exec into local .venv if torch is missing
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    _here = Path(__file__).resolve().parent
    _venv_py = _here / ".venv" / "bin" / "python3"
    if not _venv_py.is_file():
        _venv_py = _here.parent / ".venv" / "bin" / "python3"
    if _venv_py.is_file() and sys.executable != str(_venv_py):
        print(f"[Auto-Environment] Switching to project virtualenv: {_venv_py}")
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
    raise


class Interpolate(nn.Module):
    """Bilinear interpolation layer for upsampling feature maps."""

    def __init__(
        self,
        size: Optional[Tuple[int, int]] = None,
        scale_factor: Optional[float] = None,
        mode: str = "bilinear",
        align_corners: bool = False,
    ):
        super().__init__()
        self.size = size
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=self.size,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )


def conv3x3(in_channels: int, out_channels: int) -> nn.Conv2d:
    """3x3 convolution with padding=1 to preserve spatial dimensions."""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)


class ConvRelu(nn.Module):
    """A Conv3x3 followed by ReLU activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = conv3x3(in_channels, out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.conv(x))


class DecoderBlockV2(nn.Module):
    """UNet decoder block using bilinear upsampling + 2 ConvRelu layers."""

    def __init__(
        self,
        in_channels: int,
        middle_channels: int,
        out_channels: int,
        is_deconv: bool = False,
    ):
        super().__init__()
        if is_deconv:
            self.block = nn.Sequential(
                ConvRelu(in_channels, middle_channels),
                nn.ConvTranspose2d(
                    middle_channels,
                    out_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.ReLU(inplace=True),
            )
        else:
            self.block = nn.Sequential(
                Interpolate(scale_factor=2, mode="bilinear", align_corners=False),
                ConvRelu(in_channels, middle_channels),
                ConvRelu(middle_channels, out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet16(nn.Module):
    """
    UNet architecture with VGG16-style encoder backbone.

    Parameters:
    - num_classes: Number of output channels (default: 1 for binary crack mask)
    - num_filters: Base number of filters in decoder (default: 32)
    - is_deconv: Whether to use ConvTranspose2d instead of Bilinear interpolation
    """

    def __init__(
        self,
        num_classes: int = 1,
        num_filters: int = 32,
        is_deconv: bool = False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU(inplace=True)

        # Encoder stages (matching VGG16 features structure)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            self.relu,
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            self.relu,
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            self.relu,
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.relu,
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.relu,
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            self.relu,
        )

        # Bottleneck & Decoder stages
        self.center = DecoderBlockV2(512, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec5 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec4 = DecoderBlockV2(512 + num_filters * 8, num_filters * 8 * 2, num_filters * 8, is_deconv)
        self.dec3 = DecoderBlockV2(256 + num_filters * 8, num_filters * 4 * 2, num_filters * 2, is_deconv)
        self.dec2 = DecoderBlockV2(128 + num_filters * 2, num_filters * 2 * 2, num_filters, is_deconv)
        self.dec1 = ConvRelu(64 + num_filters, num_filters)
        self.final = nn.Conv2d(num_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input: Tensor of shape (B, 3, H, W)
        Output: Raw logits tensor of shape (B, 1, H, W)
        """
        conv1 = self.conv1(x)
        conv2 = self.conv2(self.pool(conv1))
        conv3 = self.conv3(self.pool(conv2))
        conv4 = self.conv4(self.pool(conv3))
        conv5 = self.conv5(self.pool(conv4))

        center = self.center(self.pool(conv5))

        dec5 = self.dec5(torch.cat([center, conv5], dim=1))
        dec4 = self.dec4(torch.cat([dec5, conv4], dim=1))
        dec3 = self.dec3(torch.cat([dec4, conv3], dim=1))
        dec2 = self.dec2(torch.cat([dec3, conv2], dim=1))
        dec1 = self.dec1(torch.cat([dec2, conv1], dim=1))

        if self.num_classes > 1:
            return F.log_softmax(self.final(dec1), dim=1)
        return self.final(dec1)


def get_default_device() -> torch.device:
    """
    Selects the fastest hardware accelerator available:
    1. CUDA (NVIDIA GPU)
    2. Apple Silicon MPS (Metal Performance Shaders on macOS)
    3. CPU fallback
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_unet16_model(
    weights_path: Union[str, Path] = "model_unet_vgg_16_best.pt",
    device: Optional[Union[str, torch.device]] = None,
) -> Tuple[UNet16, Dict[str, Any]]:
    """
    Loads and initializes the UNet16 model from a trained checkpoint.

    Returns:
        (model, metadata_dict)
    """
    if device is None:
        device = get_default_device()
    elif isinstance(device, str):
        device = torch.device(device)

    weights_file = Path(weights_path)
    if not weights_file.is_file():
        # Check current working directory or relative path
        alt = Path(__file__).resolve().parent / weights_file.name
        if alt.is_file():
            weights_file = alt
        else:
            raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")

    # Load checkpoint safely on CPU first
    try:
        checkpoint = torch.load(weights_file, map_location="cpu", weights_only=True)
    except Exception:
        checkpoint = torch.load(weights_file, map_location="cpu", weights_only=False)

    metadata: Dict[str, Any] = {
        "weights_path": str(weights_file),
        "file_size_mb": round(weights_file.stat().st_size / (1024 * 1024), 2),
        "device": str(device),
    }

    if isinstance(checkpoint, dict):
        if "epoch" in checkpoint:
            metadata["epoch"] = checkpoint["epoch"]
        if "valid_iou" in checkpoint:
            metadata["valid_iou"] = checkpoint["valid_iou"]
        if "valid_loss" in checkpoint:
            metadata["valid_loss"] = checkpoint["valid_loss"]
        if "train_loss" in checkpoint:
            metadata["train_loss"] = checkpoint["train_loss"]

        raw_state = None
        for key in ("model", "state_dict", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                raw_state = checkpoint[key]
                break
        if raw_state is None:
            raw_state = checkpoint
    else:
        raw_state = checkpoint

    # Clean keys: remove DataParallel 'module.' prefix and ignore 'encoder.' aliases
    clean_state = {}
    for k, v in raw_state.items():
        k_clean = k[7:] if k.startswith("module.") else k
        if not k_clean.startswith("encoder."):
            clean_state[k_clean] = v

    model = UNet16(num_classes=1, num_filters=32, is_deconv=False)
    load_res = model.load_state_dict(clean_state, strict=True)

    metadata["missing_keys"] = load_res.missing_keys
    metadata["unexpected_keys"] = load_res.unexpected_keys
    metadata["total_params"] = sum(p.numel() for p in model.parameters())

    model = model.to(device)
    model.eval()

    return model, metadata


if __name__ == "__main__":
    print("Testing UNet16 model instantiation and weight loading...")
    dev = get_default_device()
    print(f"Optimal compute device: {dev}")

    model, meta = load_unet16_model("model_unet_vgg_16_best.pt", device=dev)
    print("\nModel successfully loaded:")
    for k, v in meta.items():
        print(f"  - {k}: {v}")

    # Forward pass test
    dummy = torch.randn(1, 3, 448, 448, device=dev)
    with torch.no_grad():
        out = model(dummy)
        prob = torch.sigmoid(out)

    print(f"\nDummy forward pass successful:")
    print(f"  - Input shape:  {list(dummy.shape)}")
    print(f"  - Output shape: {list(out.shape)}")
    print(f"  - Prob min/max: {prob.min().item():.4f} / {prob.max().item():.4f}")
