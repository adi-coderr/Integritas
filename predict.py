"""
Crack Segmentation Prediction Engine & CLI.

Provides high-accuracy inference using the UNet-VGG16 model.
Supports:
- Single image or batch folder prediction
- 'full' mode (fast 448x448 resize)
- 'tiled' mode (sliding window patch-based inference for high-res images,
  preserving thin hairline cracks without downscale blurring)
- Rich visual outputs: glowing overlay, probability heatmap, binary mask,
  and multi-panel diagnostic comparison panels
- Comprehensive crack statistics: severity metrics, area %, blob counts,
  bounding boxes, and inference time
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Auto-environment switcher: re-exec into local .venv if dependencies are missing
try:
    import cv2
    import numpy as np
    from PIL import Image, ImageOps
    import torch
except ImportError:
    _here = Path(__file__).resolve().parent
    _venv_py = _here / ".venv" / "bin" / "python3"
    if not _venv_py.is_file():
        _venv_py = _here.parent / ".venv" / "bin" / "python3"
    if _venv_py.is_file() and sys.executable != str(_venv_py):
        print(f"[Auto-Environment] Switching to project virtualenv: {_venv_py}")
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
    raise

from unet_model import get_default_device, load_unet16_model

# ImageNet normalization parameters
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 448
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_rgb_image(image_input: Union[str, Path, np.ndarray, Image.Image]) -> np.ndarray:
    """Loads image as RGB uint8 numpy array, respecting EXIF orientation."""
    if isinstance(image_input, np.ndarray):
        if image_input.ndim == 2:
            return cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        if image_input.shape[2] == 4:
            return cv2.cvtColor(image_input, cv2.COLOR_RGBA2RGB)
        return image_input.copy()

    if isinstance(image_input, Image.Image):
        im = ImageOps.exif_transpose(image_input)
        return np.array(im.convert("RGB"))

    path = Path(image_input)
    if not path.is_file():
        raise FileNotFoundError(f"Image not found at: {path}")

    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.array(im.convert("RGB"))


class CrackPredictor:
    """Inference engine for UNet-VGG16 crack detection."""

    def __init__(
        self,
        weights_path: Union[str, Path] = "model_unet_vgg_16_best.pt",
        device: Optional[Union[str, torch.device]] = None,
    ):
        self.device = get_default_device() if device is None else (
            torch.device(device) if isinstance(device, str) else device
        )
        self.model, self.metadata = load_unet16_model(weights_path, device=self.device)

    @torch.no_grad()
    def _predict_batch_patches(self, patches: List[np.ndarray]) -> np.ndarray:
        """
        Runs batch prediction on a list of 448x448 RGB uint8 patches.
        Returns: numpy array of shape (N, 448, 448) with probabilities in [0.0, 1.0].
        """
        if not patches:
            return np.empty((0, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)

        x = np.stack(patches).astype(np.float32) / 255.0
        x = (x - _MEAN) / _STD
        tensor = torch.from_numpy(x).permute(0, 3, 1, 2).contiguous().to(self.device)
        logits = self.model(tensor)
        probs = torch.sigmoid(logits)[:, 0]
        return probs.cpu().numpy()

    def predict_full(self, img_rgb: np.ndarray) -> np.ndarray:
        """
        Fast full-image inference.
        Resizes entire image to 448x448, runs model, and scales probability map
        back to the original image dimensions.
        """
        orig_h, orig_w = img_rgb.shape[:2]
        resized = cv2.resize(img_rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_AREA)
        prob_patch = self._predict_batch_patches([resized])[0]
        prob_full = cv2.resize(prob_patch, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        return np.clip(prob_full, 0.0, 1.0)

    def predict_tiled(
        self,
        img_rgb: np.ndarray,
        work_size: int = 1344,
        overlap: float = 0.25,
        batch_size: int = 4,
    ) -> np.ndarray:
        """
        High-fidelity tiled inference for large photos.
        Slices image into overlapping 448x448 patches, predicts each, and averages
        overlapping regions. Preserves fine hairline cracks without downscale loss.
        """
        orig_h, orig_w = img_rgb.shape[:2]

        # Scale so longest side is at most work_size (never upscale)
        scale = work_size / max(orig_h, orig_w)
        if scale < 1.0:
            target_w = max(1, round(orig_w * scale))
            target_h = max(1, round(orig_h * scale))
            work_img = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
        else:
            work_img = img_rgb.copy()

        h, w = work_img.shape[:2]

        # Pad if smaller than 448 in either dimension
        pad_h = max(0, INPUT_SIZE - h)
        pad_w = max(0, INPUT_SIZE - w)
        if pad_h or pad_w:
            work_img = cv2.copyMakeBorder(
                work_img, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101
            )
        big_h, big_w = work_img.shape[:2]

        # Compute tile coordinates
        stride = max(1, int(INPUT_SIZE * (1.0 - overlap)))
        ys = list(range(0, big_h - INPUT_SIZE + 1, stride))
        xs = list(range(0, big_w - INPUT_SIZE + 1, stride))
        if ys[-1] != big_h - INPUT_SIZE:
            ys.append(big_h - INPUT_SIZE)
        if xs[-1] != big_w - INPUT_SIZE:
            xs.append(big_w - INPUT_SIZE)
        coords = [(y, x) for y in ys for x in xs]

        acc_probs = np.zeros((big_h, big_w), dtype=np.float32)
        weight_acc = np.zeros((big_h, big_w), dtype=np.float32)

        # Batch inference
        for i in range(0, len(coords), batch_size):
            chunk = coords[i : i + batch_size]
            patches = [work_img[y : y + INPUT_SIZE, x : x + INPUT_SIZE] for y, x in chunk]
            batch_probs = self._predict_batch_patches(patches)
            for (y, x), patch_prob in zip(chunk, batch_probs):
                acc_probs[y : y + INPUT_SIZE, x : x + INPUT_SIZE] += patch_prob
                weight_acc[y : y + INPUT_SIZE, x : x + INPUT_SIZE] += 1.0

        # Average overlaps
        weight_acc = np.maximum(weight_acc, 1e-6)
        prob_work = acc_probs / weight_acc

        # Crop padding
        prob_work = prob_work[:h, :w]

        # Resize back to original image size
        if (h, w) != (orig_h, orig_w):
            prob_final = cv2.resize(prob_work, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
        else:
            prob_final = prob_work

        return np.clip(prob_final, 0.0, 1.0)

    def predict(
        self,
        image_input: Union[str, Path, np.ndarray, Image.Image],
        mode: str = "tiled",
        threshold: float = 0.5,
        min_blob_area: int = 30,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end crack segmentation and returns mask, overlays, and metrics.
        """
        t0 = time.perf_counter()
        img_rgb = load_rgb_image(image_input)
        h, w = img_rgb.shape[:2]

        if mode == "full":
            prob_map = self.predict_full(img_rgb)
        else:
            prob_map = self.predict_tiled(img_rgb)

        inference_time_ms = (time.perf_counter() - t0) * 1000

        # Thresholding
        raw_mask = (prob_map >= threshold).astype(np.uint8)

        # Morphological noise removal & connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            raw_mask, connectivity=8
        )

        clean_mask = np.zeros_like(raw_mask)
        bounding_boxes = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area >= min_blob_area:
                clean_mask[labels == i] = 255
                bx = int(stats[i, cv2.CC_STAT_LEFT])
                by = int(stats[i, cv2.CC_STAT_TOP])
                bw = int(stats[i, cv2.CC_STAT_WIDTH])
                bh = int(stats[i, cv2.CC_STAT_HEIGHT])
                bounding_boxes.append({"x": bx, "y": by, "w": bw, "h": bh, "area": int(area)})

        crack_pixels = int((clean_mask > 0).sum())
        total_pixels = clean_mask.size
        crack_percentage = (crack_pixels / total_pixels) * 100.0
        peak_prob = float(prob_map.max())
        mean_prob = float(prob_map.mean())

        # Verdict
        crack_detected = crack_pixels >= min_blob_area and peak_prob >= max(0.5, threshold)

        return {
            "crack_detected": crack_detected,
            "crack_percentage": round(crack_percentage, 3),
            "crack_pixels": crack_pixels,
            "total_pixels": total_pixels,
            "peak_probability": round(peak_prob, 4),
            "mean_probability": round(mean_prob, 5),
            "num_crack_instances": len(bounding_boxes),
            "bounding_boxes": bounding_boxes,
            "inference_time_ms": round(inference_time_ms, 1),
            "image_size": (w, h),
            "mode": mode,
            "threshold": threshold,
            "prob_map": prob_map,
            "binary_mask": clean_mask,
            "original_rgb": img_rgb,
        }

    @staticmethod
    def render_overlay(
        img_rgb: np.ndarray,
        binary_mask: np.ndarray,
        color_rgb: Tuple[int, int, int] = (255, 30, 30),
        alpha: float = 0.55,
    ) -> np.ndarray:
        """Renders semi-transparent colored crack overlay on the original image."""
        overlay = img_rgb.copy()
        mask_bool = binary_mask > 0
        if np.any(mask_bool):
            colored_mask = np.zeros_like(img_rgb)
            colored_mask[:] = color_rgb
            overlay[mask_bool] = (
                (1.0 - alpha) * img_rgb[mask_bool] + alpha * colored_mask[mask_bool]
            ).astype(np.uint8)

            # Draw crisp contour borders
            contours, _ = cv2.findContours(
                binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(overlay, contours, -1, (255, 230, 0), thickness=1, lineType=cv2.LINE_AA)

        return overlay

    @staticmethod
    def render_heatmap(prob_map: np.ndarray) -> np.ndarray:
        """Converts float probability map into Jet colormap."""
        scaled = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
        heatmap_bgr = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
        return cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    @classmethod
    def render_diagnostic_panel(
        cls,
        result: Dict[str, Any],
        panel_height: int = 480,
    ) -> np.ndarray:
        """
        Creates a high-resolution 3-panel comparison image:
        [ Original Photo ] | [ Probability Heatmap ] | [ Crack Overlay + HUD ]
        """
        img_rgb = result["original_rgb"]
        prob_map = result["prob_map"]
        mask = result["binary_mask"]

        h, w = img_rgb.shape[:2]
        scale = panel_height / h
        new_w = max(1, round(w * scale))

        # 1. Original
        p1 = cv2.resize(img_rgb, (new_w, panel_height), interpolation=cv2.INTER_AREA)

        # 2. Heatmap
        heatmap = cls.render_heatmap(prob_map)
        p2 = cv2.resize(heatmap, (new_w, panel_height), interpolation=cv2.INTER_AREA)

        # 3. Crack Overlay
        overlay = cls.render_overlay(img_rgb, mask)
        p3 = cv2.resize(overlay, (new_w, panel_height), interpolation=cv2.INTER_AREA)

        # Header badges
        font = cv2.FONT_HERSHEY_SIMPLEX
        for p, title in ((p1, "Original Input"), (p2, "Probability Heatmap"), (p3, "Crack Detection")):
            cv2.rectangle(p, (0, 0), (new_w, 36), (20, 20, 20), -1)
            cv2.putText(p, title, (12, 24), font, 0.65, (255, 255, 255), 1, cv2.LINE_AA)

        # Verdict HUD banner on p3
        detected = result["crack_detected"]
        banner_color = (180, 20, 20) if detected else (30, 140, 40)
        verdict_str = f"CRACK DETECTED ({result['crack_percentage']}%)" if detected else "NO SIGNIFICANT CRACK"
        cv2.rectangle(p3, (0, panel_height - 38), (new_w, panel_height), banner_color, -1)
        cv2.putText(p3, verdict_str, (12, panel_height - 14), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(p3, f"{result['inference_time_ms']}ms", (new_w - 95, panel_height - 14), font, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Concatenate panels side by side with separator borders
        sep = np.full((panel_height, 4, 3), 40, dtype=np.uint8)
        composite = np.hstack([p1, sep, p2, sep, p3])
        return composite


def main():
    parser = argparse.ArgumentParser(description="UNet-VGG16 Crack Segmentation CLI")
    parser.add_argument(
        "--input",
        type=str,
        default="sample_images",
        help="Path to an image file or a directory containing images",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="model_unet_vgg_16_best.pt",
        help="Path to trained model weights checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="predictions",
        help="Directory to save output masks and visualizations",
    )
    parser.add_argument(
        "--mode",
        choices=["tiled", "full"],
        default="tiled",
        help="Inference mode: 'tiled' (high-res sliding window) or 'full' (fast 448x448)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for crack classification (0.0 to 1.0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Compute device: 'mps', 'cuda', 'cpu', or None for auto-detect",
    )
    parser.add_argument(
        "--save-panel",
        action="store_true",
        default=True,
        help="Save 3-panel comparison image (Original | Heatmap | Overlay)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input path does not exist: {input_path}")
        sys.exit(1)

    if input_path.is_file():
        image_files = [input_path]
    else:
        image_files = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
        )

    if not image_files:
        print(f"[ERROR] No valid images found at: {input_path}")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading UNet-VGG16 model...")
    predictor = CrackPredictor(weights_path=args.weights, device=args.device)
    print(f"Device: {predictor.device} | Images to process: {len(image_files)} | Mode: {args.mode}")
    print("-" * 75)

    summary = []
    for idx, img_file in enumerate(image_files, 1):
        print(f"[{idx}/{len(image_files)}] Processing: {img_file.name} ... ", end="", flush=True)
        res = predictor.predict(
            img_file,
            mode=args.mode,
            threshold=args.threshold,
        )

        stem = img_file.stem
        # Save binary mask
        mask_path = output_dir / f"{stem}_mask.png"
        cv2.imwrite(str(mask_path), res["binary_mask"])

        # Save diagnostic 3-panel visualization
        if args.save_panel:
            panel = predictor.render_diagnostic_panel(res)
            panel_bgr = cv2.cvtColor(panel, cv2.COLOR_RGB2BGR)
            panel_path = output_dir / f"{stem}_comparison.jpg"
            cv2.imwrite(str(panel_path), panel_bgr)

        # Save overlay image
        overlay = predictor.render_overlay(res["original_rgb"], res["binary_mask"])
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
        overlay_path = output_dir / f"{stem}_overlay.jpg"
        cv2.imwrite(str(overlay_path), overlay_bgr)

        verdict_str = "CRACK DETECTED" if res["crack_detected"] else "CLEAN"
        print(f"{verdict_str} ({res['crack_percentage']}% coverage, {res['inference_time_ms']}ms)")

        summary.append({
            "filename": img_file.name,
            "verdict": verdict_str,
            "crack_detected": res["crack_detected"],
            "crack_percentage": res["crack_percentage"],
            "peak_probability": res["peak_probability"],
            "crack_instances": res["num_crack_instances"],
            "inference_time_ms": res["inference_time_ms"],
            "mask_file": mask_path.name,
            "overlay_file": overlay_path.name,
        })

    # Save summary report JSON
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 75)
    print(f"Done! Results and visualizations saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
