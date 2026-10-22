"""
Quickstart Single-Image Test.

Run directly with:
    python3 quick_test.py

Demonstrates how to load the model and predict on a sample image in just a few lines.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Auto-switch to .venv if needed
try:
    import cv2
    import numpy as np
    import torch
except ImportError:
    _here = Path(__file__).resolve().parent
    _venv_py = _here / ".venv" / "bin" / "python3"
    if _venv_py.is_file() and sys.executable != str(_venv_py):
        import os
        os.execv(str(_venv_py), [str(_venv_py)] + sys.argv)
    raise

from predict import CrackPredictor

def run_quick_demo():
    print("=" * 60)
    print("  UNet-VGG16 Crack Segmentation - Quick Demonstration")
    print("=" * 60)

    # 1. Initialize predictor
    print("\n1. Loading model checkpoint...")
    predictor = CrackPredictor("model_unet_vgg_16_best.pt")
    print(f"   Model active on: {predictor.device}")

    # 2. Run inference on sample image
    test_img = Path("sample_images/pure_crack_01.jpg")
    print(f"\n2. Running inference on '{test_img}'...")
    result = predictor.predict(test_img, mode="tiled", threshold=0.5)

    # 3. Print inspection results
    print("\n3. Detection Results:")
    print(f"   - Crack Detected:      {result['crack_detected']}")
    print(f"   - Peak Confidence:     {result['peak_probability'] * 100:.2f}%")
    print(f"   - Mean Confidence:     {result['mean_probability'] * 100:.3f}%")
    print(f"   - Affected Surface:    {result['crack_percentage']:.2f}% ({result['crack_pixels']:,} px)")
    print(f"   - Distinct Instances:  {result['num_crack_instances']}")
    print(f"   - Inference Time:      {result['inference_time_ms']} ms")

    # 4. Save visual output
    out_file = Path("quick_test_output.jpg")
    panel = predictor.render_diagnostic_panel(result)
    cv2.imwrite(str(out_file), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    print(f"\n4. Diagnostic 3-panel image saved to: {out_file.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    run_quick_demo()
