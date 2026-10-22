"""
Flask Web Application for UNet-VGG16 Crack Inspection Studio.

Supports:
- High-resolution image analysis (drag-and-drop & pre-loaded samples)
- Live interactive HTML5 camera streaming directly in Safari / Chrome
- Real-time crack segmentation overlay, heatmaps, and metrics HUD
- Interactive Before / After split-view comparison

Run with:
    python3 web_app.py
Then open:
    http://localhost:5050 in your browser.
"""

from __future__ import annotations

import base64
import io
import os
import sys
import time
from pathlib import Path

# Auto-switch to .venv if needed
try:
    import cv2
    from flask import Flask, jsonify, render_template, request
    import numpy as np
    from PIL import Image
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

from predict import CrackPredictor, INPUT_SIZE

app = Flask(__name__, template_folder="templates")
predictor: CrackPredictor | None = None
SAMPLE_DIR = Path(__file__).resolve().parent / "sample_images"
_frame_count = 0


def get_predictor() -> CrackPredictor:
    global predictor
    if predictor is None:
        print("[App] Initializing UNet-VGG16 predictor...")
        predictor = CrackPredictor("model_unet_vgg_16_best.pt")
        print(f"[App] Model ready on device: {predictor.device}")
    return predictor


def apply_clahe_contrast(img_bgr: np.ndarray) -> np.ndarray:
    """Enhance crack contrast for low-light or uneven webcam frames."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def numpy_to_base64_jpeg(img_rgb: np.ndarray, quality: int = 85) -> str:
    """Encodes RGB numpy array to base64 JPEG data URI."""
    im = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def mask_to_base64_png(mask_uint8: np.ndarray) -> str:
    """Encodes binary mask to base64 PNG data URI."""
    im = Image.fromarray(mask_uint8)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/info", methods=["GET"])
def get_info():
    pred = get_predictor()
    samples = []
    if SAMPLE_DIR.is_dir():
        for p in sorted(SAMPLE_DIR.glob("*.jpg")):
            lbl = p.stem.replace("_", " ").title()
            cat = "Clear Crack" if "pure" in p.stem else ("Moss/Rough" if "moss" in p.stem else ("Distractor" if "like" in p.stem else "Sound Surface"))
            samples.append({
                "name": p.name,
                "label": lbl,
                "type": cat
            })

    return jsonify({
        "status": "ready",
        "device": str(pred.device),
        "epoch": pred.metadata.get("epoch", 4),
        "valid_iou": pred.metadata.get("valid_iou", 0.6544),
        "samples": samples,
    })


@app.route("/api/predict_frame", methods=["POST"])
def predict_frame():
    """Ultra-low latency endpoint for live webcam frame streaming."""
    t0 = time.perf_counter()
    data = request.get_json(force=True)
    raw_data = data.get("image", "")
    threshold = float(data.get("threshold", 0.45))
    use_clahe = bool(data.get("use_clahe", False))
    view_mode = data.get("view_mode", "overlay")  # overlay, side_by_side, heatmap

    if "," in raw_data:
        raw_data = raw_data.split(",", 1)[1]

    img_bytes = base64.b64decode(raw_data)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame_bgr is None:
        return jsonify({"success": False, "error": "Invalid frame data"}), 400

    if use_clahe:
        frame_bgr = apply_clahe_contrast(frame_bgr)

    orig_h, orig_w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    pred = get_predictor()
    # Fast 448x448 inference
    prob_map = pred.predict_full(frame_rgb)

    infer_ms = (time.perf_counter() - t0) * 1000.0

    # Binary mask
    raw_mask = (prob_map >= threshold).astype(np.uint8)

    # Filter small noise specks
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    clean_mask = np.zeros_like(raw_mask)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= 25:
            clean_mask[labels == i] = 255

    crack_pixels = int((clean_mask > 0).sum())
    crack_pct = round((crack_pixels / clean_mask.size) * 100.0, 2)
    max_prob = round(float(prob_map.max()), 4)
    has_crack = crack_pixels >= 40 and max_prob >= max(0.45, threshold)

    global _frame_count
    _frame_count += 1
    if _frame_count % 10 == 1 or has_crack:
        verdict_str = f"⚠️ CRACK DETECTED ({crack_pct}%)" if has_crack else "CLEAN"
        print(f"[ML Model Stream] Frame #{_frame_count:<5} | {infer_ms:5.1f}ms | Conf: {max_prob*100:5.1f}% | Verdict: {verdict_str}")

    # Generate visual output according to view_mode
    if view_mode == "heatmap":
        prob_u8 = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
        rendered_bgr = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
    elif view_mode == "side_by_side":
        overlay = frame_bgr.copy()
        if crack_pixels > 0:
            colored = np.zeros_like(frame_bgr)
            colored[:] = (20, 20, 255)  # BGR Red
            idx = clean_mask > 0
            overlay[idx] = cv2.addWeighted(frame_bgr[idx], 0.35, colored[idx], 0.65, 0)
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 240, 120), 2, cv2.LINE_AA)
        rendered_bgr = np.hstack([frame_bgr, overlay])
    else:  # overlay
        overlay = frame_bgr.copy()
        if crack_pixels > 0:
            colored = np.zeros_like(frame_bgr)
            colored[:] = (20, 20, 255)  # BGR Red
            idx = clean_mask > 0
            overlay[idx] = cv2.addWeighted(frame_bgr[idx], 0.35, colored[idx], 0.65, 0)
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 240, 120), 2, cv2.LINE_AA)
        rendered_bgr = overlay

    # Compress JPEG for high throughput
    _, buf = cv2.imencode(".jpg", rendered_bgr, [cv2.IMWRITE_JPEG_QUALITY, 75])
    b64_output = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "success": True,
        "image": b64_output,
        "crack_detected": has_crack,
        "crack_percentage": crack_pct,
        "peak_probability": max_prob,
        "inference_time_ms": round(infer_ms, 1),
    })


@app.route("/api/deep_scan", methods=["POST"])
def run_deep_scan():
    """
    High-precision deep scan endpoint on captured photo or uploaded image.
    Applies multi-tile overlapping inference + TTA to pinpoint exact crack locations.
    """
    data = request.get_json(force=True)
    raw_data = data.get("image", "")
    threshold = float(data.get("threshold", 0.45))
    use_clahe = bool(data.get("use_clahe", False))

    if isinstance(raw_data, dict) and raw_data.get("type") == "sample":
        sample_name = raw_data.get("name")
        img_path = SAMPLE_DIR / sample_name
        if not img_path.is_file():
            return jsonify({"error": f"Sample not found: {sample_name}"}), 404
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
    elif isinstance(raw_data, str):
        if "," in raw_data:
            raw_data = raw_data.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_data)
        nparr = np.frombuffer(img_bytes, np.uint8)
        frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return jsonify({"error": "Invalid image payload"}), 400
        if use_clahe:
            frame_bgr = apply_clahe_contrast(frame_bgr)
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    else:
        return jsonify({"error": "No valid image provided"}), 400

    pred = get_predictor()
    scan_res = pred.deep_scan(img_rgb, threshold=threshold, use_tta=True)

    # Render visualizations
    annotated_rgb = pred.render_pinpoint_overlay(img_rgb, scan_res)
    overlay_rgb = pred.render_overlay(img_rgb, scan_res["binary_mask"])
    heatmap_rgb = pred.render_heatmap(scan_res["prob_map"])
    panel_rgb = pred.render_diagnostic_panel(scan_res)

    print(f"\n[Deep Scan] Image ({img_rgb.shape[1]}x{img_rgb.shape[0]}) -> "
          f"Found {scan_res['num_crack_instances']} cracks | "
          f"{scan_res['crack_percentage']}% damaged | "
          f"Time: {scan_res['inference_time_ms']}ms")

    return jsonify({
        "success": True,
        "crack_detected": bool(scan_res["crack_detected"]),
        "crack_percentage": scan_res["crack_percentage"],
        "crack_pixels": scan_res["crack_pixels"],
        "total_pixels": scan_res["total_pixels"],
        "peak_probability": scan_res["peak_probability"],
        "num_crack_instances": scan_res["num_crack_instances"],
        "cracks": scan_res["cracks"],
        "inference_time_ms": scan_res["inference_time_ms"],
        "threshold": threshold,
        "images": {
            "original": numpy_to_base64_jpeg(img_rgb),
            "annotated": numpy_to_base64_jpeg(annotated_rgb),
            "overlay": numpy_to_base64_jpeg(overlay_rgb),
            "heatmap": numpy_to_base64_jpeg(heatmap_rgb),
            "comparison": numpy_to_base64_jpeg(panel_rgb),
            "mask": mask_to_base64_png(scan_res["binary_mask"]),
        }
    })


@app.route("/api/predict", methods=["POST"])
def run_predict():
    data = request.get_json(force=True)
    mode = data.get("mode", "tiled")
    threshold = float(data.get("threshold", 0.50))
    img_spec = data.get("image", {})

    if img_spec.get("type") == "sample":
        sample_name = img_spec.get("name")
        img_path = SAMPLE_DIR / sample_name
        if not img_path.is_file():
            return jsonify({"error": f"Sample not found: {sample_name}"}), 404
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
    elif img_spec.get("type") == "data":
        data_uri = img_spec.get("data", "")
        if "," in data_uri:
            data_uri = data_uri.split(",", 1)[1]
        raw_bytes = base64.b64decode(data_uri)
        img_rgb = np.array(Image.open(io.BytesIO(raw_bytes)).convert("RGB"))
    else:
        return jsonify({"error": "No valid image payload provided"}), 400

    pred = get_predictor()
    res = pred.predict(img_rgb, mode=mode, threshold=threshold)

    # Render visualizations
    overlay_rgb = pred.render_overlay(img_rgb, res["binary_mask"])
    heatmap_rgb = pred.render_heatmap(res["prob_map"])
    panel_rgb = pred.render_diagnostic_panel(res)

    return jsonify({
        "crack_detected": bool(res["crack_detected"]),
        "crack_percentage": res["crack_percentage"],
        "crack_pixels": res["crack_pixels"],
        "total_pixels": res["total_pixels"],
        "peak_probability": res["peak_probability"],
        "mean_probability": res["mean_probability"],
        "inference_time_ms": res["inference_time_ms"],
        "mode": res["mode"],
        "threshold": res["threshold"],
        "images": {
            "original": numpy_to_base64_jpeg(img_rgb),
            "overlay": numpy_to_base64_jpeg(overlay_rgb),
            "heatmap": numpy_to_base64_jpeg(heatmap_rgb),
            "comparison": numpy_to_base64_jpeg(panel_rgb),
            "mask": mask_to_base64_png(res["binary_mask"]),
        }
    })


def find_open_port(starting_port: int = 5050, max_attempts: int = 20) -> int:
    import socket
    for p in range(starting_port, starting_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", p)) != 0:
                return p
    return starting_port


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UNet Crack Inspection Studio Web App")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5050)), help="Port to run web app on")
    args = parser.parse_args()

    port = find_open_port(args.port)
    print(f"\n🚀 UNet Crack Inspection Studio starting on http://localhost:{port}")
    print(f"👉 Open in your browser: http://localhost:{port}\n")
    app.run(host="127.0.0.1", port=port, debug=False)
