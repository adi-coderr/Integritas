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

import threading

try:
    cv2.setNumThreads(1)
except Exception:
    pass

import json
import math
import urllib.request
from typing import Optional

try:
    import serial
    import serial.tools.list_ports as serial_ports
except ImportError:
    serial = None
    serial_ports = None

from predict import CrackPredictor, INPUT_SIZE

app = Flask(__name__, template_folder="templates")
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # Support up to 50MB high-res inspection uploads
predictors: dict[str, CrackPredictor] = {}
SAMPLE_DIR = Path(__file__).resolve().parent / "sample_images"
_frame_count = 0
_inference_lock = threading.Lock()


class MPU6050Manager:
    """
    Thread-safe ingestion manager for MPU-6050 6-Axis IMU sensor telemetry.
    Supports:
      - Direct USB Serial port streaming (115200 baud JSON lines)
      - Wi-Fi HTTP polling (from ESP32-CAM http://<ip>/telemetry)
      - Browser Web Serial client feeds via /api/mpu/feed
      - Realistic physics-based simulation with dynamic vibration & tilt
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.connected = False
        self.connection_type = "disconnected"  # "serial", "wifi", "simulation", "client"
        self.connection_target = ""
        self.stop_event = threading.Event()
        self.worker_thread: Optional[threading.Thread] = None
        self.serial_obj = None

        # Calibration Tare offsets
        self.tare_pitch = 0.0
        self.tare_roll = 0.0

        # Latest raw & calibrated telemetry
        now = time.time()
        self.latest = {
            "pitch": 0.0,
            "roll": 0.0,
            "raw_pitch": 0.0,
            "raw_roll": 0.0,
            "ax": 0.0,
            "ay": 0.0,
            "az": 1.0,
            "gx": 0.0,
            "gy": 0.0,
            "gz": 0.0,
            "temp": 25.0,
            "shock": False,
            "total_g": 1.0,
            "timestamp": now,
        }
        self.history: list[dict] = []  # sliding window of last 60 points for charts
        self.history_max = 60
        self.last_update = now
        self.shock_count = 0
        self.last_shock_time = None
        self.packets_received = 0

    def update_reading(self, data: dict, source: Optional[str] = None):
        with self.lock:
            try:
                raw_pitch = float(data.get("pitch", self.latest.get("raw_pitch", 0.0)))
                raw_roll = float(data.get("roll", self.latest.get("raw_roll", 0.0)))
                ax = float(data.get("ax", self.latest.get("ax", 0.0)))
                ay = float(data.get("ay", self.latest.get("ay", 0.0)))
                az = float(data.get("az", self.latest.get("az", 1.0)))
                gx = float(data.get("gx", self.latest.get("gx", 0.0)))
                gy = float(data.get("gy", self.latest.get("gy", 0.0)))
                gz = float(data.get("gz", self.latest.get("gz", 0.0)))
                temp = float(data.get("temp", self.latest.get("temp", 25.0)))
                total_g = math.sqrt(ax * ax + ay * ay + az * az)
                shock = bool(data.get("shock", False)) or (total_g > 1.85)

                now = time.time()
                # Apply tare offset
                cal_pitch = round(raw_pitch - self.tare_pitch, 2)
                cal_roll = round(raw_roll - self.tare_roll, 2)

                if shock and (not self.latest.get("shock", False) or (now - (self.last_shock_time or 0) > 1.2)):
                    self.shock_count += 1
                    self.last_shock_time = now

                self.latest = {
                    "pitch": cal_pitch,
                    "roll": cal_roll,
                    "raw_pitch": round(raw_pitch, 2),
                    "raw_roll": round(raw_roll, 2),
                    "ax": round(ax, 3),
                    "ay": round(ay, 3),
                    "az": round(az, 3),
                    "gx": round(gx, 2),
                    "gy": round(gy, 2),
                    "gz": round(gz, 2),
                    "temp": round(temp, 1),
                    "shock": shock,
                    "total_g": round(total_g, 3),
                    "timestamp": now,
                }
                self.packets_received += 1
                self.last_update = now

                if source:
                    self.connected = True
                    self.connection_type = source

                self.history.append({
                    "t": round(now, 2),
                    "p": cal_pitch,
                    "r": cal_roll,
                    "g": round(total_g, 3),
                    "s": 1 if shock else 0
                })
                if len(self.history) > self.history_max:
                    self.history.pop(0)
            except Exception as e:
                pass

    def tare(self):
        with self.lock:
            self.tare_pitch = self.latest.get("raw_pitch", 0.0)
            self.tare_roll = self.latest.get("raw_roll", 0.0)
            self.latest["pitch"] = 0.0
            self.latest["roll"] = 0.0

    def reset_tare(self):
        with self.lock:
            self.tare_pitch = 0.0
            self.tare_roll = 0.0
            self.latest["pitch"] = self.latest.get("raw_pitch", 0.0)
            self.latest["roll"] = self.latest.get("raw_roll", 0.0)

    def trigger_test_shock(self):
        with self.lock:
            self.latest["shock"] = True
            self.latest["ax"] += 1.85
            self.latest["az"] += 1.45
            self.latest["total_g"] = math.sqrt(
                self.latest["ax"] ** 2 + self.latest["ay"] ** 2 + self.latest["az"] ** 2
            )
            self.shock_count += 1
            self.last_shock_time = time.time()

    def connect_serial(self, port: str, baud: int = 115200) -> bool:
        self.disconnect()
        if not serial:
            raise RuntimeError("pyserial module is not installed.")
        self.stop_event.clear()
        self.connection_target = port
        self.connection_type = "serial"
        self.worker_thread = threading.Thread(target=self._serial_worker, args=(port, baud), daemon=True)
        self.worker_thread.start()
        return True

    def connect_wifi(self, url: str) -> bool:
        self.disconnect()
        self.stop_event.clear()
        if not url.startswith("http://") and not url.startswith("https://"):
            url = f"http://{url}"
        if not ("/sensor" in url or "/telemetry" in url):
            url = f"{url.rstrip('/')}/sensor"
        self.connection_target = url
        self.connection_type = "wifi"
        self.worker_thread = threading.Thread(target=self._wifi_worker, args=(url,), daemon=True)
        self.worker_thread.start()
        return True

    def start_simulation(self) -> bool:
        self.disconnect()
        self.stop_event.clear()
        self.connection_target = "Physics Simulator"
        self.connection_type = "simulation"
        self.worker_thread = threading.Thread(target=self._sim_worker, daemon=True)
        self.worker_thread.start()
        return True

    def disconnect(self):
        self.stop_event.set()
        if self.serial_obj:
            try:
                self.serial_obj.close()
            except Exception:
                pass
            self.serial_obj = None
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.6)
        self.worker_thread = None
        with self.lock:
            self.connected = False
            self.connection_type = "disconnected"
            self.connection_target = ""

    def _serial_worker(self, port: str, baud: int):
        try:
            ser = serial.Serial(port, baudrate=baud, timeout=1.0)
            self.serial_obj = ser
            with self.lock:
                self.connected = True
            while not self.stop_event.is_set():
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if line.startswith("{") and line.endswith("}"):
                    try:
                        data = json.loads(line)
                        self.update_reading(data, source="serial")
                    except json.JSONDecodeError:
                        pass
        except Exception as e:
            print(f"[MPU-6050] Serial worker connection ended: {e}")
        finally:
            with self.lock:
                self.connected = False

    def _wifi_worker(self, url: str):
        with self.lock:
            self.connected = True
        consecutive_fails = 0
        active_url = url
        while not self.stop_event.is_set():
            try:
                req = urllib.request.Request(active_url, headers={"User-Agent": "MPU6050Studio/1.0"})
                with urllib.request.urlopen(req, timeout=1.2) as resp:
                    if resp.status == 200:
                        payload = resp.read().decode("utf-8")
                        data = json.loads(payload)
                        self.update_reading(data, source="wifi")
                        consecutive_fails = 0
            except urllib.error.HTTPError as e:
                # If 404, toggle between /sensor and /telemetry
                if e.code == 404:
                    if "/sensor" in active_url:
                        active_url = active_url.replace("/sensor", "/telemetry")
                    elif "/telemetry" in active_url:
                        active_url = active_url.replace("/telemetry", "/sensor")
                consecutive_fails += 1
                if consecutive_fails > 10:
                    with self.lock:
                        self.connected = False
            except Exception:
                consecutive_fails += 1
                if consecutive_fails > 10:
                    with self.lock:
                        self.connected = False
            time.sleep(0.06)  # ~16 Hz polling

    def _sim_worker(self):
        with self.lock:
            self.connected = True
        t = 0.0
        while not self.stop_event.is_set():
            t += 0.05
            # Realistic slight structural sway / probe movement
            p = math.sin(t * 1.2) * 5.5 + math.cos(t * 0.4) * 2.0
            r = math.cos(t * 0.9) * 4.2 + math.sin(t * 0.3) * 1.5
            rad_p = math.radians(p)
            rad_r = math.radians(r)
            ax = -math.sin(rad_p) + (math.sin(t * 8.0) * 0.015)
            ay = math.sin(rad_r) * math.cos(rad_p) + (math.cos(t * 7.0) * 0.015)
            az = math.cos(rad_r) * math.cos(rad_p) + (math.sin(t * 9.0) * 0.01)
            gx = math.cos(t * 0.9) * 4.2 * 0.9 + (math.sin(t * 11.0) * 0.2)
            gy = -math.sin(t * 1.2) * 5.5 * 1.2 + (math.cos(t * 13.0) * 0.2)
            gz = math.sin(t * 0.5) * 1.1
            temp = 26.5 + math.sin(t * 0.05) * 0.8
            shock = (math.sin(t * 0.2) > 0.985)

            self.update_reading({
                "pitch": p,
                "roll": r,
                "ax": ax,
                "ay": ay,
                "az": az,
                "gx": gx,
                "gy": gy,
                "gz": gz,
                "temp": temp,
                "shock": shock
            }, source="simulation")
            time.sleep(0.05)

    @staticmethod
    def list_serial_ports() -> list[dict]:
        ports = []
        if serial_ports:
            try:
                for p in serial_ports.comports():
                    ports.append({
                        "port": p.device,
                        "description": p.description,
                        "hwid": p.hwid or ""
                    })
            except Exception:
                pass
        return ports

    def get_status(self) -> dict:
        with self.lock:
            now = time.time()
            stale = (now - self.last_update) > 2.5 if self.connected else True
            return {
                "success": True,
                "connected": self.connected and not stale,
                "source": self.connection_type,
                "target": self.connection_target,
                "latest": self.latest,
                "history": self.history[-40:],
                "tare": {"pitch": self.tare_pitch, "roll": self.tare_roll},
                "shock_count": self.shock_count,
                "last_shock_time": self.last_shock_time,
                "packets_received": self.packets_received,
                "is_stale": stale,
            }


mpu_manager = MPU6050Manager()


def resolve_device(engine: str = "fast") -> str:
    """Resolves optimal compute device based on engine selection and hardware availability."""
    if engine in ("gentle", "cpu"):
        return "cpu"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_predictor(device: Optional[str] = None) -> CrackPredictor:
    global predictors
    if device is None:
        device = resolve_device("fast")
    if device not in predictors:
        print(f"[App] Initializing UNet-VGG16 predictor on {device.upper()}...")
        predictors[device] = CrackPredictor("model_unet_vgg_16_best.pt", device=device)
        print(f"[App] Model ready on device: {device.upper()}")
    return predictors[device]


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
    opt_device = resolve_device("fast")
    pred = get_predictor(opt_device)
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
        "optimal_device": opt_device,
        "has_mps": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
        "has_cuda": bool(torch.cuda.is_available()),
        "epoch": pred.metadata.get("epoch", 4),
        "valid_iou": round(float(pred.metadata.get("valid_iou", 0.6544)), 4),
        "samples": samples,
        "local_ip": get_local_ip(),
        "push_url": f"http://{get_local_ip()}:5050/api/mpu/feed",
    })


@app.route("/api/predict_frame", methods=["POST"])
def predict_frame():
    """Ultra-low latency endpoint for live webcam frame streaming."""
    t0 = time.perf_counter()
    data = request.get_json(force=True)
    raw_data = data.get("image", "")
    threshold = float(data.get("threshold", 0.45))
    use_clahe = bool(data.get("use_clahe", False))
    view_mode = data.get("view_mode", "overlay")  # overlay, side_by_side, heatmap, mask
    engine = data.get("engine", "fast")

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

    device_name = resolve_device(engine)
    pred = get_predictor(device_name)

    # Non-blocking lock: if model is currently processing a frame, drop this frame immediately.
    # This prevents request queuing, GPU contention, and eliminates laptop freezing.
    if not _inference_lock.acquire(blocking=False):
        return jsonify({"success": False, "skipped": True}), 200

    try:
        prob_map = pred.predict_full(frame_rgb)
    finally:
        _inference_lock.release()

    infer_ms = (time.perf_counter() - t0) * 1000.0

    # Binary mask
    raw_mask = (prob_map >= threshold).astype(np.uint8)

    # Filter small noise specks and locate crack instances
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    clean_mask = np.zeros_like(raw_mask)
    detected_boxes = []

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 25:
            clean_mask[labels == i] = 255
            bx = int(stats[i, cv2.CC_STAT_LEFT])
            by = int(stats[i, cv2.CC_STAT_TOP])
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            detected_boxes.append((bx, by, bw, bh, area))

    # Sort largest cracks first
    detected_boxes.sort(key=lambda b: b[4], reverse=True)

    crack_pixels = int((clean_mask > 0).sum())
    crack_pct = round((crack_pixels / clean_mask.size) * 100.0, 2)
    max_prob = round(float(prob_map.max()), 4)
    has_crack = crack_pixels >= 35 and max_prob >= max(0.40, threshold)

    global _frame_count
    _frame_count += 1
    if _frame_count % 20 == 1 or has_crack:
        verdict_str = f"⚠️ CRACK DETECTED ({crack_pct}%)" if has_crack else "CLEAN"
        print(f"[ML Live Stream] Frame #{_frame_count:<5} | {infer_ms:5.1f}ms | Conf: {max_prob*100:5.1f}% | Verdict: {verdict_str}")

    # Generate visual output according to view_mode
    if view_mode == "heatmap":
        prob_u8 = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
        jet = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
        rendered_bgr = cv2.addWeighted(frame_bgr, 0.45, jet, 0.55, 0)
    elif view_mode == "mask":
        rendered_bgr = np.zeros_like(frame_bgr)
        rendered_bgr[clean_mask > 0] = (0, 240, 120)  # Bright neon green
    elif view_mode == "side_by_side":
        overlay = frame_bgr.copy()
        if crack_pixels > 0:
            colored = np.zeros_like(frame_bgr)
            colored[:] = (25, 25, 255)  # Vibrant red
            idx = clean_mask > 0
            overlay[idx] = cv2.addWeighted(frame_bgr[idx], 0.35, colored[idx], 0.65, 0)
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 240, 120), 2, cv2.LINE_AA)
            for b_idx, (bx, by, bw, bh, _) in enumerate(detected_boxes[:4], 1):
                cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), (0, 220, 255), 2, cv2.LINE_AA)
        rendered_bgr = np.hstack([frame_bgr, overlay])
    else:  # overlay
        overlay = frame_bgr.copy()
        if crack_pixels > 0:
            colored = np.zeros_like(frame_bgr)
            colored[:] = (25, 25, 255)  # Vibrant red
            idx = clean_mask > 0
            overlay[idx] = cv2.addWeighted(frame_bgr[idx], 0.35, colored[idx], 0.65, 0)
            contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, (0, 240, 120), 2, cv2.LINE_AA)
            font = cv2.FONT_HERSHEY_SIMPLEX
            for b_idx, (bx, by, bw, bh, area) in enumerate(detected_boxes[:4], 1):
                x0, y0 = max(0, bx - 3), max(0, by - 3)
                x1, y1 = min(orig_w - 1, bx + bw + 3), min(orig_h - 1, by + bh + 3)
                cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 220, 255), 2, cv2.LINE_AA)
                tag_txt = f"#{b_idx} Crack"
                (tw, th), _ = cv2.getTextSize(tag_txt, font, 0.42, 1)
                cv2.rectangle(overlay, (x0, max(0, y0 - th - 6)), (x0 + tw + 8, y0), (12, 18, 26), -1)
                cv2.putText(overlay, tag_txt, (x0 + 4, y0 - 3), font, 0.42, (0, 220, 255), 1, cv2.LINE_AA)
        rendered_bgr = overlay

    # Compress JPEG for high throughput
    _, buf = cv2.imencode(".jpg", rendered_bgr, [cv2.IMWRITE_JPEG_QUALITY, 78])
    b64_output = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

    return jsonify({
        "success": True,
        "image": b64_output,
        "crack_detected": has_crack,
        "crack_percentage": crack_pct,
        "peak_probability": max_prob,
        "num_crack_instances": len(detected_boxes) if has_crack else 0,
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
    use_tta = bool(data.get("use_tta", False))

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

    engine = data.get("engine", "fast")
    device_name = resolve_device(engine)
    pred = get_predictor(device_name)
    gentle_delay = 0.020 if engine in ("gentle", "cpu") else 0.005

    with _inference_lock:
        scan_res = pred.deep_scan(
            img_rgb,
            threshold=threshold,
            use_tta=use_tta,
            work_size=896,
            gentle_delay=gentle_delay,
        )
        pred.clear_cache()

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
    with _inference_lock:
        res = pred.predict(img_rgb, mode=mode, threshold=threshold)
        pred.clear_cache()

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


# =========================================================================
# MPU-6050 6-Axis IMU & Orientation Telemetry Endpoints
# =========================================================================

@app.route("/api/mpu/telemetry", methods=["GET"])
def get_mpu_telemetry():
    """Returns real-time MPU-6050 pitch, roll, 3-axis accel, gyro & vibration."""
    return jsonify(mpu_manager.get_status())


@app.route("/api/mpu/ports", methods=["GET"])
def get_mpu_ports():
    """Scans and lists active serial ports (USB-TTL, FTDI, CP2102, CH340)."""
    return jsonify({
        "success": True,
        "ports": mpu_manager.list_serial_ports()
    })


@app.route("/api/mpu/connect", methods=["POST"])
def connect_mpu():
    """Connect to MPU-6050 via USB Serial, Wi-Fi HTTP, or Simulation."""
    data = request.get_json(force=True)
    conn_type = data.get("type", "simulation")

    try:
        if conn_type == "serial":
            port = data.get("port")
            if not port:
                return jsonify({"success": False, "error": "No serial port specified"}), 400
            baud = int(data.get("baud", 115200))
            mpu_manager.connect_serial(port, baud=baud)
            return jsonify({"success": True, "message": f"Connected to {port} @ {baud} baud"})
        elif conn_type == "wifi":
            url = data.get("url")
            if not url:
                return jsonify({"success": False, "error": "No Wi-Fi endpoint URL specified"}), 400
            mpu_manager.connect_wifi(url)
            return jsonify({"success": True, "message": f"Polling Wi-Fi endpoint: {url}"})
        elif conn_type == "simulation":
            mpu_manager.start_simulation()
            return jsonify({"success": True, "message": "Simulation active"})
        else:
            return jsonify({"success": False, "error": f"Unknown connection type: {conn_type}"}), 400
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/mpu/disconnect", methods=["POST"])
def disconnect_mpu():
    """Disconnects active MPU-6050 telemetry thread."""
    mpu_manager.disconnect()
    return jsonify({"success": True, "message": "Disconnected"})


@app.route("/api/mpu/tare", methods=["POST"])
def tare_mpu():
    """Zeroes reference orientation angles to calibrate level horizon."""
    data = request.get_json(force=True) if request.data else {}
    action = data.get("action", "set")
    if action == "reset":
        mpu_manager.reset_tare()
        return jsonify({"success": True, "message": "Tare reset to raw angles"})
    else:
        mpu_manager.tare()
        return jsonify({"success": True, "message": "Zeroed current orientation"})


@app.route("/api/mpu/simulate_shock", methods=["POST"])
def simulate_shock():
    """Triggers an instantaneous high-G shock wave to test alarm & alerting."""
    mpu_manager.trigger_test_shock()
    return jsonify({"success": True, "message": "Test shock triggered"})


def get_local_ip() -> str:
    """Returns the local LAN IP address of this machine (e.g. 10.169.71.221)."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


@app.route("/api/mpu/feed", methods=["POST", "GET"])
def feed_mpu():
    """
    Receives direct telemetry push from ESP32 over Wi-Fi or client Web Serial.
    NO ESP32 IP is needed when ESP32 pushes data to this endpoint!
    Supports both JSON POST and URL query parameters.
    """
    if request.method == "POST":
        data = request.get_json(force=True, silent=True)
        if not data:
            data = request.form.to_dict()
    else:
        data = request.args.to_dict()

    if not data:
        return jsonify({"success": False, "error": "Empty payload"}), 400

    client_ip = request.headers.get("X-Forwarded-For", request.remote_addr or "unknown")
    if client_ip in ("127.0.0.1", "::1", "localhost"):
        source = "Client Web Serial"
    else:
        source = f"ESP32 ({client_ip})"

    mpu_manager.update_reading(data, source=source)
    return jsonify({
        "success": True,
        "source": source,
        "packets_received": mpu_manager.packets_received
    })


def find_open_port(starting_port: int = 5050, max_attempts: int = 20) -> int:
    import socket
    for p in range(starting_port, starting_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("0.0.0.0", p)) != 0:
                return p
    return starting_port


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="UNet Crack Inspection Studio Web App")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5050)), help="Port to run web app on")
    args = parser.parse_args()

    port = find_open_port(args.port)
    local_ip = get_local_ip()
    print(f"\n🚀 UNet Crack Inspection Studio starting:")
    print(f"👉 Local Browser : http://localhost:{port}")
    print(f"👉 Network / ESP32: http://{local_ip}:{port}")
    print(f"👉 ESP32 Push URL: http://{local_ip}:{port}/api/mpu/feed\n")
    print(f"💡 TIP: In your ESP32 code, set pushServerHost = \"{local_ip}\" to stream data without knowing ESP32's IP!\n")
    app.run(host="0.0.0.0", port=port, debug=False)
