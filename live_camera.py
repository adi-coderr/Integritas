"""
Desktop Real-Time Camera Inspector using OpenCV & UNet-VGG16.

Runs directly as a native Mac window without a browser:
    python3 live_camera.py

Hotkeys:
    [q] or [ESC] : Quit
    [s]          : Save snapshot to disk (original + prediction + mask)
    [SPACE]      : Freeze / Unfreeze current frame
    [c]          : Toggle CLAHE lighting auto-contrast
    [m]          : Cycle display modes (Overlay -> Side-by-Side -> Heatmap -> Mask)
    [+] / [-]    : Adjust sensitivity / threshold
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
from pathlib import Path

# Auto-switch to .venv if needed
try:
    import cv2
    import numpy as np
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

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
INPUT_SIZE = 448
FONT = cv2.FONT_HERSHEY_SIMPLEX


def apply_clahe(bgr_img: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)


def main():
    parser = argparse.ArgumentParser(description="UNet-VGG16 Real-Time Camera Inspector")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--weights", type=str, default="model_unet_vgg_16_best.pt", help="Path to model weights")
    parser.add_argument("--threshold", type=float, default=0.45, help="Detection threshold")
    parser.add_argument("--clahe", action="store_true", help="Enable CLAHE contrast enhancement")
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    args = parser.parse_args()

    device = get_default_device()
    print("=" * 65)
    print("  UNet-VGG16 Desktop Camera Inspector")
    print("=" * 65)
    print(f"Loading weights from: {args.weights}")
    model, meta = load_unet16_model(args.weights, device=device)
    print(f"Device: {device} | Validation IoU: {meta.get('valid_iou', 0.6544)*100:.2f}%")

    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"\n[ERROR] Could not open camera {args.camera_id}.")
        print("Please verify camera permissions in macOS System Settings:")
        print("  System Settings > Privacy & Security > Camera > Terminal (Allow)")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    modes = ["overlay", "side_by_side", "heatmap", "mask"]
    mode_idx = 0
    threshold = args.threshold
    use_clahe = args.clahe
    frozen = False
    last_display = None
    last_mask = None
    last_frame = None

    snapshots_dir = Path("snapshots")
    snapshots_dir.mkdir(exist_ok=True)

    print("\nControls:")
    print("  [q] / [ESC] : Quit")
    print("  [s]         : Save snapshot to 'snapshots/'")
    print("  [SPACE]     : Freeze / Unfreeze frame")
    print("  [c]         : Toggle CLAHE contrast enhancement")
    print("  [m]         : Cycle view mode (Overlay -> Split -> Heatmap -> Mask)")
    print("  [+] / [-]   : Adjust detection threshold")
    print("-" * 65)

    fps_history = []
    last_time = time.perf_counter()

    while True:
        if not frozen:
            ret, frame = cap.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            last_frame = frame.copy()
            h, w = frame.shape[:2]

            # Contrast enhancement
            proc_bgr = apply_clahe(frame) if use_clahe else frame

            # Prepare tensor for 448x448 inference
            rgb = cv2.cvtColor(proc_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
            arr = (resized.astype(np.float32) / 255.0 - _MEAN) / _STD
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().to(device)

            t0 = time.perf_counter()
            with torch.no_grad():
                logits = model(tensor)
                prob_448 = torch.sigmoid(logits)[0, 0].cpu().numpy()

            infer_ms = (time.perf_counter() - t0) * 1000.0

            # Resize probability back to original frame
            prob_map = cv2.resize(prob_448, (w, h), interpolation=cv2.INTER_LINEAR)

            # Threshold & clean mask
            raw_mask = (prob_map >= threshold).astype(np.uint8)
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
            clean_mask = np.zeros_like(raw_mask)
            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] >= 25:
                    clean_mask[labels == i] = 255

            last_mask = clean_mask
            crack_pixels = int((clean_mask > 0).sum())
            crack_pct = (crack_pixels / clean_mask.size) * 100.0
            max_prob = float(prob_map.max())
            has_crack = crack_pixels >= 40 and max_prob >= max(0.45, threshold)

            # Render overlay
            overlay = frame.copy()
            if crack_pixels > 0:
                colored = np.zeros_like(frame)
                colored[:] = (20, 20, 255)  # Red BGR
                idx = clean_mask > 0
                overlay[idx] = cv2.addWeighted(frame[idx], 0.35, colored[idx], 0.65, 0)
                contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(overlay, contours, -1, (0, 240, 120), 2, cv2.LINE_AA)

            # View modes
            current_mode = modes[mode_idx]
            if current_mode == "side_by_side":
                display = np.hstack([frame, overlay])
            elif current_mode == "heatmap":
                prob_u8 = (np.clip(prob_map, 0.0, 1.0) * 255).astype(np.uint8)
                display = cv2.applyColorMap(prob_u8, cv2.COLORMAP_JET)
            elif current_mode == "mask":
                display = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
            else:  # overlay
                display = overlay

            # FPS calculation
            t_now = time.perf_counter()
            fps_history.append(1.0 / max(1e-5, t_now - last_time))
            last_time = t_now
            if len(fps_history) > 10:
                fps_history.pop(0)
            avg_fps = sum(fps_history) / len(fps_history)

            # Render HUD
            disp_h, disp_w = display.shape[:2]
            # Top HUD bar
            cv2.rectangle(display, (0, 0), (disp_w, 40), (15, 18, 24), -1)
            hud_text = f"FPS: {avg_fps:.1f} | Latency: {infer_ms:.0f}ms | Dev: {device} | Thresh: {threshold:.2f} | Mode: {current_mode.upper()}"
            cv2.putText(display, hud_text, (12, 26), FONT, 0.52, (200, 220, 240), 1, cv2.LINE_AA)

            # Bottom verdict banner
            banner_color = (25, 25, 200) if has_crack else (25, 140, 35)
            verdict_text = f"CRACK DETECTED: {crack_pct:.2f}% (Conf: {max_prob*100:.1f}%)" if has_crack else "SURFACE SOUND / CLEAN"
            cv2.rectangle(display, (0, disp_h - 36), (disp_w, disp_h), banner_color, -1)
            cv2.putText(display, verdict_text, (12, disp_h - 12), FONT, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

            last_display = display
        else:
            # Frozen banner
            display = last_display.copy()
            cv2.putText(display, "[FROZEN]", (display.shape[1] - 120, 26), FONT, 0.6, (0, 220, 255), 2, cv2.LINE_AA)

        cv2.imshow("UNet-VGG16 Crack Inspector", display)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
        elif key == 32:  # SPACE
            frozen = not frozen
            print(f"[{'FROZEN' if frozen else 'RESUMED'}] Frame")
        elif key in (ord('+'), ord('=')):
            threshold = min(0.95, round(threshold + 0.05, 2))
            print(f"Threshold increased to: {threshold:.2f}")
        elif key in (ord('-'), ord('_')):
            threshold = max(0.05, round(threshold - 0.05, 2))
            print(f"Threshold decreased to: {threshold:.2f}")
        elif key == ord('c'):
            use_clahe = not use_clahe
            print(f"CLAHE contrast enhancement: {'ON' if use_clahe else 'OFF'}")
        elif key == ord('m'):
            mode_idx = (mode_idx + 1) % len(modes)
            print(f"Display mode switched to: {modes[mode_idx]}")
        elif key == ord('s'):
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            snap_path = snapshots_dir / f"snap_{ts}_overlay.jpg"
            mask_path = snapshots_dir / f"snap_{ts}_mask.png"
            cv2.imwrite(str(snap_path), last_display)
            if last_mask is not None:
                cv2.imwrite(str(mask_path), last_mask)
            print(f"Snapshot saved: {snap_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Camera released. Exited cleanly.")


if __name__ == "__main__":
    main()
