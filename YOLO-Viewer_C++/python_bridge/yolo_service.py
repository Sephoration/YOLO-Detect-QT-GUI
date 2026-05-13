#!/usr/bin/env python3
"""
YOLO Inference Service
======================
Communication protocol: JSON lines over stdin/stdout.
Each request is a single JSON line, each response is a single JSON line.

Actions:
  - inference:       Run inference on a frame
  - analyze_model:   Get model metadata
  - shutdown:        Exit gracefully

Dependencies:
  pip install torch torchvision ultralytics opencv-python numpy
"""

import sys
import json
import base64
import traceback
import io
import os
import time
import warnings
import numpy as np
import cv2

os.environ["YOLO_VERBOSE"] = "False"
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
#  Global model cache
# ──────────────────────────────────────────────
_model_cache: dict = {}          # path -> (model, task, names)

def _load_model(model_path: str):
    """Load a YOLO model (cached)."""
    if model_path in _model_cache:
        return _model_cache[model_path]

    from ultralytics import YOLO
    model = YOLO(model_path)
    task = model.task      # detect, classify, pose, segment
    names = model.names if hasattr(model, 'names') else {}
    _model_cache[model_path] = (model, task, names)
    return model, task, names


# ──────────────────────────────────────────────
#  Decode frame from JSON
# ──────────────────────────────────────────────
def _decode_frame(obj: dict) -> np.ndarray:
    """Reconstruct OpenCV BGR image from base64-encoded JPEG."""
    data = base64.b64decode(obj["data"])
    arr = np.frombuffer(data, dtype=np.uint8)
    mat = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return mat


# ──────────────────────────────────────────────
#  Inference handler
# ──────────────────────────────────────────────
def _handle_inference(req: dict) -> dict:
    model_path = req.get("model_path", "")
    mode = req.get("mode", "detection")
    conf = float(req.get("conf_threshold", 0.5))
    iou  = float(req.get("iou_threshold", 0.45))

    frame = _decode_frame(req.get("frame", {}))
    if frame is None:
        return {"action": "inference_result", "result": {"success": False, "error": "Invalid frame"}}

    model, task, names = _load_model(model_path)
    H, W = frame.shape[:2]

    t0 = time.perf_counter()
    results = model(frame, conf=conf, iou=iou, verbose=False)
    dt = (time.perf_counter() - t0) * 1000  # ms

    r = results[0] if results else None
    if r is None:
        return {
            "action": "inference_result",
            "result": {
                "success": True,
                "data_type": mode,
                "processed_data": {},
                "stats": {"detection_count": 0, "avg_confidence": 0, "inference_time": dt, "fps": 1000/dt}
            }
        }

    resp = {
        "success": True,
        "data_type": mode,
        "processed_data": {},
        "stats": {
            "detection_count": 0,
            "avg_confidence": 0.0,
            "inference_time": dt,
            "fps": 1000.0 / dt if dt > 0 else 0,
            "keypoint_count": 0,
            "num_people": 0,
        }
    }

    # ── Detection ──
    if task == "detect" and r.boxes is not None:
        boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else r.boxes.xyxy
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else r.boxes.conf
        clses = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes.cls, 'cpu') else r.boxes.cls.astype(int)

        detections = {
            "boxes": [[float(x1), float(y1), float(x2), float(y2)] for (x1,y1,x2,y2) in boxes],
            "confidences": [float(c) for c in confs],
            "class_ids": [int(c) for c in clses],
            "labels": [str(names.get(c, f"cls_{c}")) for c in clses],
        }
        resp["processed_data"]["detection"] = detections
        resp["stats"]["detection_count"] = len(boxes)
        resp["stats"]["avg_confidence"] = float(np.mean(confs)) if len(confs) else 0

    # ── Classification ──
    elif task == "classify" and r.probs is not None:
        probs = r.probs.cpu().numpy() if hasattr(r.probs, 'cpu') else r.probs
        top5 = np.argsort(probs)[::-1][:5]
        top_predictions = [[str(names.get(i, f"cls_{i}")), float(probs[i])] for i in top5]
        resp["processed_data"]["classification"] = {"top_predictions": top_predictions}

    # ── Pose ──
    elif task == "pose" and r.keypoints is not None:
        kps_xy = r.keypoints.xy.cpu().numpy() if hasattr(r.keypoints.xy, 'cpu') else r.keypoints.xy
        kps_conf = r.keypoints.conf.cpu().numpy() if hasattr(r.keypoints.conf, 'cpu') else (np.ones_like(kps_xy[..., 0]) * 0.5)

        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else r.boxes.xyxy
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else r.boxes.conf
        else:
            boxes = np.array([[0, 0, W, H]])
            confs = np.array([1.0])

        pose_data = {
            "boxes": [[float(x1), float(y1), float(x2), float(y2)] for (x1,y1,x2,y2) in boxes],
            "confidences": [float(c) for c in confs],
            "keypoints": [],
            "keypoints_conf": [],
        }
        for i in range(len(boxes)):
            kp_list = kps_xy[i] if i < len(kps_xy) else kps_xy[0]
            kpc_list = kps_conf[i] if i < len(kps_conf) else kps_conf[0]
            kpts = [[float(kp[0]), float(kp[1])] for kp in kp_list]
            kpcf = [float(k) for k in kpc_list]
            pose_data["keypoints"].append(kpts)
            pose_data["keypoints_conf"].append(kpcf)

        resp["processed_data"]["pose"] = pose_data
        resp["stats"]["keypoint_count"] = sum(len(k) for k in pose_data["keypoints"])
        resp["stats"]["num_people"] = len(boxes)

    # ── Segmentation ──
    elif task == "segment" and r.masks is not None:
        if r.boxes is not None:
            boxes = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else r.boxes.xyxy
            confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else r.boxes.conf
            clses = r.boxes.cls.cpu().numpy().astype(int) if hasattr(r.boxes.cls, 'cpu') else r.boxes.cls.astype(int)
        else:
            boxes = np.array([[0, 0, W, H]])
            confs = np.array([1.0])
            clses = np.array([0])

        seg_data = {
            "boxes": [[float(x1), float(y1), float(x2), float(y2)] for (x1,y1,x2,y2) in boxes],
            "confidences": [float(c) for c in confs],
            "class_ids": [int(c) for c in clses],
        }
        resp["processed_data"]["segmentation"] = seg_data
        resp["stats"]["detection_count"] = len(boxes)
        resp["stats"]["avg_confidence"] = float(np.mean(confs)) if len(confs) else 0

    return {"action": "inference_result", "result": resp}


# ──────────────────────────────────────────────
#  Analyze model handler
# ──────────────────────────────────────────────
def _handle_analyze(req: dict) -> dict:
    model_path = req.get("model_path", "")
    if not model_path or not os.path.exists(model_path):
        return {"action": "error", "message": f"Model not found: {model_path}"}

    try:
        model, task, names = _load_model(model_path)
        info = {
            "model_path": model_path,
            "task_type": task,
            "input_size": "640",
            "class_count": len(names),
            "class_names": list(names.values()) if names else [],
            "num_keypoints": 0,
            "skeleton": [],
        }
        return {"action": "model_info", "info": info}
    except Exception as e:
        return {"action": "error", "message": f"Failed to analyze model: {e}"}


# ──────────────────────────────────────────────
#  Main loop
# ──────────────────────────────────────────────
def main():
    # Signal ready
    sys.stdout.write(json.dumps({"action": "status", "message": "YOLO service ready"}) + "\n")
    sys.stdout.flush()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"action": "error", "message": f"Invalid JSON: {e}"}) + "\n")
            sys.stdout.flush()
            continue

        action = req.get("action", "")

        if action == "shutdown":
            break

        try:
            if action == "inference":
                resp = _handle_inference(req)
            elif action == "analyze_model":
                resp = _handle_analyze(req)
            else:
                resp = {"action": "error", "message": f"Unknown action: {action}"}
        except Exception as e:
            tb = traceback.format_exc()
            resp = {"action": "error", "message": f"{e}\n{tb}"}

        sys.stdout.write(json.dumps(resp) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
