# 🚨 AI-Guard: Real-Time Chain Snatching & Intrusion Detection

AI-Guard is a real-time computer vision system for detecting **chain snatching**, **violent reach actions**, and **climbing/intrusion activities** using **YOLOv8 Pose Estimation** combined with **rule-based kinematic verification**.

This repository is intended for **CCTV surveillance**, **RTSP streams**, and **urban security monitoring**, with an emphasis on **reducing false positives** compared to single-stage action classifiers.

---

## Overview

The system uses **YOLOv8-Pose** to extract human keypoints and applies **geometric, velocity-based, and temporal rules** to verify whether an interaction represents a real threat.

Unlike pure action-classification models, AI-Guard validates intent using:
- Arm reach geometry
- Wrist–neck proximity
- Motion velocity
- Temporal stability

---

## Features

- Real-time human pose detection using YOLOv8-Pose
- Pose-based activity verification for snatching and climbing
- Wrist–neck proximity analysis
- Arm extension and reach-angle validation
- Velocity-based intent detection
- Temporal smoothing and alert cooldown
- Optimized for crowded scenes
- CPU and GPU compatible
- Optional Gemini AI frame-level scene description

---

## System Pipeline

```
Video / RTSP Stream
        ↓
YOLOv8 Pose Estimation
        ↓
Keypoint Extraction (Wrist, Elbow, Shoulder, Neck)
        ↓
Velocity + Geometry Analysis
        ↓
Rule-Based Threat Verification
        ↓
Alert Generation + Visualization
        ↓
(Optional) Gemini AI Description
```

---

## Tech Stack

- Python 3.8+
- PyTorch
- Ultralytics YOLOv8 (Pose)
- OpenCV
- NumPy
- facenet-pytorch (MTCNN)
- Google Gemini API (optional)

---

## Installation

### Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate         # Windows
```

### Install Dependencies

```bash
pip install ultralytics torch opencv-python numpy facenet-pytorch pillow google-generativeai
```

---

## Dataset Preparation (YOLOv8 Pose)

### Directory Structure

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
├── test/
└── data.yaml
```

### data.yaml Example

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

kpt_shape: [24, 3]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

nc: 2
names: ['climbing', 'snatching']
```

---

## Training YOLOv8-Pose

```bash
yolo task=pose mode=train \
model=yolov8l-pose.pt \
data=data.yaml \
epochs=100 imgsz=640
```

Trained weights will be saved at:

```
runs/pose/train/weights/best.pt
```

---

## Running Real-Time Detection

### Configure Input Source

Edit the configuration in `yolo_pose_esti3.py`:

```python
SOURCE = "path/to/video_or_rtsp"
MODEL_PATH = "yolov8l-pose.pt"
CONF_THRES = 0.5
```

### Run

```bash
python yolo_pose_esti3.py
```

Press **Q** or **ESC** to exit.

---

## Detection Logic (Summary)

A threat is triggered **only when all conditions are satisfied**:

- Wrist approaches victim neck or upper torso
- Arm is sufficiently extended
- Reach angle is anatomically valid
- Wrist velocity exceeds adaptive threshold
- Motion persists across required frames
- Cooldown prevents repeated alerts

This suppresses common false positives such as:
- Falls
- Handshakes
- Accidental contact
- Random gestures

---

## Output

- Real-time annotated video
- Pose skeleton visualization
- Wrist-to-neck threat lines
- HUD with FPS, people count, and alerts
- Console threat logs
- Optional Gemini-generated scene descriptions

---

## Optional: Gemini AI Integration

After a threat is detected:
- The highest-confidence frame is selected
- Face and torso are cropped using MTCNN
- Frame is sent to Gemini API
- A textual scene description is generated

This is useful for **incident reporting** and **forensic review**.

---

## Performance

| Mode | Approx. FPS |
|-----|-------------|
| GPU (RTX-class) | 25–35 |
| CPU (Optimized) | 8–12 |

Performance depends on scene density and resolution.

---

## Future Work

- Multi-camera global ID tracking
- Temporal action models (ST-GCN / SlowFast)
- Edge deployment (TensorRT / TFLite)
- Centralized alert dashboard
- Automated incident report generation

---
