# 🚨 AI-Guard-Real-Time-Chain-Snatching-Intrusion-Detection

---

## Overview 📖

This repository provides a computer vision pipeline for detecting **snatching** (e.g., grabbing objects) and **climbing** (e.g., scaling walls or fences) activities using **YOLO Pose** 🕺 — a pose estimation variant of YOLOv8. It is optimized for custom datasets and ideal for applications in **surveillance**, **security monitoring**, and **activity recognition** 🔒.

---

## Features ✨

- 🔍 Real-time **human pose detection** using YOLO Pose
- 🏃‍♂️ Action classification for **snatching** and **climbing** behaviors
- 📊 Custom dataset support in **YOLOv8 Pose format**
- 🛠️ Scripts for **data preparation**, **training**, and **inference**
- 🧠 **Frame-level analysis** using **Gemini AI** for descriptive insights
- 🎯 Auto-selection of best frame + face/torso cropping

---

## Prerequisites 🛠️

- Python 3.8+
- PyTorch 1.9+
- OpenCV
- Ultralytics YOLOv8 (Pose variant)
- NVIDIA GPU (recommended for training/inference)

---

## Preparing a Custom Dataset 📂

### 1. Collect Data 📹
- Gather images or videos of snatching and climbing activities.
- Ensure diversity in scenes, lighting, and angles 🌍.

### 2. Annotate Data ✍️
Use tools like **Roboflow**. Annotate:
- Keypoints: Human joints for pose estimation 🦴
- Action Labels: "Snatching", "Climbing", or "Normal Walk" 🏷️  
Export in **Yolov8 Pose format** (`.txt` with keypoints and labels).

### 3. Organize Dataset 📦

```
dataset/
├── train/
│   ├── images/
│   └── labels/
├── val/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
├── data.yaml
```

### 4. Update Configuration ⚙️

```yaml
train: ..path/train/images
val: ..path/valid/images
test: ..path/test/images

kpt_shape: [24, 3]
flip_idx: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]

nc: 3
names: ['climbing', 'normal_walk', 'snatching']

roboflow:
  workspace: sunfibo
  project: snatching-bgizk
  version: 12
  license: CC BY 4.0
  url: https://universe.roboflow.com/sunfibo/snatching-bgizk/dataset/12
```

---

## Instructions to Run YOLOv8 Pose Model 🧠

### 1. Check Ultralytics Installation

- Open a terminal and activate your environment:

```bash
source myenv/bin/activate  # Windows: myenv\Scripts\activate
```

- Check if Ultralytics is installed:

```bash
pip show ultralytics
```

- If not installed:

```bash
pip install ultralytics
```


### 2. Verify Model and Video Paths

- Ensure your model file exists:

```bash
ls weights/best.pt  # Windows: dir weights\best.pt
```

- Confirm the video file is available:

```bash
ls /path/to/your/input_video.mp4  # Windows: dir \path\to\your\input_video.mp4
```

- Update the script to use the correct model and video paths.


### 3. Set Device to CPU if No GPU

- Launch Python:

```bash
python
```

- Run the following:

```python
import torch
print(torch.cuda.is_available())
```

- If it prints `False`, set `device="cpu"` in the script’s `predict` method.


### 4. Adjust Confidence Level

- In the script, set the detection confidence:

```python
conf=0.6  # Balanced for CPU
```

- Alternatives:
  - `< 0.6` – for more detections (more false positives)
  - `0.6 >` – for stricter detection (less false positives)


### 5. Run Model and Check Output

- Run the script with these parameters:

```bash
python infer.py --weights/best.pt --source /path/to/your/input_video.mp4 --device cpu --conf 0.5
```

- Results will be saved in:

```
runs/pose/predict, runs/pose/predict2, ...
```

- Check the latest folder for output results.

> 🔁 **Note**: Replace `/path/to/your/input_video.mp4` with the actual video path in your script.

---

## 🔄 Activity Detection & Gemini Frame Analysis Pipeline

This pipeline (`main_pipeline.py`) extends beyond simple detection by integrating Google Gemini to generate detailed frame-level insights.

### 💡 What It Does

1. Takes an **input video**.
2. Detects activities like **snatching** and **climbing** using YOLO Pose.
3. Extracts:
   - **Best frame** (highest confidence)
   - **Previous frame**
   - **Next frame**
4. Passes best frame through:
   - **Face detection** using `MTCNN` via `main_face.py`
   - **Frame description** via `GeminiAPI` in `gemini_api.py`

---

### 📂 Required Files

- `main_pipeline.py` – Orchestrates the pipeline
- `main_face.py` – Contains `FaceTorsoExtractor` class for face/torso detection
- `gemini_api.py` – Contains `GeminiAPI` class to send frames to Gemini API

---

### ✅ How to Run the Pipeline

#### 1. Install Dependencies

Create a `requirements.txt`:

```txt
torch
facenet-pytorch
Pillow
opencv-python
ultralytics
google-generativeai
```

Then install:

```bash
pip install -r requirements.txt
```

#### 2. Configure Gemini API Key

Open `gemini_api.py` and replace:

```python
api_key = "your_gemini_api_key_here"
```

#### 3. Set Input Video Path

In `main_pipeline.py`:

```python
video_path = "path/to/your/input_video.mp4"
```

#### 4. Run the Pipeline

```bash
python main_pipeline.py
```

---

### 📤 Outputs

- **Best frame**
- **Previous frame**
- **Next frame**
- **Cropped face/torso**
- **Gemini-generated frame description**

---

# ⚠ Warning
	Daily updates on Github and JIRA are mandatory.
	Refrain from pushing secret keys, large models and environment files (virtual or ide).
	Upload trained model on Google Drive and update the README here.
	Using personal email id for sharing/backup is strictly prohibited.
