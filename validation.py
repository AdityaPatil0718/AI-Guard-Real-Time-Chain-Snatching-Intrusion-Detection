from ultralytics import YOLO

# Load the model
model = YOLO('Model/pose/yolov8n_pose_custom4/weights/best.pt')  # Pretrained model
# model = YOLO('path/to/custom.pt')  # Custom trained model

# Validate on the dataset
metrics = model.val(data='C:/Users/Aditya Deepak Patil/Desktop/SNATCHING_CLIMBING_FINAL/data.yaml', imgsz=640, device='cpu')  # Use GPU (device=0) or CPU (device='cpu')

# Print metrics
print(f"mAP@50-95: {metrics.box.map}")  # Bounding box mAP
print(f"mAP@50: {metrics.box.map50}")  # Bounding box mAP at IoU=0.5
print(f"mAP@75: {metrics.box.map75}")  # Bounding box mAP at IoU=0.75
print(f"Keypoint mAP@50-95: {metrics.kp.map}")  # Keypoint mAP