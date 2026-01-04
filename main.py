#pip install ultralytics

from ultralytics import YOLO

def train_model():
    # Load the pretrained YOLOv8n-Pose model
    model = YOLO('yolov8n-pose.pt')

    # Train the model
    results = model.train(
        data='/kaggle/input/snatchingclimbingv6/data.yaml',    # Path to your dataset configuration
        epochs=100,          # Number of training epochs
        imgsz=640,           # Input image size (640x640)
        batch=32,            # Batch size (adjust based on GPU memory)
        optimizer= 'SGD',
        device=[-1,-1],        # GPU device (0 for first GPU, 'cpu' for CPU)
        name='yolov8n_pose_custom',  # Name for this training run
        lr0=0.0001,           # Initial learning rate
        weight_decay=0.0005,
        momentum=0.937,
        # cos_lr = True,       # Momentum for SGD optimizer
        workers = 8
    )
    print("Training completed successfully.")

if __name__ == "__main__":
    train_model()