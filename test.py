from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO('pose/yolov8n_pose_custom4/weights/best.pt')

# Predict on the video file with confidence threshold, forcing CPU usage
video_path = 'Recording 2025-04-02 160422.mp4'
results = model.predict(video_path, stream=True, save=True, conf=0.6, device='cpu')  # Added device='cpu'

# Process and display each frame
for result in results:
    # Get the annotated frame with only bounding boxes (disable keypoints)
    annotated_frame = result.plot(kpt_line=False, kpt_radius=0)  # Disable keypoints and lines

    # Display the frame in a window
    cv2.imshow('YOLOv8 Pose Prediction', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cv2.destroyAllWindows()

if __name__ == "__main__":
    pass  # Already executed above; no need for a main function here unless adding more logicq