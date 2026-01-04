from facenet_pytorch import MTCNN
from PIL import Image
import torch
import os

class FaceTorsoExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, device=self.device)

    def extract_and_save(self, image_path, yolo_box, output_dir):
        """
        Detect faces in the YOLO cropped region of the image and save upper torso crops.
        """
        image = Image.open(image_path).convert("RGB")
        cropped_region = image.crop(yolo_box)

        boxes, _ = self.mtcnn.detect(cropped_region)

        if boxes is not None:
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = map(int, box)

                # Calculate face height
                face_height = y2 - y1

                # Optional: extend upward slightly for forehead/hair
                new_y1 = max(0, y1 - int(face_height * 0.7))

                # Extend downward more — full upper torso
                new_y2 = min(cropped_region.height, y2 + int(face_height * 1))

                # Add more width padding for broader context
                pad_w = int((x2 - x1) * 0.8)
                new_x1 = max(0, x1 - pad_w)
                new_x2 = min(cropped_region.width, x2 + pad_w)

                # Crop and save
                upper_torso_crop = cropped_region.crop((new_x1, new_y1, new_x2, new_y2))
                save_path = os.path.join(output_dir, f"person_{i+1}_upper_torso.png")
                upper_torso_crop.save(save_path)
                print(f"[✓] Saved torso: {save_path}")
        else:
            print("[INFO] No faces detected in the cropped activity region.")
