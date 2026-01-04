import os
import cv2
from ultralytics import YOLO
from gemini_api import GeminiAPI
from datetime import datetime
from PIL import Image
from main_face import FaceTorsoExtractor  # Assuming class is in activity_face_pipeline.py

def process_video_for_highest_confidence_frame(video_path, confidence_threshold=0.6):
    model = YOLO('weights_2/best.pt')
    model.to('cpu')

    cap = cv2.VideoCapture(video_path)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = video_name
    os.makedirs(output_dir, exist_ok=True)

    max_confidence = 0.0
    best_frame = None
    best_frame_idx = -1
    best_confidences = None
    best_box = None
    prev_frame = None
    next_frame = None
    prev_frame_idx = -1
    class_labels = ['climbing', 'normal_walk', 'snatching']

    ret, prev_frame = cap.read()
    if not ret:
        print("Failed to read the video.")
        cap.release()
        return None, None, None, None, output_dir

    frame_count = 1

    while cap.isOpened():
        ret, current_frame = cap.read()
        if not ret:
            break

        results = model(current_frame)

        for result in results:
            if result.boxes is not None:
                confidences = result.boxes.conf
                classes = result.boxes.cls
                boxes = result.boxes.xyxy

                if len(confidences) > 0:
                    max_conf = confidences.max().item()
                    max_conf_idx = confidences.argmax().item()

                    if max_conf >= confidence_threshold:
                        detected_class = class_labels[int(classes[max_conf_idx])]

                        if max_conf > max_confidence:
                            max_confidence = max_conf
                            best_frame = current_frame.copy()
                            best_frame_idx = frame_count
                            best_class = detected_class
                            best_confidences = {class_labels[int(cls)]: conf.item() for cls, conf in zip(classes, confidences)}
                            best_box = boxes[max_conf_idx].cpu().numpy()
                            best_prev_frame = prev_frame.copy() if prev_frame is not None else None
                            best_prev_frame_idx = prev_frame_idx
                            ret, next_frame = cap.read()
                            if ret:
                                best_next_frame = next_frame.copy()
                                best_next_frame_idx = frame_count + 1
                            else:
                                best_next_frame = None
                                best_next_frame_idx = -1

        prev_frame = current_frame.copy()
        prev_frame_idx = frame_count
        frame_count += 1

    cap.release()

    annotated_output_path = None

    if best_frame is not None:
        if best_prev_frame is not None:
            cv2.imwrite(os.path.join(output_dir, f'frame_before_best_{best_class}_index_{best_prev_frame_idx}.jpg'), best_prev_frame)
        if best_next_frame is not None:
            cv2.imwrite(os.path.join(output_dir, f'frame_after_best_{best_class}_index_{best_next_frame_idx}.jpg'), best_next_frame)

        annotated_frame = best_frame.copy()
        if best_box is not None:
            x1, y1, x2, y2 = best_box.astype(int)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{best_class}: {max_confidence:.4f}'
            cv2.putText(annotated_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        y_offset = 30
        for class_name, conf in best_confidences.items():
            text = f'{class_name}: {conf:.4f}'
            cv2.putText(annotated_frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            y_offset += 30

        annotated_output_path = os.path.join(output_dir, f'best_frame_{best_class}_with_{max_confidence:.4f}_confidence_levels.jpg')
        cv2.imwrite(annotated_output_path, annotated_frame)
        print(f"[✓] Saved annotated frame at {annotated_output_path}")

    return annotated_output_path, best_class, max_confidence, best_box, output_dir


def run_pipeline(video_path, api_key, user_question1, user_question2):
    print("[INFO] Running detection and Gemini analysis pipeline...")

    current_dt = datetime.now()
    current_datetime = current_dt.strftime("%I:%M %p IST on %A, %B %d, %Y")

    image_path, detected_class, confidence, yolo_box, output_dir = process_video_for_highest_confidence_frame(video_path)
    if image_path is None:
        print("[ERROR] No confident frame found.")
        return

    print(f"[✓] Using frame: {image_path} (Class: {detected_class}, Confidence: {confidence:.4f})")

    # Perform face + torso extraction from YOLO box
    if yolo_box is not None:
        print("[INFO] Performing face and torso extraction...")
        extractor = FaceTorsoExtractor()
        extractor.extract_and_save(image_path, tuple(map(int, yolo_box)), output_dir)

    questions = [user_question1, user_question2]
    gemini = GeminiAPI(api_key)
    results = []
    for i, question in enumerate(questions, 1):
        print(f"\n[INFO] Processing question {i}: {question}")
        result = gemini.generate_analysis_from_image(image_path, question)
        results.append((question, result))
        print(f"\n=== Gemini Analysis Result for Question {i} ===\n{result}")

    report_path = os.path.join(output_dir, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=== Analysis Report ===\n\n")
        f.write(f"Date and Time: {current_datetime}\n\n")
        for i, (question, result) in enumerate(results, 1):
            f.write(f"Question {i}: {question}\n")
            f.write(f"Gemini Response {i}:\n{result}\n")
            if i < len(results):
                f.write("\n-----\n\n")

    print(f"[✓] Analysis report saved at: {report_path}")
    return results


if __name__ == "__main__":
    video_path = "Untitled video - Made with Clipchamp_2.mp4"
    api_key = "AIzaSyA_YUyQB6HXSHPWDNpxAFJ2ZxvJwrLUj4U"
    user_question1 = "Is there any suspicious or theft-related activity in this image?"
    user_question2 = "Describe the person performing it — clothing color (upper and lower), helmet/headgear, etc"
    run_pipeline(video_path, api_key, user_question1, user_question2)
