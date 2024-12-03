import supervision as sv
import cv2
import numpy as np
import os


# Initialize model
model = YOLO("runs/detect/train2/weights/best.pt")

classes = ["eye"]
model.set_classes(classes)

def process_frame(frame: np.ndarray, _) -> np.ndarray:
    results = model.infer(frame, text=classes)
    
    detections = sv.Detections.from_inference(results)

    box_annotator = sv.BoxAnnotator(thickness=4, text_thickness=4, text_scale=2)

    labels = [f"{model.names[class_id]} {confidence:0.2f}" for _, _, confidence, class_id, _ in detections]
    frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

    return frame

VIDEO_PATH = os.path.join(os.path.abspath(os.getcwd()), "vid","1.mp4")
sv.process_video(source_path=VIDEO_PATH, target_path=f"result.mp4", callback=process_frame)