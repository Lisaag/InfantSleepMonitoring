import cv2
from ultralytics import YOLO  # Replace with appropriate YOLO class import
import os
import sys

# Initialize model
model = YOLO("runs/detect/train2/weights/best.pt")

# Load video
cap = cv2.VideoCapture(os.path.join(os.path.abspath(os.getcwd()), "vid","1.mp4"))

output_video = os.path.join(os.path.abspath(os.getcwd()), "vid","output.mp4")  # Path to save the processed video

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict using YOLO
    results = model(frame)

    # Annotate frame with predictions
    annotated_frame = results.render()[0]  # Render annotated frame

    # Write frame to output video
    out.write(annotated_frame)

cap.release()
out.release()