import cv2
from ultralytics import YOLO
from pathlib import Path
import os

# Define the paths
weights_path = os.path.join(os.path.abspath(os.getcwd()), "runs", "detect", "train2", "weights", "best.pt")
video_input_path = os.path.join(os.path.abspath(os.getcwd()), "vid","1.mp4")
output_video_path = os.path.join(os.path.abspath(os.getcwd()), "vid","output.mp4")


# Load the YOLO model
model = YOLO(weights_path)

# Open the video file
cap = cv2.VideoCapture(video_input_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# Process each frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Make predictions
    results = model(frame)

    # Draw predictions on the frame
    for result in results:  # Iterate through detections
        boxes = result.boxes  # Get bounding boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
            conf = box.conf[0]  # Confidence score
            cls = box.cls[0]  # Class index

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            label = f"{model.names[int(cls)]} {conf:.2f}"  # Class label and confidence
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

    # Write the frame to the output video
    out.write(frame)

    # Optional: Display the frame (comment out if not needed)
    # cv2.imshow('YOLO Prediction', frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at {output_video_path}")
