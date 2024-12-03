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

    # Render predictions on the frame
    rendered_frame = results.render()[0]

    # Write the frame to the output video
    out.write(rendered_frame)

    # Optional: Display the frame (comment out if not needed)
    # cv2.imshow('YOLO Prediction', rendered_frame)
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processed video saved at {output_video_path}")
