import cv2
from ultralytics import YOLO  # Replace with appropriate YOLO class import

# Initialize model
model = YOLO(weights="path/to/yolov5s.pt")

# Load video
cap = cv2.VideoCapture("path/to/video.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Predict using YOLO
    results = model(frame)

    # Annotate frame
    annotated_frame = results.render()  # Draw bounding boxes
    cv2.imshow("Predictions", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()