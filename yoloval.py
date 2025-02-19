from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train5/weights/best.pt")

# Customize validation settings
validation_results = model.val(data="SLAPIaabb.yaml", imgsz=640, split="test", save_json=True, device="0")

print(validation_results.box.map50)