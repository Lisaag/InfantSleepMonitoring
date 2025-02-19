from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train5/weights/best.pt")

# Customize validation settings
validation_results = model.val(data="SLAPIaabb.yaml", imgsz=640, split="test", save_json=True, device="0")

print(f'map50: {validation_results.box.map50}')
print(f'precision: {validation_results.box.precision}')
print(f'recall: {validation_results.box.recall}')
print(f'f1: {validation_results.box.f1}')