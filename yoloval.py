from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train14/weights/best.pt")

# Customize validation settings
validation_results = model.val(data="occ.yaml", imgsz=640, split="test", save_json=True, device="0")

print(f'map50: {validation_results.box.map50}')
print(f'map95: {validation_results.box.map}')
print(f'precision: {validation_results.box.p}')
print(f'recall: {validation_results.box.r}')
print(f'f1: {validation_results.box.f1}')