from ultralytics import YOLO

# Load a model
model = YOLO("runs/detect/train16/weights/best.pt")

# Customize validation settings
validation_results = model.val(data="OC.yaml", imgsz=640, split="test", save_json=True, device="0")

print(f'map50: {validation_results.box.ap50}')
print(f'map95: {validation_results.box.ap}')
print(f'precision: {validation_results.box.p}')
print(f'recall: {validation_results.box.r}')
print(f'f1: {validation_results.box.f1}')