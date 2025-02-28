from ultralytics import YOLO

# Load a model
model = YOLO("runs/AUG/aug/weights/best.pt")

# Customize validation settings
validation_results = model.val(data="occ.yaml", imgsz=640, split="test", save_json=True, device="0", conf=0.25, iou=0.45)

print(f'map50: {validation_results.box.ap50}')
print(f'map95: {validation_results.box.ap}')
print(f'precision: {validation_results.box.p}')
print(f'recall: {validation_results.box.r}')
print(f'f1: {validation_results.box.f1}')