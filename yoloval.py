from ultralytics import YOLO

# Load a model
model = YOLO("runs/OCC/occ/weights/best.pt")

# Customize validation settings
validation_results = model.val(data="occ.yaml", imgsz=640, split="test", save_json=True, device="0", conf=0.002, iou=0.45)

print(f'map50: {validation_results.box.ap50}')
print(f'map95: {validation_results.box.ap}')
print(f'box precision: {validation_results.box.p}')
print(f'box iou: {validation_results.box.iou}')
print(f'box recall: {validation_results.box.r}')
print(f'box f1: {validation_results.box.f1}')


