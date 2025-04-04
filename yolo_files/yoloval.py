from ultralytics import YOLO

model = YOLO("runs/OCC/occ3/weights/best.pt")

validation_results = model.val(data="occ.yaml", imgsz=640, split="test", save_json=True, device="0", iou=0.5)

print(f'map50: {validation_results.box.ap50}')
print(f'map95: {validation_results.box.ap}')
print(f'box precision: {validation_results.box.p}')
print(f'box recall: {validation_results.box.r}')
print(f'box f1: {validation_results.box.f1}')


