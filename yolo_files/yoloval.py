"""
This script is used to validate the YOLO model performance.
Set the model to the trained weights you want to validate, and make sure to set the corresponding yaml file.

Author: Lisa Groen
Date: May 9, 2025
"""
from ultralytics import YOLO

model = YOLO("runs/OCC/occ3/weights/best.pt")

validation_results = model.val(data="occ.yaml", imgsz=640, split="test", save_json=True, device="0", iou=0.5)

print(f'map50: {validation_results.box.ap50}')
print(f'map95: {validation_results.box.ap}')
print(f'box precision: {validation_results.box.p}')
print(f'box recall: {validation_results.box.r}')
print(f'box f1: {validation_results.box.f1}')


