from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11l.pt")

# # Tune hyperparameters on COCO8 for 30 epochs
# model.train(
#     data="SLAPIaabb.yaml",
#     epochs=100,
#     imgsz=640, 
#     augment=False,
#     device=0
# )

model.train(
    data="SLAPIaabb.yaml",
    epochs=100,
    imgsz=640,
    patience=15,
    device=0,
    fliplr=0.0,
    mosaic=0.0,
    erasing=0.0,
    scale=0.0,
    hsv_h=0.0,
    hsv_s=0.0,
    hsv_v=0.0,
    translate=0.0,
    crop_fraction=1.0
)
