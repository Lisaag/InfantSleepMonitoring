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
    hsv_h=(0.01, 0.02),
    hsv_s=(0.4, 0.8),
    degrees=(-25.0, 25.0),
    crop_fraction=(0.6, 1.0),
    translate=(0.0, 0.3),
    scale=0.0,
    fliplr=0.0,
    mosaic=1.0,
    erasing=0.0,
    iterations=100,
    patience=5,
    augment=False,
    device=0
)
