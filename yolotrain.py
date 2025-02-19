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
    scale=0.0,
    fliplr=0.0,
    mosaic=1.0,
    erasing=0.0,
    device=0
)
