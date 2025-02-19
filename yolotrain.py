from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11l.pt")

# Tune hyperparameters on COCO8 for 30 epochs
model.train(
    data="SLAPIaabb.yaml",
    optimizer="AdamW",
    epochs=100,
    imgsz=640, 
    translate=0.0,
    scale=0.0,
    fliplr=0.0,
    mosaic=1.0,
    erasing=0.0,
    lr0= 0.00761,
    lrf= 0.01,
    weight_decay= 0.00041,
    box= 0.14959,
    cls= 0.3306,
    hsv_h= 0.01262,
    hsv_s= 0.41616,
    hsv_v= 0.48301,
    crop_fraction= 0.95435,
    degrees= 0.0,
    device=0
)