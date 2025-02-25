from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11l.pt")

model.train(
    data="aug.yaml",
    epochs=100,
    imgsz=640,
    patience=15,
    device=0,
    plots=True,
    project="runs/AUG",
    name="no-aug",
    auto_augment="randaugment",
    copy_paste_mode="flip",
    hsv_h=0.0,
    hsv_s=0.,
    hsv_v=0.0,
    degrees=0.0,
    translate=0.0,
    scale=0.0,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.0,
    bgr=0.0,
    mosaic=0.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.0,
    crop_fraction=1.0
)
