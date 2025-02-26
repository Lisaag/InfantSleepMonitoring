from ultralytics import YOLO

model = YOLO("yolo11l.pt")

model.train(
    data="aug.yaml",
    epochs=100,
    imgsz=640,
    patience=15,
    device=0,
    plots=True,
    project="runs/AUG",
    name="sub9-aug",
    auto_augment="randaugment",
    copy_paste_mode="flip",
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    bgr=0.0,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.0,
    erasing=0.4,
    crop_fraction=0.8
)