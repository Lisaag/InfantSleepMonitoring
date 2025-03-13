from ultralytics import YOLO

model = YOLO("yolo11l.pt")

search_space = {'mosaic':[0.7, 1.0], 'scale':[0.3, 0.7], 'crop_fraction':[0.7, 1.0], 'degrees':[0, 25.0], 'translate':[0.0, 0.3], 'fliplr':[0.3, 0.7]}

model.tune(
    data="aug.yaml",
    epochs=100,
    imgsz=640,
    patience=15,
    device=0,
    iterations=100,
    close_mosaic=0,
    space=search_space,
    plots=False,
    save=False,
    val=True,
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
    crop_fraction=1.0
)