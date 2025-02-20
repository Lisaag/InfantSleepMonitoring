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
    epochs=400,
    imgsz=640,
    patience=0,
    device=0,     
    hsv_s=0.4,
    hsv_v=0.4,
    translate=0.1,
    scale=0.3,
    crop_fraction=0.85,
    #auto_augment="autoaugment"
    #augment=False
    # hsv_h=0.0,
    # hsv_s=0.0,
    # hsv_v=0.0,
    # degrees=0.0,
    # translate=0.0,
    # scale=0.0,
    # shear=0.0,
    # perspective=0.0,
    # flipud=0.0,
    # fliplr=0.0,
    # bgr=0.0,
    # mosaic=0.0,
    # mixup=0.0,
    # copy_paste=0.0,
    # erasing=0.0,
    # crop_fraction=1.0
)
