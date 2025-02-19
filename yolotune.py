#yolo tune data=SLAPIaabb.yaml task=detect model=yolo11l.pt epochs=100 imgsz=640 
# translate=0.0 scale=0.0 fliplr=0.0 mosaic=1.0 erasing=0.0 plots=True save=True val=True device=0 
# hyp="{'lr0': [1e-5, 1e-1], 'lrf'=[0.01, 1.0], 'weight_decay'=[0.0, 0.001], 'box'=[0.01, 0.2], 'cls'=[0.1, 4.0], 
# 'hsv_h'=[0.0, 0.1], 'hsv_s'=[0.0, 0.9], 'hsv_v'=[0.0, 0.9], 'crop_fraction':[0.6, 1.0], 'degrees':[-25.0, 25.0]}"

from ultralytics import YOLO

# Initialize the YOLO model
model = YOLO("yolo11l.pt")

# Define search space
search_space = {'hsv_h':[0.0, 0.1], 'hsv_s':[0.0, 0.9], 'hsv_v':[0.0, 0.9], 'crop_fraction':[0.6, 1.0], 'degrees':[-25.0, 25.0], 'translate':[0.0, 0.3]}

# Tune hyperparameters on COCO8 for 30 epochs
model.tune(
    data="SLAPIaabb.yaml",
    epochs=100,
    imgsz=640, 
    translate=0.0,
    scale=0.0,
    fliplr=0.0,
    mosaic=1.0,
    erasing=0.0,
    iterations=100,
    patience=10,
    space=search_space,
    plots=False,
    save=False,
    val=False,
)