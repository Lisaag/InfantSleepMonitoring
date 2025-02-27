import torch
from ultralytics import YOLO
import os

model = YOLO("yolo11l.pt")



image_path = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "aug", "test", "images", "frame_CG_360_29-03-2022-7.jpg")
# Get the penultimate fully connected layer
def get_penultimate_fc_output(module, input, output):
    # Save the output of the second-to-last fully connected layer
    print(output)  # This will print the output of the layer

# Register the hook
layer_name = 'model.model[-2]'  # This is just an example, you need to identify the right layer
model.model.model[-2].register_forward_hook(get_penultimate_fc_output)

# Now, run a sample image through the model to get the output
results = model(image_path)  # Run the image through the model

#model = YOLO(os.path.join(os.path.abspath(os.getcwd()), "runs", "AUG", "default-aug", "weights", "best.pt"))



