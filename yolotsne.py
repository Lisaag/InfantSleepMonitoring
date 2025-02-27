import torch
from ultralytics import YOLO
import os

#model = YOLO("yolo11l.pt")
model = YOLO(os.path.join(os.path.abspath(os.getcwd()), "runs", "AUG", "default-aug", "weights", "best.pt"))



#image_path = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "aug", "test", "images", "frame_CG_360_29-03-2022-7.jpg")
image_path = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "aug", "test", "images", "frame_417_12-10-2022-47.jpg")
# Get the penultimate fully connected layer
def get_penultimate_fc_output(module, input, output):
    # Save the output of the second-to-last fully connected layer
    #print(f'SHAPE {input.size()}')
    #print(input)  # This will print the output of the layer
    #print(f'SHAPE {output.size()}')
    #print(output)
    print(module)
    

# Register the hook
print(f'MODEL {len(model.model.model)}')
model.model.model[-1].register_forward_hook(get_penultimate_fc_output)

# Now, run a sample image through the model to get the output
results = model(image_path)  # Run the image through the model




