"""
This script contains all global settings for the pipeline.

Author: Lisa Groen
Date: April 30, 2025
"""

import os

cur_vid = "228-processed.mp4"
all_vids = ["223-processed.mov", "228-processed.mp4", "318-processed.mov", "417-processed.mp4", "929-processed.mp4", "360-processed.mp4"]
fragment_length = 90 if cur_vid[-4:] == ".mov" else 45 #mov files are at 60 fps, mp4's at 30 fps

frame_stack_count = 6 #number of frames in a stack that is used as input for the REM model
img_size = 64

model_filename = "model_architecture.json"
checkpoint_filename = "checkpoint.model.keras"

is_combined = True #whether to use the combined model, or the open-closed model

eye_loc_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "loc") #path containing eye loclization data
model_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "model") #path containing model weights
eye_frag_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "frags") #path containing all fragments of a full length video
predictions_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "predictions") #path containing REM predition over a full length video
yolo_weights_path = os.path.join(os.path.abspath(os.getcwd()), "runs", "OC", "open-closed", "weights", "best.pt") #path containing YOLO eye localization weights
video_path = os.path.join(os.path.abspath(os.getcwd()), cur_vid) #path containing current video that is processed