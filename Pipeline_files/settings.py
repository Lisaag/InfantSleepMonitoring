import os

cur_vid = "223-processed.mov"
fragment_length = 90 if cur_vid[-4:] == ".mov" else 45

frame_stack_count = 6
img_size = 64

model_filename = "model_architecture.json"
checkpoint_filename = "checkpoint.model.keras"

eye_loc_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "loc")
model_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "model")
eye_frag_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "frags")
predictions_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "predictions", cur_vid[:-4])
yolo_weights_path = os.path.join(os.path.abspath(os.getcwd()), "runs", "OC", "open-closed", "weights", "best.pt")
video_path = os.path.join(os.path.abspath(os.getcwd()), cur_vid)