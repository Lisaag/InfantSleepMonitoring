import os

cur_vid = "228-processed.mp4"
all_vids = ["223-processed.mov", "228-processed.mp4", "318-processed.mov", "417-processed.mp4", "929-processed.mp4", "360-processed.mp4"]
fragment_length = 90 if cur_vid[-4:] == ".mov" else 45

frame_stack_count = 6
img_size = 64

model_filename = "model_architecture.json"
checkpoint_filename = "checkpoint.model.keras"

is_combined = True

eye_loc_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "loc")
model_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "model")
eye_frag_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "frags")
predictions_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "predictions")
yolo_weights_path = os.path.join(os.path.abspath(os.getcwd()), "runs", "OC", "open-closed", "weights", "best.pt")
video_path = os.path.join(os.path.abspath(os.getcwd()), cur_vid)