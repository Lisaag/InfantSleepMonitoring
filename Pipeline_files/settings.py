import os

fragment_length = 90

cur_vid = "223-processed.mov"

eye_loc_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "loc")
yolo_weights_path = os.path.join(os.path.abspath(os.getcwd()), "runs", "OC", "open-closed", "weights", "best.pt")
video_path = os.path.join(os.path.abspath(os.getcwd()), cur_vid)