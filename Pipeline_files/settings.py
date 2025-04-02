import os

fragment_length = 45

cur_vid = "929-processed.mp4"

eye_loc_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "loc")
yolo_weights_path = os.path.join(os.path.abspath(os.getcwd()), "runs", "OC", "open-closed", "weights", "best.pt")
video_path = os.path.join(os.path.abspath(os.getcwd()), cur_vid)