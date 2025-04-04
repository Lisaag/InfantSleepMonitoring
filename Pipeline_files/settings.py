import os

cur_vid = "223-processed.mov"
fragment_length = 90 if cur_vid[-4:] == ".mov" else 45

frame_stack_count = 6

eye_loc_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "loc")
eye_frag_path = os.path.join(os.path.abspath(os.getcwd()), "PIPELINE", "frags")
yolo_weights_path = os.path.join(os.path.abspath(os.getcwd()), "runs", "OC", "open-closed", "weights", "best.pt")
video_path = os.path.join(os.path.abspath(os.getcwd()), cur_vid)