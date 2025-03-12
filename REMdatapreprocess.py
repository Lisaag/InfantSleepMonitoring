import pandas as pd
import os
import cv2


video_input_path = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cutout", "554_02-03-2023")

for REMclass in os.listdir(video_input_path):
    for fragment in os.listdir(os.path.join(video_input_path, REMclass)):

        # Open the video file
        cap = cv2.VideoCapture(os.path.join(video_input_path, REMclass, fragment))
        current_frame = 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"FRAMECOUNT {frame_count}")
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break    