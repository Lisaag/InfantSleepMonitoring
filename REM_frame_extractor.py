import os
import cv2
import pandas as pd
import numpy as np
import glob

def get_csv(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv")) 
    return os.path.join(folder_path, csv_files[0]) if csv_files else None

#get the xyxy, as 1:1 square ratio
def xyxy_to_square(x1, y1, x2, y2):
    #Get width and heigh, as  1:1 ratio
    width = int(abs(x1 - x2))
    height = width

    #get bounding box center
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    #get new xyxy positions, with 1:1 ratio
    x1 = int(x_center - width / 2)
    x2 = int(x_center + width / 2)
    y1 = int(y_center - height / 2)
    y2 = int(y_center + height / 2)

    return [x1, y1, x2, y2]

def center_pos_frames(df_bboxes):
    #Get center frame index
    frame_count = len(df_bboxes)
    center_index = int(frame_count / 2)

    #Get bbox data, top left (x1, y1) bottom right (x2, y2)
    bbox = xyxy_to_square(df_bboxes["x1"][center_index], df_bboxes["y1"][center_index], df_bboxes["x2"][center_index], df_bboxes["y2"][center_index])
    cutouts = [bbox for _ in range(frame_count)]

    return cutouts    

def extract_frames(video_dir:str, file_name:str, csv_dir:str, patient_id:str, REMclass:str):
    video_input_path =  os.path.join(video_dir, file_name)
    cap = cv2.VideoCapture(video_input_path)

    aug_frame_count = 3 #total number of frames used for temporal data augmentation = 3 frames before AND after original clip, so original clip is [3, length-4]

    #Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    frame_stack_count = 6
    df_bboxes = pd.read_csv(csv_dir)

    frame_indices = np.linspace(aug_frame_count, len(df_bboxes) - aug_frame_count - 1, frame_stack_count, dtype=int).tolist()
    #frame_indices = np.linspace(0, len(df_bboxes) - aug_frame_count - aug_frame_count - 1, frame_stack_count, dtype=int).tolist()
    #frame_indices = np.linspace(aug_frame_count+aug_frame_count, len(df_bboxes) - 1, frame_stack_count, dtype=int).tolist()

    center_pos_frames = center_pos_frames(df_bboxes[aug_frame_count, int(len(df_bboxes)) - 1 -aug_frame_count])
    interpolate_pos_frames = []
    every_pos_frames = []

    cropped_dir = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cropped")
    center_frames_dir = os.path.join(cropped_dir, "center", patient_id, REMclass, file_name.replace(".mp4", ""))
    if not os.path.exists(center_frames_dir): os.makedirs(center_frames_dir)
    interpolate_frames_dir = ""
    every_frames_dir = ""

    # write results to video, for debugging
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    center_vid = cv2.VideoWriter(center_frames_dir, fourcc, fps, (frame_width, frame_height))
    #interpolate_vid = cv2.VideoWriter(interpolate_frames_dir, fourcc, fps, (frame_width, frame_height))
    #every_vid = cv2.VideoWriter(every_frames_dir, fourcc, fps, (frame_width, frame_height))
    
    #TODO return array of n evenly spaced integers between 3 and framecount-3 (bc of augmentation offset)
    #TODO return array of center pos 
    #TODO return array of interpolate between first and last
    #TODO return array of every YOLO frame

    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count+=1 

        x1, y1, x2, y2 = center_pos_frames[current_frame]

        #save stack of frames
        if np.isin(current_frame, frame_indices):
            cv2.imwrite(os.path.join(center_frames_dir, "FRAME" + str(current_frame) + ".jpg"), frame[y1:y2, x1:x2])  


        #show crop in original video for debugging
        frame_center = frame.copy()
        #frame_interpolate = frame.copy()
        #frame_every = frame.copy()

        cv2.rectangle(frame_center, (x1, y1), (x2, y2), (0, 255, 0), 2)
        center_vid.write(frame_center)

    cap.release()
    center_vid.release()

        

def detect_vid():
    video_dir:str = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cutout")
    frames_dir:str = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "frames")
    #for patient in os.listdir(video_dir):
    patient = "554_02-03-2023"
    patient_dir:str = os.path.join(video_dir, patient)
    for eye_state_dir in os.listdir(patient_dir):
        fragment_dir:str = os.path.join(patient_dir, eye_state_dir)
        for fragment_file in os.listdir(fragment_dir):
            bbox_csv = get_csv(os.path.join(frames_dir, patient, eye_state_dir, fragment_file.replace(".mp4", "")))
            extract_frames(fragment_dir, fragment_file, bbox_csv, patient, eye_state_dir)

detect_vid()