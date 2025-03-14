import os
import cv2
import pandas as pd
import numpy as np
import glob
import shutil


def remove_folder_recursively(folder_path):
    try:
        shutil.rmtree(folder_path)
        print(f"Successfully deleted {folder_path}")
    except Exception as e:
        print(f"Error removing folder {folder_path}: {e}")

def get_csv(folder_path):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv")) 
    return csv_files[0] if csv_files else None

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

def center_pos_frames(df_bboxes,  min_bounds, max_bounds):
    frame_count = max_bounds - min_bounds
    #Get center frame
    center_frame = min_bounds + int(frame_count / 2)
    #Get the frame closest to the center one, with a valid detection (cecause not every frame might have a bbox detection)
    center_index = min(df_bboxes['frame'], key=lambda v: abs(v - center_frame))

    #Get bbox data, top left (x1, y1) bottom right (x2, y2)
    bbox = xyxy_to_square(df_bboxes["x1"][center_index], df_bboxes["y1"][center_index], df_bboxes["x2"][center_index], df_bboxes["y2"][center_index])
    cutouts = [bbox for _ in range(frame_count)]

    return cutouts    

def extract_frames(video_dir:str, file_name:str, csv_dir:str, patient_id:str, REMclass:str, cropped_dir:str):
    video_input_path =  os.path.join(video_dir, file_name)
    cap = cv2.VideoCapture(video_input_path)

    #Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    print(f'PROCESSING VIDEO - {video_input_path}')

    aug_frame_count = 3 #total number of frames used for temporal data augmentation = 3 frames before AND after original clip, so original clip is [3, length-4]
    min_bounds = aug_frame_count
    max_bounds = frame_count - 1 - aug_frame_count

    frame_stack_count = 6
    df_bboxes = pd.read_csv(csv_dir)

    center_frames = center_pos_frames(df_bboxes, min_bounds, max_bounds)
    interpolate_pos_frames = []
    every_pos_frames = []

    frame_indices = np.linspace(min_bounds, max_bounds, frame_stack_count, dtype=int).tolist()
    print(f'Frame indices {frame_indices}')


    center_frames_dir = os.path.join(cropped_dir, "center", patient_id, REMclass, file_name.replace(".mp4", ""))
    if not os.path.exists(center_frames_dir): os.makedirs(center_frames_dir)
    interpolate_frames_dir = ""
    every_frames_dir = ""

    # write results to video, for debugging
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    center_vid = cv2.VideoWriter(center_frames_dir, fourcc, fps, (frame_width, frame_height))
    print(f'Center vid dir: {center_frames_dir}')
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
        current_frame+=1
        if(current_frame < min_bounds or current_frame > max_bounds): continue

        x1, y1, x2, y2 = center_frames[current_frame - min_bounds]

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
    cropped_dir = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cropped")
    remove_folder_recursively(cropped_dir)
    video_dir:str = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cutout")
    frames_dir:str = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "frames")
    #for patient in os.listdir(video_dir):
    patient = "554_02-03-2023"
    patient_dir:str = os.path.join(video_dir, patient)
    for eye_state_dir in os.listdir(patient_dir):
        fragment_dir:str = os.path.join(patient_dir, eye_state_dir)
        for fragment_file in os.listdir(fragment_dir):
            bbox_csv = get_csv(os.path.join(frames_dir, patient, eye_state_dir, fragment_file.replace(".mp4", "")))
            extract_frames(fragment_dir, fragment_file, bbox_csv, patient, eye_state_dir, cropped_dir)

detect_vid()