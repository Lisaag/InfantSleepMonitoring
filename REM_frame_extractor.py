import os
import cv2
import pandas as pd

def extract_evenly_spaced_elements(arr, count):
    n = len(arr)
    
    if n <= count:
        return arr

    step = (n - 1) / (count-1)
    extracted = [arr[round(i * step)] for i in range(count)]

    return extracted


def extract_frames(video_dir:str, file_name:str, csv_dir:str):
    video_input_path =  os.path.join(video_dir, file_name)
    # Open the video file
    cap = cv2.VideoCapture(video_input_path)

    current_frame = 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    df_all = pd.read_csv(csv_dir)

    #TODO return array of n evenly spaced integers between 3 and framecount-3 (bc of augmentation offset)
    #TODO return array of center pos 
    #TODO return array of interpolate between first and last
    #TODO return array of every YOLO frame

    # Process each frame
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count+=1   


def detect_vid():
    video_dir:str = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cutout")
    frames_dir:str = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "frames")
    for patient in os.listdir(video_dir):
        patient_dir:str = os.path.join(video_dir, patient)
        for eye_state_dir in os.listdir(patient_dir):
            fragment_dir:str = os.path.join(patient_dir, eye_state_dir)
            for fragment_file in os.listdir(fragment_dir):
                os.path.join(frames_dir, patient, eye_state_dir)
