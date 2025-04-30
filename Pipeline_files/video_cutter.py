"""
This script is used to crop the eyes from all frames of a full-length video.
It uses the tracked eye positions and bounding box size to cut out the eye.
Saves cropped eyes as jpg, stored in a folder per fragment of 45 frames (1.5 second)
In case of 2 detected eyes, uses only the eye with the highest mean confidence score.

Author: Lisa Groen
Date: April 30, 2025
"""

import ast
from collections import defaultdict
import numpy as np
import os
import pandas as pd
import statistics

import cv2

import settings


#index of last cut, to continue process from that point instead of starting over
def get_last_index(directory):
    existing_folders = []
    for dir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, dir)):
            if(dir.isdigit()):
                existing_folders.append(int(dir))
    
    next_folder = max(existing_folders, default=0) + 1  # Default to 0 if no numeric folders exist
    print(f'NEXT FOLDER {next_folder}')
    return int(next_folder)

def get_frame_count(path):
    cap = cv2.VideoCapture(path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

#In case of no eye detection at the frame of center index, get frame closest to the center index instead
def get_valid_center_index(bboxes):
    center_index = len(bboxes) // 2
    if bboxes[center_index] is not None:
        return center_index
    
    left:int = center_index - 1
    right:int = center_index + 1

    while left >= 0 or right < len(bboxes):
        if left >= 0 and bboxes[left] is not None:
            return left
        if right < len(bboxes) and bboxes[right] is not None:
            return right
        
        left -= 1
        right += 1
    
    print("NO VALID CENTER INDEX FOUND")
    return None

#get the xyxy, as 1:1 square ratio
def xyxy_to_square(x1, y1, x2, y2, size):
    #get bounding box center
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    #get new xyxy positions, with 1:1 ratio
    x1 = int((x_center - size / 2))
    x2 = int((x_center + size / 2))
    y1 = int((y_center - size / 2))
    y2 = int((y_center + size / 2))

    return [x1, y1, x2, y2]

#Get crop size as the max bbox dimension over all frames in a fragment
def get_crop_size(bboxes):
    #Get center frame
    center_index = get_valid_center_index(bboxes)
    x1, y1, x2, y2 = bboxes[center_index]

    bboxes  = [x for x in bboxes if x is not None]
    bboxes = np.array(bboxes).T

    #size of bounding box as max width
    size = max(int(abs(x1 - x2)) for x1, x2 in zip(bboxes[0], bboxes[2]))
    height = max(int(abs(y1 - y2)) for y1, y2 in zip(bboxes[1], bboxes[3]))

    size = max(size, height)

    bbox = xyxy_to_square(x1, y1, x2, y2 , size)

    return bbox

#Crop eye from frames in a video
def crop_eye(frag_idx, box, vid_path, fragment):
    cap = cv2.VideoCapture(vid_path)

    current_frame_idx = settings.fragment_length * frag_idx

    frame_indices = np.linspace(current_frame_idx, current_frame_idx + settings.fragment_length - 1, settings.frame_stack_count, dtype=int).tolist()

    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret: 
            print("FRAME NOT FOUND") 
            continue

        x1, y1, x2, y2 = box

        save_path = os.path.join(settings.eye_frag_path, settings.cur_vid[:-4], str(fragment))
        if not os.path.exists(save_path): os.makedirs(save_path)

        cv2.imwrite(os.path.join(save_path, str(i)+".jpg"), frame[y1:y2, x1:x2])

    cap.release()


#gets boxes with highest mean confidence
def get_boxes(df_bboxes, fragment_idx):
    all_boxes = defaultdict(lambda:[None] * settings.fragment_length); all_classes = defaultdict(lambda:[None] * settings.fragment_length); all_confs = defaultdict(lambda:[None] * settings.fragment_length)

    curr_starting_frame = fragment_idx * settings.fragment_length
    idx = 0
    for i in range(curr_starting_frame, curr_starting_frame + settings.fragment_length):
        row = df_bboxes.loc[df_bboxes['frame'] == i]
        if(row.empty):
            print(f'Row at frame {i} not found in csv file!')
        
        confs:dict = ast.literal_eval(row["confs"].iloc[0])
        classes:dict = ast.literal_eval(row["classes"].iloc[0])
        boxes:dict = ast.literal_eval(row["boxes"].iloc[0])

        for key in boxes.keys():
            all_boxes[key][idx] = boxes[key]
            all_classes[key][idx] = classes[key]
            all_confs[key][idx] = confs[key]
        idx+=1
        
    highest_conf = -1
    highest_conf_idx = -1
    for key in all_confs.keys():
        mean_conf = statistics.mean([x for x in all_confs[key] if x is not None])
        if(mean_conf > highest_conf):
            highest_conf = mean_conf
            highest_conf_idx = key

    if highest_conf == -1: return None, None
    return all_boxes.get(highest_conf_idx), all_classes.get(highest_conf_idx)


fragment_path = os.path.join(settings.eye_frag_path, settings.cur_vid[:-4])
if not os.path.exists(fragment_path): os.makedirs(fragment_path)
with open(os.path.join(fragment_path, "info.csv"), "w") as file:
    file.write("idx;positions;open_count" + "\n")

#Get all localized positions over full-length video
df_bboxes = pd.read_csv(os.path.join(settings.eye_loc_path, settings.cur_vid +".csv"), delimiter=';')

vid_path = os.path.join(os.path.abspath(os.getcwd()), "2_out.mp4")
frame_count = get_frame_count(vid_path) 
fragment_count = int((frame_count - (frame_count % settings.fragment_length)) / settings.fragment_length)

for i in range(0, fragment_count):
    fragment = i
    print(f'Processing fragment {fragment} out of {fragment_count}')
    boxes, classes = get_boxes(df_bboxes, fragment)
    if boxes is None: 
        print(f"NO DETECTIONS FOR FRAGMENT {fragment}")
        continue

    #get cropping size for all frames in fragment
    crop_box = get_crop_size(boxes)
    with open(os.path.join(fragment_path, "info.csv"), "a") as file:
        file.write(str(fragment) + ";" + str([[(x1+x2)//2, (y1+y2)//2] for box in boxes if box is not None for x1, y1, x2, y2 in [box]])+ ";" + str(classes.count(1.0)) + "\n")
    #crop the eyes from fragment
    crop_eye(i, crop_box, vid_path, fragment)