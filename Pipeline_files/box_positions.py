import cv2
import pandas as pd
import os
import settings
import ast

import statistics

from collections import defaultdict

import numpy as np


def get_frame_count(path):
    cap = cv2.VideoCapture(path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

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


df_bboxes = pd.read_csv(os.path.join(settings.eye_loc_path, settings.cur_vid +".csv"), delimiter=';')

for vid in range(2, 19):  
    vid_path = os.path.join(os.path.abspath(os.getcwd()), str(vid)+"_out.mp4")

    frame_count = get_frame_count(vid_path) 
    fragment_count = int((frame_count - (frame_count % settings.fragment_length)) / settings.fragment_length)
    
    for i in range(0, fragment_count):
        print(f'Processing fragment {i} out of {fragment_count}')
        boxes, classes = get_boxes(df_bboxes, i)
        if boxes is None: 
            print(f"NO DETECTIONS FOR FRAGMENT {i}")
            continue

        with open(os.path.join(fragment_path, "info.csv"), "a") as file:
            file.write(str(i) + ";" + str([[(x1+x2)//2, (y1+y2)//2] for box in boxes if box is not None for x1, y1, x2, y2 in [box]])+ ";" + str(classes.count(1.0)) + "\n")
