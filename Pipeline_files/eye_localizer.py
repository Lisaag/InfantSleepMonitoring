"""
This script is used to track eyes in full-length video's.
Input = path of the full-length video.
Output = a csv file containing the bounding box position, confidence score, class throughout the whole video.
Takes a little time depending on the length of the video.

Author: Lisa Groen
Date: April 30, 2025
"""


from collections import defaultdict
import os

import cv2
from ultralytics import YOLO

import settings

def get_frame_count(path):
    cap = cv2.VideoCapture(path)
    return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    

def track_vid_aabb(frag_idx:int, vid_path:str):
    """
    Track localized instances of eyes across frames of a 1.5 second fragment

    Parameters:
    - frag_idx: index of the fragment of the full-length video
    - vid_path: path of the full-length video
    Returns:
    dict: containing track informataion(bbox size&pos, conf score, class) of each found eye instance
    """
    model = YOLO(settings.yolo_weights_path)

    print(f'Processing {settings.video_path}, fragment index {frag_idx}, frame {frag_idx*settings.fragment_length}')
    cap = cv2.VideoCapture(vid_path)

    box_history = defaultdict(lambda: {})

    frame_idx = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, frag_idx * settings.fragment_length)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break 

        results = model.track(frame, verbose=False, persist=True)

        # Draw predictions on the frame
        for result in results:  # Iterate through detections
            boxes = result.boxes  # Get bounding boxes
            if(boxes.id == None): continue

            track_ids = boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_history[track_id][frag_idx * settings.fragment_length + frame_idx] = [[x1,y1,x2,y2], box.cls.numpy().item(), box.conf.numpy().item()]


        frame_idx+=1
        if(frame_idx >= settings.fragment_length):
            break

    #Delete track instances with detections in less than half of frames
    to_del = list()
    for key in box_history.keys():
        if(len(box_history[key]) < settings.fragment_length / 2):
            to_del.append(key)

    for index in to_del:       
        del box_history[index]             

    cap.release()
    cv2.destroyAllWindows()
    
    return box_history

def save_boxes_csv(boxes:defaultdict, fragment_idx:int):
    """
    append bbox info for a 1.5 second fragment to csv file

    Parameters:
    - boxes: dictionary containing track informataion(bbox size&pos, conf score, class) of each found eye instance
    - fragment_idx: index of the fragment of the full-length video
    """
    starting_frame_idx = settings.fragment_length * fragment_idx
    for i in range(starting_frame_idx, starting_frame_idx + settings.fragment_length):
        frame_boxes = {}; frame_classes = {}; frame_confs = {}   

        for detection in boxes.keys():
            if i in boxes[detection]:
                box,cls,conf = boxes[detection][i]
                frame_boxes[detection] = box; frame_classes[detection] = cls; frame_confs[detection] = conf
    
        with open(os.path.join(settings.eye_loc_path, settings.cur_vid+".csv"), "a") as file:
            file.write(str(i) + ";" + str(frame_boxes) + ";" + str(frame_classes) + ";" + str(frame_confs) + "\n")


def detect_vid(vid_path:str):
    """
    Save eye localization information for n fragments of 1.5 seconds in given video file

    Parameters:
    - vid_path: path of the video
    """
    with open(os.path.join(settings.eye_loc_path, settings.cur_vid+".csv"), "w") as file:
        file.write("frame;boxes;classes;confs" + "\n")

        frame_count = get_frame_count(vid_path) 

        fragment_count = int((frame_count - (frame_count % settings.fragment_length)) / settings.fragment_length)

        for i in range(fragment_count):
            boxes = track_vid_aabb(i, vid_path)
            save_boxes_csv(boxes, i)
    

detect_vid(os.path.join(os.path.abspath(os.getcwd()),"2_out.mp4"))