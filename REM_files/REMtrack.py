"""
This script is used to track eyes, and save bounding boxes of eye with highest mean confidence score of fragments for the REM dataset.

Author: Lisa Groen
Date: May 7, 2025
"""
import cv2
from collections import defaultdict
import os
from ultralytics import YOLO



def track_eyes(relative_weights_path:str, root_dir:str, file_name:str):
    """
    Track eyes, omit localizations with only few detections

    Parameters:
    relative_weights_path: relative path to YOLO weights (trained to localize infants eyes)
    root_dir: root dir of video
    file_name: video file name

    Returns:
    box_history: contains frame index, and bounding box/confidence score/class information over the fragment of the whole fragment. Can be of 2 eyes in case of 2 localized eyes
    """
    weights_path = os.path.join(os.path.abspath(os.getcwd()), relative_weights_path)
    model = YOLO(weights_path)

    print(f'Processing {file_name} from dir {root_dir}')
    video_input_path =  os.path.join(root_dir, file_name)
    cap = cv2.VideoCapture(video_input_path)

    box_history = defaultdict(lambda: {})

    current_frame = 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break      
        
        results = model.track(frame, verbose=False, persist=True)

        # Draw predictions on the frame
        for result in results:  
            boxes = result.boxes  # Get bounding boxes
            if(boxes.id == None): continue

            track_ids = boxes.id.int().cpu().tolist()

            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                box_history[track_id][current_frame] = [x1,y1,x2,y2]


        current_frame += 1   

    #Delete track instances with only few detections    
    to_del = list()
    for key in box_history.keys():
        if(len(box_history[key]) < frame_count / 2):
            to_del.append(key)

    for index in to_del:       
        del box_history[index]         

    cap.release()
    cv2.destroyAllWindows()
    
    return box_history


def write_bbox(boxes:defaultdict, video_input_path:str, root_dir:str, file_name:str):
    """
    Write bbox over video, for debugging purpose

    Parameters:
    boxes: bbox info over fragment
    video_input_path: root path of video
    file_name: video file name

    """
    box_data = list()

    ratio = 1/1

    bbox_folder = os.path.join(root_dir, file_name.replace(".mp4", ""))
    if not os.path.exists(bbox_folder): os.makedirs(bbox_folder)
    bbox_video_output_path =  os.path.join(bbox_folder, file_name)

    cap = cv2.VideoCapture(os.path.join(video_input_path, file_name))

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_bbox = cv2.VideoWriter(bbox_video_output_path, fourcc, fps, (frame_width, frame_height))

    current_frame = 0

    for key in boxes.keys():
        keys = list(boxes[key].keys())
        center_index = len(keys) // 2 
        center_key = keys[center_index]
        x1, y1, x2, y2 = boxes[key][center_key]
        width = int(abs(x1 - x2))
        height = int(width * ratio)
        x_center = (x1 + x2) / 2
        x1 = int(x_center - width / 2)
        x2 = int(x_center + width / 2)
        y_center = (y1 + y2) / 2
        y1 = int(y_center - height / 2)
        y2 = int(y_center + height / 2)

        box_data.append([key, x1, y1, width, height])
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # top-left corner and bottom-right corner of rectangle
        for box in box_data:
            key, x1, y1, width, height = box
            cv2.putText(frame, str(key), (x1,   y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 
               1.5, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.rectangle(frame, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)

        out_bbox.write(frame)

        current_frame += 1

    cap.release()
    out_bbox.release()
    cv2.destroyAllWindows()


def save_boxes_csv(boxes:defaultdict, root_dir:str, file_name:str):
    """
    Save bbox info to csv file

    Parameters:
    boxes: bboxes, over fragment
    root_dir: dir of video file
    file_name: file name of video file
    """
    fragement_dir = os.path.join(root_dir, file_name.replace(".mp4", ""))

    if not os.path.exists(fragement_dir):
        os.makedirs(fragement_dir)

    for key in boxes.keys():
        dir = os.path.join(fragement_dir, str(key) + ".csv")

        with open(dir, "w") as file:
            file.write("frame,x1,y1,x2,y2" + "\n")

        for k in boxes[key].keys():
            with open(dir, "a") as file:
                file.write(str(k) + "," + str(boxes[key][k][0]) + "," + str(boxes[key][k][1]) + "," + str(boxes[key][k][2]) + "," + str(boxes[key][k][3]) + "\n")


        print(f'Write csv for eye {key}')


def detect_vid(relative_weights_path:str):
    """
    Detect eyes in video

    Parameters:
    relative_weights_path: path to YOLO eye localization wegiths
    """
    root_dir:str = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cutout")
    frames_dir:str = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "frames")

    for patient in os.listdir(root_dir):
        patient_dir:str = os.path.join(root_dir, patient)
        if os.path.exists(os.path.join(frames_dir, patient)): continue
        for eye_state_dir in os.listdir(patient_dir):
            fragment_dir:str = os.path.join(patient_dir, eye_state_dir)
            for fragment_file in os.listdir(fragment_dir):
                all_boxes = track_eyes(relative_weights_path, fragment_dir, fragment_file)
                save_boxes_csv(all_boxes, os.path.join(frames_dir, patient, eye_state_dir), fragment_file)
                write_bbox(all_boxes, fragment_dir, os.path.join(frames_dir, patient, eye_state_dir), fragment_file)

  
detect_vid(os.path.join(os.path.abspath(os.getcwd()), "runs", "OC", "open-closed", "weights", "best.pt"))