import cv2
from ultralytics import YOLO
from pathlib import Path
import os
import numpy as np
from itertools import chain 
from collections import defaultdict
import statistics
import ast


import csv

def extract_evenly_spaced_elements(arr):
    n = len(arr)
    
    # If array size is less than 6, return all elements.
    if n <= 6:
        return arr

    # Calculate the step size for evenly spaced elements.
    step = (n - 1) / 5

    # Extract 6 evenly spaced elements including the first and last.
    extracted = [arr[round(i * step)] for i in range(6)]

    return extracted


def track_vid_aabb(relative_weights_path:str, root_dir:str, file_name:str):
    weights_path = os.path.join(os.path.abspath(os.getcwd()), relative_weights_path)
    model = YOLO(weights_path)

    print(f'Processing {file_name} from dir {root_dir}')
    video_input_path =  os.path.join(root_dir, "raw", file_name)
    # Open the video file
    cap = cv2.VideoCapture(video_input_path)

    # Store the track history
    box_history = defaultdict(lambda: {})

    current_frame = 0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Process each frame
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
                box_history[track_id][current_frame] = [x1,y1,x2,y2]


        current_frame += 1   
        
    to_del = list()
    for key in box_history.keys():
        if(len(box_history[key]) < frame_count / 2):
            to_del.append(key)

    for index in to_del:       
        del box_history[index]         

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    return box_history

def save_boxes_csv(boxes:defaultdict, root_dir:str, file_name:str):
    box_index = 0

    fragement_dir = os.path.join(root_dir, "data", file_name.replace(".mp4", ""))

    print(f'WRITING TO {fragement_dir}')

    if not os.path.exists(fragement_dir):
        os.makedirs(fragement_dir)

    for key in boxes.keys():
        dir = os.path.join(fragement_dir, str(box_index) + ".csv")

        with open(dir, "w") as file:
            file.write("frame;box" + "\n")

        for k in boxes[key].keys():
            with open(dir, "a") as file:
                file.write(str(k) + ";" + str(boxes[key][k]) + "\n")

        box_index += 1

def read_boxes_csv(fragment_dir:str):
    box_csv_files = [file for file in os.listdir(fragment_dir) if file.endswith('.csv')]

    boxes = list()

    for c in box_csv_files:
        box = defaultdict(lambda: [])
        with open(os.path.join(fragment_dir, c), newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=';')
            for row in reader:
                box[int(row['frame'])] = ast.literal_eval(row['box'])
        boxes.append(box)

    return boxes

def write_bbox(boxes:defaultdict, root_dir:str, file_name:str):
    box_data = list()

    ratio = 1/1

    bbox_folder = os.path.join(root_dir, "data", file_name.replace(".mp4", ""))
    if not os.path.exists(bbox_folder): os.makedirs(bbox_folder)
    bbox_video_output_path =  os.path.join(bbox_folder, file_name)
    video_input_path =  os.path.join(root_dir, "raw", file_name)

    cap = cv2.VideoCapture(video_input_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
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

        box_data.append([x1, y1, width, height])
    

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if boxes[key].get(current_frame) != None:
            # top-left corner and bottom-right corner of rectangle
            box_idx = 0
            for box in box_data:
                box_idx += 1
                cv2.putText(frame, str(box_idx), (x1,   y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                   1.5, (0, 255, 0), 2, cv2.LINE_AA)
                x1, y1, width, height = box
                cv2.rectangle(frame, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)

        out_bbox.write(frame)

        current_frame += 1

    # Release resources
    cap.release()
    out_bbox.release()
    cv2.destroyAllWindows()


def write_cropped_frames(boxes:defaultdict, root_dir:str, fragment_name:str):
    box_index = 0
    for box in boxes:
        box_index += 1
        ratio = 1/1
        cropped_dir = os.path.join(root_dir, "data", fragment_name, str(box_index))
        frames_dir = os.path.join(root_dir, "data", fragment_name, str(box_index), "frames")

        video_input_path =  os.path.join(root_dir, "raw", fragment_name+".mp4")
        if not os.path.exists(cropped_dir): os.makedirs(cropped_dir)
        cropped_video_output_path =  os.path.join(cropped_dir, fragment_name+".mp4")
        frame_output_path =  os.path.join(frames_dir, fragment_name)
        if not os.path.exists(frame_output_path): os.makedirs(frame_output_path)
        # Open the video file
        cap = cv2.VideoCapture(video_input_path)

        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        current_frame = 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = np.linspace(0, frame_count-1, 6, dtype=int)

        keys = list(box.keys())
        center_index = len(keys) // 2 
        center_key = keys[center_index]
        x1, y1, x2, y2 = box[center_key]
        width = int(abs(x1 - x2))
        height = int(width * ratio)
        x_center = (x1 + x2) / 2
        x1 = int(x_center - width / 2)
        x2 = int(x_center + width / 2)
        y_center = (y1 + y2) / 2
        y1 = int(y_center - height / 2)
        y2 = int(y_center + height / 2)
    
        out_cropped = cv2.VideoWriter(cropped_video_output_path, fourcc, fps, (width, height))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            #save sequence of frames
            if np.isin(current_frame, frame_indices):
                print(f'saved frame {os.path.join(frame_output_path, "FRAME" + str(current_frame) + ".jpg")}, size {width} x {height}')
                cv2.imwrite(os.path.join(frame_output_path, "FRAME" + str(current_frame) + ".jpg"), frame[y1:y1+height, x1:x1+width])

            out_cropped.write(frame[y1:y1+width, x1:x1+height])

            current_frame += 1

        # Release resources
        cap.release()
        out_cropped.release()
        cv2.destroyAllWindows()


def make_dataset(patient_nr:str):
    if(patient_nr == "all"):
        root_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags")
        for patient in os.listdir(root_dir):
            patient_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags", patient)
            for eye_state_dir in os.listdir(patient_dir):
                fragment_dir:str = os.path.join(patient_dir, eye_state_dir, "data")
                if not os.path.exists(fragment_dir):
                    print(f'No fragments in {os.path.join(patient_dir, eye_state_dir)}')
                    continue
                for fragment in os.listdir(fragment_dir):
                    boxes = read_boxes_csv(os.path.join(fragment_dir, fragment))
                    write_cropped_frames(boxes, os.path.join(patient_dir, eye_state_dir), fragment)
    else:
        patient_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags", patient_nr)
        for eye_state_dir in os.listdir(patient_dir):
            fragment_dir:str = os.path.join(patient_dir, eye_state_dir, "data")
            if not os.path.exists(fragment_dir):
                print(f'No fragments in {os.path.join(patient_dir, eye_state_dir)}')
                continue
            for fragment in os.listdir(fragment_dir):
                    boxes = read_boxes_csv(os.path.join(fragment_dir, fragment))
                    write_cropped_frames(boxes, os.path.join(patient_dir, eye_state_dir), fragment)



def detect_vid(relative_weights_path:str, patient_nr:str):
    if(patient_nr == "all"):
        root_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags")
        for patient in os.listdir(root_dir):
            patient_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags", patient)
            for eye_state_dir in os.listdir(patient_dir):
                fragment_dir:str = os.path.join(patient_dir, eye_state_dir, "raw")
                for fragment_file in os.listdir(fragment_dir):
                    all_boxes = track_vid_aabb(relative_weights_path, os.path.join(patient_dir, eye_state_dir), fragment_file)
                    #detect_vid_aabb_filter(all_boxes, os.path.join(patient_dir, eye_state_dir), fragment_file)
                    save_boxes_csv(all_boxes, os.path.join(patient_dir, eye_state_dir), fragment_file)
                    write_bbox(all_boxes, os.path.join(patient_dir, eye_state_dir), fragment_file)

    else:
        patient_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags", patient_nr)
        for eye_state_dir in os.listdir(patient_dir):
            fragment_dir:str = os.path.join(patient_dir, eye_state_dir, "raw")
            for fragment_file in os.listdir(fragment_dir):
                all_boxes = track_vid_aabb(relative_weights_path, os.path.join(patient_dir, eye_state_dir), fragment_file)
                save_boxes_csv(all_boxes, os.path.join(patient_dir, eye_state_dir), fragment_file)
                write_bbox(all_boxes, os.path.join(patient_dir, eye_state_dir), fragment_file)

                #detect_vid_aabb_filter(all_boxes, os.path.join(patient_dir, eye_state_dir), fragment_file)


