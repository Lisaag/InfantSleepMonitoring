import cv2
from ultralytics import YOLO
from pathlib import Path
import os
import numpy as np
from itertools import chain 
from collections import defaultdict
import statistics

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

                # print(f'track id {track_id}')
                # print(boxes)

            # for box, track_id in zip(boxes, track_ids):
            #     x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            #     box_history[track_id][current_frame] = [x1,y1,x2,y2]



            # if (current_track_epoch == max_track_epoch or current_frame == frame_count-1):
            #     #if currently tracked object does not exist in current epoch, set to -1
            #     if(current_track_id not in track_history): current_track_id = -1
            #     for key in track_history.keys():
            #         if(current_track_id == -1):
            #             current_track_id = key
            #             continue
            #         if(current_track_id == key):
            #             continue
            #         highest_length = len(track_history[current_track_id])
            #         current_length = len(track_history[key])
            #         #First determine the initial object to be detected
            #         if(previous_track_id == -1):
            #             #Track the one with the highest number of detections within max_track_epoch epochs
            #             if(current_length > highest_length):
            #                 current_track_id = key
            #         #For following detections..
            #         else:
            #             #if current track length is higher than previous highest, then switch
            #             if(current_length > highest_length + (highest_length * 0.2)):
            #                 current_track_id = key

            #     previous_track_id = current_track_id
            #     tracked_boxes = box_history[current_track_id]
            #     for key in tracked_boxes:

            #         all_boxes[key] = tracked_boxes[key]
            #     box_history = defaultdict(lambda: {})
            #     track_history = defaultdict(lambda: [])
            #     current_track_epoch = 0


            # current_track_epoch += 1
            # current_frame += 1             

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print(box_history)
    return box_history

def detect_vid_aabb_filter(box:defaultdict, root_dir:str, file_name:str):
    ratio = 1/1
    cropped_video_output_path =  os.path.join(root_dir, "cropped", file_name)
    if not os.path.exists(os.path.join(root_dir, "bbox")):
        os.makedirs(os.path.join(root_dir, "bbox"))
    bbox_video_output_path =  os.path.join(root_dir, "bbox", file_name)
    video_input_path =  os.path.join(root_dir, "raw", file_name)
    frame_output_path =  os.path.join(root_dir, "frames", file_name[0:len(file_name)-4])

    if not os.path.exists(frame_output_path):
        os.makedirs(frame_output_path)
    # Open the video file
    cap = cv2.VideoCapture(video_input_path)

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_bbox = cv2.VideoWriter(bbox_video_output_path, fourcc, fps, (frame_width, frame_height))

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

        if box.get(current_frame) != None:
            # top-left corner and bottom-right corner of rectangle
            cv2.rectangle(frame, (x1, y1), (x1+width, y1+height), (0, 255, 0), 2)

        out_cropped.write(frame[y1:y1+width, x1:x1+height])

        out_bbox.write(frame)

        current_frame += 1
    
    # Release resources
    cap.release()
    out_bbox.release()
    cv2.destroyAllWindows()



def detect_vid(relative_weights_path:str, patient_nr:str):
    if(patient_nr == "all"):
        root_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags")
        for patient in os.listdir(root_dir):
            patient_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags", patient)
            for eye_state_dir in os.listdir(patient_dir):
                fragment_dir:str = os.path.join(patient_dir, eye_state_dir, "raw")
                for fragment_file in os.listdir(fragment_dir):
                    all_boxes = track_vid_aabb(relative_weights_path, os.path.join(patient_dir, eye_state_dir), fragment_file)
                    detect_vid_aabb_filter(all_boxes, os.path.join(patient_dir, eye_state_dir), fragment_file)
    else:
        patient_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags", patient_nr)
        for eye_state_dir in os.listdir(patient_dir):
            fragment_dir:str = os.path.join(patient_dir, eye_state_dir, "raw")
            for fragment_file in os.listdir(fragment_dir):
                all_boxes = track_vid_aabb(relative_weights_path, os.path.join(patient_dir, eye_state_dir), fragment_file)

                # detect_vid_aabb_filter(all_boxes, os.path.join(patient_dir, eye_state_dir), fragment_file)


