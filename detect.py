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


def track_vid_aabb(relative_weights_path:str, annotation_type:str="aabb"):
    weights_path = os.path.join(os.path.abspath(os.getcwd()), relative_weights_path)
    model = YOLO(weights_path)
    IN_directory = os.path.join(os.path.abspath(os.getcwd()), "vid", "IN")

    tracking_data = defaultdict(lambda: [])

    all_boxes = defaultdict(lambda: {})

    for filename in os.listdir(IN_directory):
        print(f'Processing {filename}')
        video_output_path =  os.path.join(os.path.abspath(os.getcwd()), "vid", "OUT", "OUT"+str(filename))
        video_input_path =  os.path.join(os.path.abspath(os.getcwd()), "vid", "IN", filename)
        # Open the video file
        cap = cv2.VideoCapture(video_input_path)

        # Store the track history
        track_history = defaultdict(lambda: [])
        box_history = defaultdict(lambda: {})
        current_track_id = -1
        previous_track_id = -1
        max_track_epoch = 15
        current_track_epoch = 1
        test = 0
        current_frame = 0
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break      
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            results = model.track(frame, verbose=False, persist=True)

            # Draw predictions on the frame
            for result in results:  # Iterate through detections

                               #Save bboxes

                boxes = result.boxes  # Get bounding boxes
                if(boxes.id == None): continue

                track_ids = boxes.id.int().cpu().tolist()

                for box, track_id in zip(boxes, track_ids):
                    conf = float(box.conf[0])  # Confidence score
                    x1, y1, x2, y2 = map(int, box.xyxy[0]) 
                    track_history[track_id].append(conf)
                    box_history[track_id][current_frame] = [x1,y1,x2,y2]

                if (current_track_epoch == max_track_epoch or current_frame == frame_count-1):
                    #if currently tracked object does not exist in current epoch, set to -1
                    if(current_track_id not in track_history): current_track_id = -1
                    for key in track_history.keys():
                        if(current_track_id == -1):
                            current_track_id = key
                            continue
                        if(current_track_id == key):
                            continue
                        highest_length = len(track_history[current_track_id])
                        current_length = len(track_history[key])
                        #First determine the initial object to be detected
                        if(previous_track_id == -1):
                            #Track the one with the highest number of detections within max_track_epoch epochs
                            if(current_length > highest_length):
                                current_track_id = key
                            #If the number of detections is the same, pick the one with the highest mean confidence score
                            elif(current_length == highest_length):
                                    if(statistics.mean(track_history[key]) > statistics.mean(track_history[current_track_id])):
                                        current_track_id = key
                        #For following detections..
                        else:
                            #if current track length is higher than previous highest, then switch
                            if(current_length > highest_length + (highest_length * 0.2)):
                                current_track_id = key

                    previous_track_id = current_track_id
                    tracking_data[filename].append(current_track_id)
                    tracked_boxes = box_history[current_track_id]
                    print(f'num tb{len(tracked_boxes)}')
                    for key in tracked_boxes:
                        print(current_frame)
                        test+=1

                        all_boxes[filename][key] = tracked_boxes[key]
                    box_history = defaultdict(lambda: {})
                    track_history = defaultdict(lambda: [])
                    current_track_epoch = 0


                current_track_epoch += 1
                current_frame += 1             

        # Release resources
        cap.release()
        cv2.destroyAllWindows()
       
        print(f"Finish processing {video_output_path}")
        print(f'TEST {test}')
    print(all_boxes)
    return all_boxes

def detect_vid_aabb_filter(box:defaultdict):
    ratio = 16/9

    IN_directory = os.path.join(os.path.abspath(os.getcwd()), "vid", "IN")
    for filename in os.listdir(IN_directory):
        video_output_path =  os.path.join(os.path.abspath(os.getcwd()), "vid", "OUT", "OUT"+str(filename))
        video_input_path =  os.path.join(os.path.abspath(os.getcwd()), "vid", "IN", filename)
        frame_output_path =  os.path.join(os.path.abspath(os.getcwd()), "vid", "frag", filename[0:len(filename)-4])
        # Open the video file
        cap = cv2.VideoCapture(video_input_path)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

        current_frame = 0
        num_boxes = 0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_indices = np.linspace(0, frame_count, 6, dtype=int)


        if filename in box:
            keys = list(box[filename].keys())
            center_index = len(keys) // 2  # Integer division
            center_key = keys[center_index]
            x1, y1, x2, y2 = box[filename][center_key]
            height = abs(y1 - y2)
            width = height * ratio
            x_center = (x1 + x2) / 2
            x1 = int(x_center - width / 2)
            x2 = int(x_center + width / 2)
        else:
            print(f'Tracking info of file {filename} not found!!')

        
        out2 = cv2.VideoWriter(os.path.join(os.path.abspath(os.getcwd()), "vid", "OUT", "OUTCROP"+str(filename)), fourcc, fps, (x2-x1+1, y2-y1+1))
        out2.write(frame[y1:y2+1, x1:x2+1])


        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if np.isin(current_frame, frame_indices):
                cv2.imwrite(os.path.join(frame_output_path, "FRAME", current_frame + ".jpg"), frame)



            if box[filename].get(current_frame) != None:
                # top-left corner and bottom-right corner of rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            out.write(frame)

            current_frame += 1
        

        print(f'Frame count: {frame_count}')
        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved at {video_output_path}")


def detect_vid_aabb(relative_weights_path:str):
    weights_path = os.path.join(os.path.abspath(os.getcwd()), relative_weights_path)
    model = YOLO(weights_path)

    IN_directory = os.path.join(os.path.abspath(os.getcwd()), "vid", "IN")
    OUT_directory = os.path.join(os.path.abspath(os.getcwd()), "vid", "OUT")
    for filename in os.listdir(IN_directory):
        print(filename)
        video_output_path =  os.path.join(os.path.abspath(os.getcwd()), "vid", "OUT", "OUT"+str(filename))
        video_input_path =  os.path.join(os.path.abspath(os.getcwd()), "vid", "IN", filename)
        # Open the video file
        cap = cv2.VideoCapture(video_input_path)

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))


        frame_count = 0
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count+=1

            # Make predictions
            results = model(frame, verbose=False)


            # Draw predictions on the frame
            for result in results:  # Iterate through detections
                boxes = result.boxes  # Get bounding boxes
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                    conf = box.conf[0]  # Confidence score
                    cls = box.cls[0]  # Class index

                    boxColor = (255, 0, 0)
                    if cls == 0: boxColor = (0, 255, 0)
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), boxColor, 2)
                    label = f"{model.names[int(cls)]} {conf:.2f}"  # Class label and confidence
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.0, boxColor, 2)

            # Write the frame to the output video
            out.write(frame)

            # Optional: Display the frame (comment out if not needed)
            # cv2.imshow('YOLO Prediction', frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

        # Release resources
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Processed video saved at {video_output_path}")



def detect_vid(annotation_type:str, relative_weights_path:str, detection_type:str):
    if(annotation_type == "aabb" or annotation_type == "ocaabb"):
        if (detection_type == "detect"):
            detect_vid_aabb(relative_weights_path)
        elif (detection_type == "track"):
            all_boxes = track_vid_aabb(relative_weights_path, annotation_type)
            detect_vid_aabb_filter(all_boxes)
    else:
        print("annotation type [" + annotation_type + "] not recognized")

