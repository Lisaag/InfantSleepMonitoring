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
    if(len(csv_files) > 1):print(f"MORE THAN 1 CSV FILE FOR {folder_path}")
    elif(len(csv_files) == 0):print(f"NO CSV FILE FOR {folder_path}")
    return csv_files[0] if csv_files else None

#get the xyxy, as 1:1 square ratio
def xyxy_to_square(x1, y1, x2, y2, size, offset = [0, 0]):
    #get bounding box center
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2

    #get new xyxy positions, with 1:1 ratio
    x1 = int((x_center - size / 2) + offset[0])
    x2 = int((x_center + size / 2) + offset[0])
    y1 = int((y_center - size / 2) + offset[1])
    y2 = int((y_center + size / 2) + offset[1])

    return [x1, y1, x2, y2]

def center_pos_frames(df_bboxes,  min_bounds, max_bounds, pos_offset = 0, size_factor = 1):
    #+1, because max_bounds is valid index, not length
    frame_count = max_bounds + 1 - min_bounds

    #Get the frame closest to the first and last, with a valid detection (cecause not every frame might have a bbox detection)
    first_index = min(df_bboxes['frame'], key=lambda v: abs(v - min_bounds))
    last_index = min(df_bboxes['frame'], key=lambda v: abs(v - max_bounds))

    df_index_first = df_bboxes.index[df_bboxes['frame'] == first_index][0]
    df_index_last = df_bboxes.index[df_bboxes['frame'] == last_index][0]

    #size of bounding box as max width
    size = max(int(abs(x1 - x2)) for x1, x2 in zip(df_bboxes['x1'][df_index_first:df_index_last], df_bboxes['x2'][df_index_first:df_index_last]))

    height = max(int(abs(y1 - y2)) for y1, y2 in zip(df_bboxes['y1'][df_index_first:df_index_last], df_bboxes['y2'][df_index_first:df_index_last]))

    #Get center frame
    center_frame = min_bounds + int(frame_count / 2)

    #Get the frame closest to the center one, with a valid detection (cecause not every frame might have a bbox detection)
    center_index = min(df_bboxes['frame'], key=lambda v: abs(v - center_frame))

    #Get bbox data, top left (x1, y1) bottom right (x2, y2)
    _, x1, y1, x2, y2 = df_bboxes.loc[df_bboxes['frame'] == center_index].iloc[0]
    bbox = xyxy_to_square(x1, y1, x2, y2 , size * size_factor, [pos_offset * size, pos_offset * height])

    cutouts = [bbox for _ in range(frame_count)]

    return cutouts    

def interpolate_pos_frames(df_bboxes,  min_bounds, max_bounds, pos_offset = 0, size_factor = 1):
    #+1, because max_bounds is valid index, not length
    frame_count = max_bounds + 1 - min_bounds

    #Get the frame closest to the first and last, with a valid detection (cecause not every frame might have a bbox detection)
    first_index = min(df_bboxes['frame'], key=lambda v: abs(v - min_bounds))
    last_index = min(df_bboxes['frame'], key=lambda v: abs(v - max_bounds))

    df_index_first = df_bboxes.index[df_bboxes['frame'] == first_index][0]
    df_index_last = df_bboxes.index[df_bboxes['frame'] == last_index][0]

    #size of bounding box as max width
    size = max(int(abs(x1 - x2)) for x1, x2 in zip(df_bboxes['x1'][df_index_first:df_index_last], df_bboxes['x2'][df_index_first:df_index_last]))

    height = max(int(abs(y1 - y2)) for y1, y2 in zip(df_bboxes['y1'][df_index_first:df_index_last], df_bboxes['y2'][df_index_first:df_index_last]))

    #Get bbox data, top left (x1, y1) bottom right (x2, y2)
    _, x1, y1, x2, y2 = df_bboxes.loc[df_bboxes['frame'] == first_index].iloc[0]
    x1_first, y1_first, x2_first, y2_first = xyxy_to_square(x1, y1, x2, y2, size * size_factor, [pos_offset * size, pos_offset * height])
    _, x1, y1, x2, y2 = df_bboxes.loc[df_bboxes['frame'] == last_index].iloc[0]
    x1_last, y1_last, x2_last, y2_last =  xyxy_to_square(x1, y1, x2, y2, size * size_factor, [pos_offset * size, pos_offset * height])

    x1_vals = np.linspace(x1_first, x1_last, frame_count, dtype=int)
    y1_vals = np.linspace(y1_first, y1_last, frame_count, dtype=int)
    x2_vals = np.linspace(x2_first, x2_last, frame_count, dtype=int)
    y2_vals = np.linspace(y2_first, y2_last, frame_count, dtype=int)

    cutouts = list(zip(x1_vals, y1_vals, x2_vals, y2_vals))
    return cutouts  

def every_pos_frames(df_bboxes,  min_bounds, max_bounds, pos_offset = 0, size_factor = 1):
    cutouts = []

    #Get the frame closest to the first and last, with a valid detection (cecause not every frame might have a bbox detection)
    first_index = min(df_bboxes['frame'], key=lambda v: abs(v - min_bounds))
    last_index = min(df_bboxes['frame'], key=lambda v: abs(v - max_bounds))

    df_index_first = df_bboxes.index[df_bboxes['frame'] == first_index][0]
    df_index_last = df_bboxes.index[df_bboxes['frame'] == last_index][0]

    #size of bounding box as max width
    size = max(int(abs(x1 - x2)) for x1, x2 in zip(df_bboxes['x1'][df_index_first:df_index_last], df_bboxes['x2'][df_index_first:df_index_last]))

    height = max(int(abs(y1 - y2)) for y1, y2 in zip(df_bboxes['y1'][df_index_first:df_index_last], df_bboxes['y2'][df_index_first:df_index_last]))

    for i in range(min_bounds, max_bounds + 1):
        #not every frame has a localisation, so in that case take the frame closest by that does have a localisation
        index = min(df_bboxes['frame'], key=lambda v: abs(v - i))
        _, x1, y1, x2, y2 = df_bboxes.loc[df_bboxes['frame'] == index].iloc[0]
        bbox = xyxy_to_square(x1, y1, x2, y2, size * size_factor, [pos_offset * size, pos_offset * height])

        cutouts.append(bbox)

    return cutouts  

def save_frame_stack(frame, vid, current_frame, frame_indices, bbox, dir):
    #unpack bbox values
    x1, y1, x2, y2 = bbox
    
    #save stack of frames
    #if np.isin(current_frame, frame_indices):
    cv2.imwrite(os.path.join(dir, "FRAME" + str(current_frame) + ".jpg"), frame[y1:y2, x1:x2])

    #save video for debugging
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    vid.write(frame)


def extract_frames(video_dir:str, file_name:str, csv_dir:str, patient_id:str, REMclass:str, cropped_dir:str, suffix:str="", temp_aug_offset=[3, 3], pos_aug_offset=0, size_factor=1):
    video_input_path =  os.path.join(video_dir, file_name)
    cap = cv2.VideoCapture(video_input_path)

    #Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    min_bounds = temp_aug_offset[0]
    max_bounds = frame_count - 1 - temp_aug_offset[1]

    frame_stack_count = 6
    df_bboxes = pd.read_csv(csv_dir)

    print(f'PROCESSING VIDEO - {video_input_path}')

    center_frames = center_pos_frames(df_bboxes, min_bounds, max_bounds, pos_aug_offset, size_factor)
    interpolate_frames = interpolate_pos_frames(df_bboxes, min_bounds, max_bounds, pos_aug_offset, size_factor)
    every_frames = every_pos_frames(df_bboxes, min_bounds, max_bounds, pos_aug_offset, size_factor)


    frame_indices = np.linspace(min_bounds, max_bounds, frame_stack_count, dtype=int).tolist()

    center_frames_dir = os.path.join(cropped_dir, "center", patient_id, REMclass, file_name.replace(".mp4", "")+suffix)
    if not os.path.exists(center_frames_dir): os.makedirs(center_frames_dir)
    interpolate_frames_dir = os.path.join(cropped_dir, "interpolate", patient_id, REMclass, file_name.replace(".mp4", "")+suffix)
    if not os.path.exists(interpolate_frames_dir): os.makedirs(interpolate_frames_dir)
    every_frames_dir = os.path.join(cropped_dir, "every", patient_id, REMclass, file_name.replace(".mp4", "")+suffix)
    if not os.path.exists(every_frames_dir): os.makedirs(every_frames_dir)


    # write results to video, for debugging
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    center_vid = cv2.VideoWriter(os.path.join(center_frames_dir, "center.mp4"), fourcc, fps, (frame_width, frame_height))
    interpolate_vid = cv2.VideoWriter(os.path.join(interpolate_frames_dir, "interpolate.mp4"), fourcc, fps, (frame_width, frame_height))
    every_vid = cv2.VideoWriter(os.path.join(every_frames_dir, "every.mp4"), fourcc, fps, (frame_width, frame_height))


    current_frame = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if(current_frame < min_bounds or current_frame > max_bounds):
            current_frame+=1
            continue

        save_frame_stack(frame.copy(), center_vid, current_frame, frame_indices, center_frames[current_frame - min_bounds], center_frames_dir)
        save_frame_stack(frame.copy(), interpolate_vid, current_frame, frame_indices, interpolate_frames[current_frame - min_bounds], interpolate_frames_dir)
        save_frame_stack(frame.copy(), every_vid, current_frame, frame_indices, every_frames[current_frame - min_bounds], every_frames_dir)

        current_frame+=1


    cap.release()
    center_vid.release()
    interpolate_vid.release()
    cv2.destroyAllWindows()

        

def detect_vid():
    #cropped_dir = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cropped")
    cropped_dir = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cropped")
    #remove_folder_recursively(cropped_dir)
    video_dir:str = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cutout")
    frames_dir:str = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "frames")
    for patient in os.listdir(video_dir):
        # if(patient != '773_02-11-2022'):
        #     continue
        patient_dir:str = os.path.join(video_dir, patient)
        for eye_state_dir in os.listdir(patient_dir):
            fragment_dir:str = os.path.join(patient_dir, eye_state_dir)
            for fragment_file in os.listdir(fragment_dir):
                bbox_csv = get_csv(os.path.join(frames_dir, patient, eye_state_dir, fragment_file.replace(".mp4", "")))
                if (bbox_csv == None): continue
                # extract_frames(fragment_dir, fragment_file, bbox_csv, patient, eye_state_dir, cropped_dir)
                # extract_frames(fragment_dir, fragment_file, bbox_csv, patient, eye_state_dir, cropped_dir, suffix="TEMP1AUG", temp_aug_offset=[0, 6])
                # extract_frames(fragment_dir, fragment_file, bbox_csv, patient, eye_state_dir, cropped_dir, suffix="TEMP2AUG", temp_aug_offset=[6, 0])
                # extract_frames(fragment_dir, fragment_file, bbox_csv, patient, eye_state_dir, cropped_dir, suffix="POS1AUG", pos_aug_offset=0.075)
                # extract_frames(fragment_dir, fragment_file, bbox_csv, patient, eye_state_dir, cropped_dir, suffix="POS2AUG", pos_aug_offset=-0.075)
                extract_frames(fragment_dir, fragment_file, bbox_csv, patient, eye_state_dir, cropped_dir, suffix="SIZE1AUG", size_factor=1.1)
                extract_frames(fragment_dir, fragment_file, bbox_csv, patient, eye_state_dir, cropped_dir, suffix="SIZE2AUG", size_factor=0.9)

detect_vid()