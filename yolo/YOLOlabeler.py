import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import re
import cv2
import pandas as pd
import settings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import List
import math

#directories
all_labels_dir = ""
all_images_dir = ""

# all_aabb_labels_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "labels", "aabb")
# all_obb_labels_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "labels", "obb")
all_images_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "images")
vis_aabb_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "vis", "aabb")
vis_obb_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "vis", "obb")
    
##Delete all files in directory
def delete_files_in_directory(directory_path):
   try:
     files = os.listdir(directory_path)
     for file in files:
       file_path = os.path.join(directory_path, file)
       if os.path.isfile(file_path):
         os.remove(file_path)
     print("All files deleted successfully.")
   except OSError:
     print("Error occurred while deleting files.")

##Get aabb data from raw annotation data
def get_aabb_from_string(input_string: str):
    pattern = r'"x":(\d+),"y":(\d+),"width":(\d+),"height":(\d+)'

    match = re.search(pattern, input_string)

    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        width = int(match.group(3))
        height = int(match.group(4))
        return ([x, y, width, height])
    else:
        print(input_string)
        return None
    
##Get obb data from raw annotation data
def get_obb_from_string(input_string: str):
    pattern = r'"name":"polygon","all_points_x":\[(\d+),(\d+),(\d+),(\d+)\],"all_points_y":\[(\d+),(\d+),(\d+),(\d+)\]'

    match = re.search(pattern, input_string)

    if match:
        all_points_x = [int(match.group(1)), int(match.group(2)), int(match.group(3)), int(match.group(4))]
        all_points_y = [int(match.group(5)), int(match.group(6)), int(match.group(7)), int(match.group(8))]
        return all_points_x, all_points_y
    else:
        print(input_string + " no match")
        return None
    
def str_to_bool(s: str):
   return s.lower() == "true"

#write aabb label in YOLO format
def write_aabb_label(file_name, dir_name, x, y, w, h, object_class):
    file_name = re.sub(r'\.jpg$', '', file_name)

    with open(os.path.join(os.path.abspath(os.getcwd()), dir_name, file_name + '.txt'), "a") as file:
        file.write(str(object_class) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")

#write obb label in YOLO format
def write_obb_label(file_name, dir_name, all_points_x, all_points_y, object_class):
    file_name = re.sub(r'\.jpg$', '', file_name)

    with open(os.path.join(os.path.abspath(os.getcwd()), dir_name, file_name + '.txt'), "a") as file:
        file.write(str(object_class) + " " + str(all_points_x[0]) + " " + str(all_points_y[0]) + " "  + str(all_points_x[1]) + " " + str(all_points_y[1]) + " "+ str(all_points_x[2]) + " " + str(all_points_y[2]) + " "+ str(all_points_x[3]) + " " + str(all_points_y[3]) + "\n")

#Draw aabb on image to check if implementation is correct
def test_aabb(file_name, x, y, w, h):
    im_path = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "vis", "aabb", file_name)
    if(os.path.exists(im_path)):
        image = cv2.imread(im_path, cv2.IMREAD_COLOR)
    else: 
        im_path = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "images", file_name)
        image = cv2.imread(im_path, cv2.IMREAD_COLOR)

    #To draw a rectangle, you need top-left corner and bottom-right corner of rectangle.
    cv2.rectangle(image, (int(x-(w/2)), int(y-(h/2))), (int(x+(w/2)), int(y+(h/2))), (0,255,0), 3)
    cv2.circle(image,(int(x-(w/2)), int(y-(h/2))), 10, (255,0,0), -1)
    if not cv2.imwrite(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "vis", "aabb", file_name), image):
        print("imwrite failed")

#Draw aabb on image to check if implementation is correct
def test_obb(file_name, all_points_x, all_points_y):
    im_path = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "vis", "obb", file_name)
    if(os.path.exists(im_path)):
        image = cv2.imread(im_path, cv2.IMREAD_COLOR)
    else: 
        im_path = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "images", file_name)
        image = cv2.imread(im_path, cv2.IMREAD_COLOR)

    pts = np.array([[all_points_x[0], all_points_y[0]],[all_points_x[1], all_points_y[1]],[all_points_x[2], all_points_y[2]],[all_points_x[3], all_points_y[3]]], np.int32)
    pts = pts.reshape((-1,1,2))
    cv2.polylines(image,[pts],True,(0,255,255))
    cv2.circle(image, (all_points_x[0], all_points_y[0]), 10, (255,0,0), -1)
    if not cv2.imwrite(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "vis", "obb", file_name), image):
        print("imwrite failed")

##Create dummy for testing without data
def create_dummy_data(file_name, dir_name):
    file_name = re.sub(r'\.jpg$', '', file_name)

    with open(os.path.join(os.path.abspath(os.getcwd()), dir_name, file_name + '.jpg'), "w") as file:
        file.write("0 ")

def get_attributes_from_string(input_string: str):
    open_pattern = r'"open":"(true|false)"'
    occlusion_pattern = r'"occlusion":{(.*?)}'
    side_pattern = r'"side":"(true|false)"'

    open_match = re.search(open_pattern, input_string)
    occlusion_match = re.search(occlusion_pattern, input_string)
    side_match = re.search(side_pattern, input_string)

    open_value = open_match.group(1) if open_match else None
    open_value = str_to_bool(open_value)
    occlusion_value = (
        re.findall(r'"(\w+)":(?:true|false)', occlusion_match.group(1)) if occlusion_match else []
    )
    side_value = side_match.group(1) if side_match else None
    side_value = str_to_bool(side_value)

    return [open_value, occlusion_value, side_value]

@dataclass
class Split:
    open_samples: List[str] = field(default_factory=list)
    closed_samples: List[str] = field(default_factory=list)
    open_samples_occ: List[str] = field(default_factory=list)
    closed_samples_occ: List[str] = field(default_factory=list)

def reduce_splits(train_split:Split, val_split:Split, test_split:Split, percentage:float):
    percentage = min(percentage, 100) / 100 #cap to 100%

    train_samples = train_split.open_samples[0:math.ceil(len(train_split.open_samples) * percentage)] + train_split.closed_samples[0:math.ceil(len(train_split.closed_samples) * percentage)]
    val_samples = val_split.open_samples[0:math.ceil(len(val_split.open_samples) * percentage)] + val_split.closed_samples[0:math.ceil(len(val_split.closed_samples) * percentage)]
    test_samples = test_split.open_samples[0:math.ceil(len(test_split.open_samples) * percentage)] + test_split.closed_samples[0:math.ceil(len(test_split.closed_samples) * percentage)]

    print(f'reduced {percentage * 100}%: train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}, total: {len(train_samples)+len(val_samples)+len(test_samples)}')

    return(train_samples, val_samples, test_samples)


def create_yolo_labels():
    global all_labels_dir
    all_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "labels", "aabb")
        #delete_files_in_directory(all_labels_dir)
        #delete_files_in_directory(vis_aabb_dir)
    df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "annotations", "aabb.csv"))
    stats = defaultdict(lambda: [0, 0, defaultdict(lambda: 0)])

    train_ids = [137, 260, 416, 440, 524, 554, 614, 616, 701, 773, 777, 863, 867, 887, 901, 866, 704, 657, 778, 976]
    val_ids =  [4, 399, 875, 971, 43]
    test_ids = [228, 360, 417, 545, 663, 929]

    train_split = Split()
    val_split = Split()
    test_split = Split()

    curr_split = Split()

    total_samples = 0
    total_samples_filtered = 0 #total samples not including those with occlusions
    
    for i in range(len(df_all)):
        match = re.search(r'frame_(?:CG_)?(.*)', df_all["filename"][i])
        attributes = get_attributes_from_string(df_all["region_attributes"][i])
        if attributes[2]: continue
        
        total_samples += 1

        patient_id = int(match.group(1)[0:3])
        if(patient_id in train_ids): curr_split = train_split
        elif(patient_id in val_ids): curr_split = val_split
        elif(patient_id in test_ids): curr_split = test_split

        if(('none' in attributes[1]) or (len(attributes[1]) == 1 and attributes[1][0] == 'shadow')):
            if(attributes[0]): curr_split.open_samples.append(df_all["filename"][i])
            else: curr_split.closed_samples.append(df_all["filename"][i])
            total_samples_filtered += 1
        else:
            if(attributes[0]): curr_split.open_samples_occ.append(df_all["filename"][i])
            else: curr_split.closed_samples_occ.append(df_all["filename"][i])

    print(f'TOTAL {total_samples}')
    print(f'TOTAL FILTERED {total_samples_filtered}')
    print(f'TRAIN O:{len(train_split.open_samples)} - C:{len(train_split.closed_samples)} OCCLUDED O:{len(train_split.open_samples_occ)} - C:{len(train_split.closed_samples_occ)}')
    print(f'VAL O:{len(val_split.open_samples)} - C:{len(val_split.closed_samples)} OCCLUDED O:{len(val_split.open_samples_occ)} - C:{len(val_split.closed_samples_occ)}')
    print(f'TEST O:{len(test_split.open_samples)} - C:{len(test_split.closed_samples)} OCCLUDED O:{len(test_split.open_samples_occ)} - C:{len(test_split.closed_samples_occ)}')

    reduce_splits(train_split, val_split, test_split, 25)
    reduce_splits(train_split, val_split, test_split, 50)
    reduce_splits(train_split, val_split, test_split, 75)
    reduce_splits(train_split, val_split, test_split, 100)


    #         x, y, w, h = get_aabb_from_string(df_all["region_shape_attributes"][i])
    #         x=x+(w/2)
    #         y=y+(h/2)

    #         if(not is_dummy):
    #             test_aabb(df_all["filename"][i], x, y, w, h)
    #             image = cv2.imread(os.path.join(os.path.abspath(os.getcwd()), all_images_dir, df_all["filename"][i]))
    #             height, width, _ = image.shape
    #             x/=width; w/=width; y/=height; h/=height

    #         object_class = "0"
    #         if(annotation_type == "ocaabb"):
    #             attributes = get_attributes_from_string(df_all["region_attributes"][i])
    #             if(attributes[0]): object_class = "1"

    #         write_aabb_label(df_all["filename"][i], all_labels_dir, x, y, w, h, object_class)
    # elif(annotation_type == "obb" or annotation_type == "ocobb"):
    #     delete_files_in_directory(all_labels_dir)
    #     delete_files_in_directory(vis_obb_dir)
    #     df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "annotations", "obb.csv"))
    #     for i in range(len(df_all)):
    #         all_points_x, all_points_y = get_obb_from_string(df_all["region_shape_attributes"][i])
    #         if(not is_dummy):
    #             test_obb(df_all["filename"][i], all_points_x, all_points_y)
    #             image = cv2.imread(os.path.join(os.path.abspath(os.getcwd()), all_images_dir, df_all["filename"][i]))
    #             height, width, _ = image.shape
    #             for x in range(len(all_points_x)): all_points_x[x] /= width
    #             for y in range(len(all_points_y)): all_points_y[y] /= height

    #         object_class = "0"
    #         if(annotation_type == "ocobb"):
    #             attributes = get_attributes_from_string(df_all["region_attributes"][i])
    #             if(attributes[0]): object_class = "1"

    #         write_obb_label(df_all["filename"][i], all_labels_dir, all_points_x, all_points_y, object_class)
    # else:
    #     print("unknown annotation type")

create_yolo_labels()