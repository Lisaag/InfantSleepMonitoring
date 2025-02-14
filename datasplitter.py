import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass, field
from typing import List
import re
import shutil
import pandas as pd
from dataclasses import dataclass, field
from typing import List
import math
import cv2

#directories
train_labels_dir = ""
train_images_dir = ""
val_labels_dir = ""
val_images_dir = ""
test_labels_dir = ""
test_images_dir = ""
all_labels_dir = ""
all_images_dir = ""

@dataclass
class SetInfo:
    open_count = 0
    closed_count = 0
    occlusion_count = 0

@dataclass
class SampleInfo:
    file_names: List[str] = field(default_factory=list)
    open_count = 0
    closed_count = 0

all_data:dict = dict()
train:dict = dict()
test:dict = dict()
val:dict = dict()

train_info = SetInfo()
val_info = SetInfo()
test_info = SetInfo()

def str_to_bool(s: str):
   return s.lower() == "true"

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

def copy_files(old_image_path:str, old_label_path, new_image_path:str, new_label_path:str, image_filename:str, label_filename:str, prefix:str = ""):
    shutil.copy(os.path.join(old_image_path, prefix+image_filename), os.path.join(new_image_path, prefix+image_filename))
    shutil.copy(os.path.join(old_label_path, prefix+label_filename), os.path.join(new_label_path, prefix+label_filename))

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
    print(f'FILECOUNT reduced {percentage * 100}%: train: {len(set(train_samples))}, val: {len(set(val_samples))}, test: {len(set(test_samples))}, total: {len(set(train_samples))+len(set(val_samples))+len(set(test_samples))}')

    return(train_samples, val_samples, test_samples)

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

#write aabb label in YOLO format
def write_aabb_label(file_name, dir_name, x, y, w, h, object_class):
    file_name = re.sub(r'\.jpg$', '', file_name)

    with open(os.path.join(os.path.abspath(os.getcwd()), dir_name, file_name + '.txt'), "a") as file:
        file.write(str(object_class) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")

def create_splits(split_type):
    global train_labels_dir
    train_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "train", "labels")
    global train_images_dir
    train_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "train", "images")
    global val_labels_dir
    val_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "val", "labels")
    global val_images_dir
    val_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type,"val", "images")
    global test_labels_dir
    test_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "test", "labels")
    global test_images_dir
    test_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "test", "images")
    global all_labels_dir
    all_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "labels")
    global all_images_dir
    all_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "images")

    delete_files_in_directory(train_labels_dir)
    delete_files_in_directory(test_labels_dir)
    delete_files_in_directory(val_labels_dir)
    delete_files_in_directory(train_images_dir)
    delete_files_in_directory(test_images_dir)
    delete_files_in_directory(val_images_dir)

    df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "annotations", "aabb.csv"))

    train_ids = [137, 260, 416, 440, 524, 554, 614, 616, 701, 773, 777, 863, 867, 887, 901, 866, 704, 657, 778, 976]
    val_ids =  [4, 399, 875, 971, 43]
    test_ids = [228, 360, 417, 545, 663, 929]

    train_split = Split()
    val_split = Split()
    test_split = Split()

    curr_split = Split()

    total_samples = 0
    total_samples_filtered = 0

    filtered = [] #samples not including those with occlusions
    all = []
    
    for i in range(len(df_all)):
        match = re.search(r'frame_(?:CG_)?(.*)', df_all["filename"][i])
        attributes = get_attributes_from_string(df_all["region_attributes"][i])
        if attributes[2]: continue
        
        total_samples += 1
        all.append(df_all["filename"][i])

        patient_id = int(match.group(1)[0:3])
        if(patient_id in train_ids): curr_split = train_split
        elif(patient_id in val_ids): curr_split = val_split
        elif(patient_id in test_ids): curr_split = test_split

        if(('none' in attributes[1]) or (len(attributes[1]) == 1 and attributes[1][0] == 'shadow')):
            if(attributes[0]): curr_split.open_samples.append(df_all["filename"][i])
            else: curr_split.closed_samples.append(df_all["filename"][i])
            total_samples_filtered += 1
            filtered.append(df_all["filename"][i])

            x, y, w, h = get_aabb_from_string(df_all["region_shape_attributes"][i])
            x=x+(w/2); y=y+(h/2)
            image = cv2.imread(os.path.join(os.path.abspath(os.getcwd()), all_images_dir, df_all["filename"][i]))
            height, width, _ = image.shape  
            x/=width; w/=width; y/=height; h/=height
            write_aabb_label(df_all["filename"][i], all_labels_dir, x, y, w, h, "0")
        else:
            if(attributes[0]): curr_split.open_samples_occ.append(df_all["filename"][i])
            else: curr_split.closed_samples_occ.append(df_all["filename"][i])

    print(f'TOTAL FILES {len(set(all))}')
    print(f'TOTAL FILTERED FILES {len(set(filtered))}')
    print(f'TOTAL SAMPLES {total_samples}')
    print(f'TOTAL FILTERED SAMPLES {total_samples_filtered}')
    print(f'TRAIN O:{len(train_split.open_samples)} - C:{len(train_split.closed_samples)} OCCLUDED O:{len(train_split.open_samples_occ)} - C:{len(train_split.closed_samples_occ)}')
    print(f'VAL O:{len(val_split.open_samples)} - C:{len(val_split.closed_samples)} OCCLUDED O:{len(val_split.open_samples_occ)} - C:{len(val_split.closed_samples_occ)}')
    print(f'TEST O:{len(test_split.open_samples)} - C:{len(test_split.closed_samples)} OCCLUDED O:{len(test_split.open_samples_occ)} - C:{len(test_split.closed_samples_occ)}')

    train_samples, val_samples, test_samples = reduce_splits(train_split, val_split, test_split, 25)

    for sample in train_samples:
       label_file = re.sub(r'\.jpg$', '', sample) + ".txt"
       copy_files(all_images_dir, all_labels_dir, train_images_dir, train_labels_dir, image_filename=sample, label_filename=label_file)
    for sample in val_samples:
       label_file = re.sub(r'\.jpg$', '', sample) + ".txt"
       copy_files(all_images_dir, all_labels_dir, val_images_dir, val_labels_dir, image_filename=sample, label_filename=label_file)
    for sample in test_samples:
       label_file = re.sub(r'\.jpg$', '', sample) + ".txt"
       copy_files(all_images_dir, all_labels_dir, test_images_dir, test_labels_dir, image_filename=sample, label_filename=label_file)

create_splits("aabb")
