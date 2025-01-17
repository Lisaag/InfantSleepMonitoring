import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass, field
from typing import List
import re
import shutil
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import statistics

#directories
train_labels_dir = ""
train_images_dir = ""
val_labels_dir = ""
val_images_dir = ""
test_labels_dir = ""
test_images_dir = ""
all_labels_dir = ""
all_images_dir = ""
all_csv = ""

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

def copy_files(old_image_path:str, old_label_path, new_image_path:str, new_label_path:str, image_filename:str, label_filename:str, prefix:str = ""):
    shutil.copy(os.path.join(old_image_path, prefix+image_filename), os.path.join(new_image_path, prefix+image_filename))
    shutil.copy(os.path.join(old_label_path, prefix+label_filename), os.path.join(new_label_path, prefix+label_filename))

def copy_to_split(file_name:str, attributes):
    open_value, occlusion_value, side_value = attributes
    if(side_value == True):
        return

    aug_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "aug", "labels")
    aug_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets","SLAPI", "raw", "aug", "images")

    label_file = re.sub(r'\.jpg$', '', file_name) + ".txt"

    # Regular expression to extract the desired part
    pattern = r"^frame_(.*?)-\d+\.jpg$"

    # Perform the match
    match = re.search(pattern, file_name)
    # key = re.sub(r'_\d+\.jpg$', '', df_all["filename"][i])
    # print(key)

    if not match:
        print("no match found")
        return

    key = match.group(1)
    #print(key)
    #key = re.sub(r'_\d+\.jpg$', '', file_name)

    if(train.get(key) != None):
        copy_files(all_images_dir, all_labels_dir, train_images_dir, train_labels_dir, file_name, label_file)
        #copy_files(aug_images_dir, aug_labels_dir, train_images_dir, train_labels_dir, file_name, label_file, prefix="CLAHE_")
        copy_files(aug_images_dir, aug_labels_dir, train_images_dir, train_labels_dir, file_name, label_file, prefix="ROT_")
        copy_files(aug_images_dir, aug_labels_dir, train_images_dir, train_labels_dir, file_name, label_file, prefix="CROP_")
        update_set_properties(train_info, attributes)
        print("COPY train")

    elif(test.get(key) != None):
        copy_files(all_images_dir, all_labels_dir, test_images_dir, test_labels_dir, file_name, label_file)
        update_set_properties(test_info, attributes)
        print("COPY test")

    elif(val.get(key) != None):
        copy_files(all_images_dir, all_labels_dir, val_images_dir, val_labels_dir, file_name, label_file)
        #copy_files(aug_images_dir, aug_labels_dir, val_images_dir, val_labels_dir, file_name, label_file, prefix="CLAHE_")
        copy_files(aug_images_dir, aug_labels_dir, val_images_dir, val_labels_dir, file_name, label_file, prefix="ROT_")
        copy_files(aug_images_dir, aug_labels_dir, val_images_dir, val_labels_dir, file_name, label_file, prefix="CROP_")
        update_set_properties(val_info, attributes)
        print("COPY val")


def update_set_properties(set_info, attributes):
    open_value, occlusion_value, side_value = attributes

    if(open_value): set_info.open_count += 1
    else: set_info.closed_count += 1

    # if(occlusion_value[0] != 'none'):
    #     set_info.occlusion_count += 1

def update_sample_properties(sample_info, attributes, file_name):
    open_value, occlusion_value, side_value = attributes

    sample_info.file_names.append(file_name)

    if(open_value): sample_info.open_count += 1
    else: sample_info.closed_count += 1

def divide_train_val(sample_dict:dict, open_val, closed_val):
    val_open_count = 0 #how many open samples are in the validation set currently
    val_closed_count = 0 #how many closed samples are in the validation set currently
    for key in sample_dict:
        print(f'key {key}')
        # print(int(statistics.median(train_val_dic[key].qualities)))
        if(val_open_count < open_val and sample_dict[key].open_count != 0):
            val[key] = 1
            val_open_count += sample_dict[key].open_count
            val_closed_count += sample_dict[key].closed_count
        elif(val_closed_count < closed_val and sample_dict[key].closed_count != 0):
            val[key] = 1
            val_open_count += sample_dict[key].open_count
            val_closed_count += sample_dict[key].closed_count
        else:
            train[key] = 1

def train_val_split():
    df_all = pd.read_csv(all_csv)

    train_val_dic = dict()
    total_open_count = 0
    total_closed_count = 0

    for i in range(len(df_all)):
        # Regular expression to extract the desired part
        pattern = r"^frame_(.*?)-\d+\.jpg$"

        # Perform the match
        match = re.search(pattern, df_all["filename"][i])
        # key = re.sub(r'_\d+\.jpg$', '', df_all["filename"][i])
        # print(key)

        if not match:
            print("no match found")
            continue

        key = match.group(1)
        print(key)

        if key in test: continue

        if key not in train_val_dic:
            train_val_dic[key] = SampleInfo()

        attributes = get_attributes_from_string(df_all["region_attributes"][i])

        if(attributes[0]): total_open_count += 1
        else: total_closed_count += 1
        update_sample_properties(train_val_dic[key], attributes, df_all["filename"][i])


    # q1_samples = dict()
    # q2_samples = dict()
    # q3_samples = dict()

    # open_counts = [0, 0, 0]
    # closed_counts = [0, 0, 0]

    # for key in train_val_dic:
    #     quality_median = int(statistics.median(train_val_dic[key].qualities))
    #     if(quality_median == 1):
    #         q1_samples[key] = train_val_dic[key]
    #         open_counts[0] += q1_samples[key].open_count
    #         closed_counts[0] += q1_samples[key].closed_count
    #     elif(quality_median == 2):
    #         q2_samples[key] = train_val_dic[key]
    #         open_counts[1] += q2_samples[key].open_count
    #         closed_counts[1] += q2_samples[key].closed_count
    #     elif(quality_median >= 3):
    #         q3_samples[key] = train_val_dic[key]
    #         open_counts[2] += q3_samples[key].open_count
    #         closed_counts[2] += q3_samples[key].closed_count


    divide_train_val(train_val_dic, int(total_open_count * 0.2), int(total_closed_count * 0.2))
    # divide_train_val(q2_samples, int(open_counts[1] * 0.2), int(closed_counts[1] * 0.2))
    # divide_train_val(q3_samples, int(open_counts[2] * 0.2), int(closed_counts[2] * 0.2))
        
    print(len(train_val_dic))

def split_dataset(annotation_type:str = "aabb"):
    global train_labels_dir
    train_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", annotation_type, "train", "labels")
    global train_images_dir
    train_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", annotation_type, "train", "images")
    global val_labels_dir
    val_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", annotation_type, "val", "labels")
    global val_images_dir
    val_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", annotation_type,"val", "images")
    global test_labels_dir
    test_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", annotation_type, "test", "labels")
    global test_images_dir
    test_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", annotation_type, "test", "images")
    global all_labels_dir
    all_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "labels", annotation_type)
    global all_images_dir
    all_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "images")

    csv_filename = annotation_type + ".csv"
    if(annotation_type == "ocaabb"): csv_filename = "aabb.csv"
    if(annotation_type == "ocobb"): csv_filename = "obb.csv"
    global all_csv
    all_csv = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "annotations", csv_filename)

    delete_files_in_directory(train_labels_dir)
    delete_files_in_directory(test_labels_dir)
    delete_files_in_directory(val_labels_dir)
    delete_files_in_directory(train_images_dir)
    delete_files_in_directory(test_images_dir)
    delete_files_in_directory(val_images_dir)

    df_info = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "annotations", "info.csv"))

    for i in range(len(df_info)):
        all_data[df_info["file"][i]] = df_info["annotated"][i]

        if(df_info["annotated"][i]): #add to test set if annotated with (OR/CR/C/O)
            test[df_info["file"][i]] = 1#df_info["annotated"][i]
        #     val[df_info["file"][i]] = 1
        # else: 
        #     train[df_info["file"][i]] = 1

        # elif(pma < 32 or pma > 36):
        #     test[df_info["file"][i]] = 1
        #     #TEMP
        #     val[df_info["file"][i]] = df_info["annotated"][i]
        # else:
        #     if (pma != 0): pmas.append(pma)

        #     train[df_info["file"][i]] = df_info["annotated"][i]

    train_val_split()

    print('train={:d}, val={:d}, test={:d}'.format(len(train), len(val), len(test)))


    df_all = pd.read_csv(all_csv)
    
    for i in range(len(df_all)):
        attributes = get_attributes_from_string(df_all["region_attributes"][i])
        copy_to_split(df_all["filename"][i], attributes)


    ##################################CHECKS############################################
    for i in range(len(df_info)):
        count = 0
        if(df_info["file"][i] in train):
            count += 1
        if(df_info["file"][i] in val):
            count += 1
        if(df_info["file"][i] in test):
            count += 1
        if(count > 1):
            print("SAME PATIENT DATA IN MULTIPLE SETS!!! " + df_info["file"][i])
        if(count == 0):
            print("PATIENT DATA NOT IN ANY SET!!! " + df_info["file"][i])


    print('train set open={:d}, closed={:d}'.format(train_info.open_count, train_info.closed_count))
    print('val set open={:d}, closed={:d}'.format(val_info.open_count, val_info.closed_count))
    print('test set open={:d}, closed={:d}'.format(test_info.open_count, test_info.closed_count))

##########################################