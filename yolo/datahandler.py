import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass, field
from typing import List
import re
import shutil
import cv2
import pandas as pd
import settings
import matplotlib.pyplot as plt
import statistics

#directories
train_labels_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "train", "labels")
train_images_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "train", "images")
val_labels_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "val", "labels")
val_images_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "val", "images")
test_labels_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "test", "labels")
test_images_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "test", "images")
all_labels_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "labels")
all_images_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "images")

@dataclass
class SetInfo:
    open_count = 0
    closed_count = 0
    q1_count = 0
    q2_count = 0
    q3_count = 0
    q4_count = 0
    q5_count = 0
    occlusion_count = 0

@dataclass
class SampleInfo:
    file_names: List[str] = field(default_factory=list)
    open_count = 0
    closed_count = 0
    qualities: List[int] = field(default_factory=list)


all_data:dict = dict()
train:dict = dict()
test:dict = dict()
val:dict = dict()

train_info = SetInfo()
val_info = SetInfo()
test_info = SetInfo()

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
    quality_pattern = r'"quality":"([1-5])"'
    occlusion_pattern = r'"occlusion":{(.*?)}'

    open_match = re.search(open_pattern, input_string)
    quality_match = re.search(quality_pattern, input_string)
    occlusion_match = re.search(occlusion_pattern, input_string)

    open_value = open_match.group(1) if open_match else None
    open_value = str_to_bool(open_value)
    quality_value = int(quality_match.group(1)) if quality_match else None
    occlusion_value = (
        re.findall(r'"(\w+)":(?:true|false)', occlusion_match.group(1)) if occlusion_match else []
    )

    return [open_value, quality_value, occlusion_value]

def copy_to_split(file_name:str, attributes):
    label_file = re.sub(r'\.jpg$', '', file_name) + ".txt"
    key = re.sub(r'_\d+\.jpg$', '', file_name)

    if(train.get(key) != None):
        shutil.copy(os.path.join(all_images_dir, file_name), train_images_dir)
        shutil.copy(os.path.join(all_labels_dir, label_file), train_labels_dir)
        update_set_properties(train_info, attributes)
    if(test.get(key) != None):
        shutil.copy(os.path.join(all_images_dir, file_name), test_images_dir)
        shutil.copy(os.path.join(all_labels_dir, label_file), test_labels_dir)
        update_set_properties(test_info, attributes)
    if(val.get(key) != None):
        shutil.copy(os.path.join(all_images_dir, file_name), val_images_dir)
        shutil.copy(os.path.join(all_labels_dir, label_file), val_labels_dir)
        update_set_properties(val_info, attributes)

def update_set_properties(set_info, attributes):
    open_value, quality_value, occlusion_value = attributes

    if(open_value): set_info.open_count += 1
    else: set_info.closed_count += 1

    if(quality_value==1):       
            set_info.q1_count+=1
    elif(quality_value==2): 
            set_info.q2_count+=1
    elif(quality_value==3): 
            set_info.q3_count+=1
    elif(quality_value==4): 
            set_info.q4_count+=1
    elif(quality_value==5): 
            set_info.q5_count+=1
    else:
        print("something went wrong")

    if(occlusion_value[0] != 'none'):
        set_info.occlusion_count += 1

def update_sample_properties(sample_info, attributes, file_name):
    open_value, quality_value, occlusion_value = attributes

    sample_info.file_names.append(file_name)
    sample_info.qualities.append(attributes[1])

    if(open_value): sample_info.open_count += 1
    else: sample_info.closed_count += 1

def divide_train_val(sample_dict:dict, open_val, closed_val):
    val_open_count = 0 #how many open samples are in the validation set currently
    val_closed_count = 0 #how many closed samples are in the validation set currently
    for key in sample_dict:
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
    df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "annotations", "all.csv"))

    train_val_dic = dict()
    total_open_count = 0
    total_closed_count = 0

    for i in range(len(df_all)):
        key = re.sub(r'_\d+\.jpg$', '', df_all["filename"][i])
        if key in test: continue

        if key not in train_val_dic:
            train_val_dic[key] = SampleInfo()

        attributes = get_attributes_from_string(df_all["region_attributes"][i])

        if(attributes[0]): total_open_count += 1
        else: total_closed_count += 1
        update_sample_properties(train_val_dic[key], attributes, df_all["filename"][i])


    q1_samples = dict()
    q2_samples = dict()
    q3_samples = dict()

    open_counts = [0, 0, 0]
    closed_counts = [0, 0, 0]

    for key in train_val_dic:
        quality_median = int(statistics.median(train_val_dic[key].qualities))
        if(quality_median == 1):
            q1_samples[key] = train_val_dic[key]
            open_counts[0] += q1_samples[key].open_count
            closed_counts[0] += q1_samples[key].closed_count
        elif(quality_median == 2):
            q2_samples[key] = train_val_dic[key]
            open_counts[1] += q2_samples[key].open_count
            closed_counts[1] += q2_samples[key].closed_count
        elif(quality_median >= 3):
            q3_samples[key] = train_val_dic[key]
            open_counts[2] += q3_samples[key].open_count
            closed_counts[2] += q3_samples[key].closed_count


    divide_train_val(q1_samples, int(open_counts[0] * 0.2), int(closed_counts[0] * 0.2))
    divide_train_val(q2_samples, int(open_counts[1] * 0.2), int(closed_counts[1] * 0.2))
    divide_train_val(q3_samples, int(open_counts[2] * 0.2), int(closed_counts[2] * 0.2))
        
    print(len(train_val_dic))

def split_dataset():
    delete_files_in_directory(train_labels_dir)
    delete_files_in_directory(test_labels_dir)
    delete_files_in_directory(val_labels_dir)
    delete_files_in_directory(train_images_dir)
    delete_files_in_directory(test_images_dir)
    delete_files_in_directory(val_images_dir)

    df_info = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "annotations", "info.csv"))

    for i in range(len(df_info)):
        all_data[df_info["file"][i]] = df_info["annotated"][i]

        pma = int(df_info["PMA"][i][:2])

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


    df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "annotations", "all.csv"))
    
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


    print('train set open={:d}, closed={:d}, q1={:d},q2={:d},q3={:d},q4={:d},q5={:d}, occlusion={:d}'.format(train_info.open_count, train_info.closed_count,
                                                                                              train_info.q1_count, train_info.q2_count, train_info.q3_count, train_info.q4_count, train_info.q5_count,
                                                                                              train_info.occlusion_count))
    print('val set open={:d}, closed={:d}, q1={:d},q2={:d},q3={:d},q4={:d},q5={:d}, occlusion={:d}'.format(val_info.open_count, val_info.closed_count,
                                                                                           val_info.q1_count, val_info.q2_count, val_info.q3_count, val_info.q4_count, val_info.q5_count,
                                                                                           val_info.occlusion_count))
    print('test set open={:d}, closed={:d}, q1={:d},q2={:d},q3={:d},q4={:d},q5={:d}, occlusion={:d}'.format(test_info.open_count, test_info.closed_count,
                                                                                            test_info.q1_count, test_info.q2_count, test_info.q3_count, test_info.q4_count, test_info.q5_count,
                                                                                            test_info.occlusion_count))

##########################################