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

##All functions for csv string interpretation
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
        return None
    
def str_to_bool(s: str):
   return s.lower() == "true"
    
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
  
##All functions for YOLO label file creation
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

def write_label(file_name, dir_name, x, y, w, h):
    file_name = re.sub(r'\.jpg$', '', file_name)

    with open(os.path.join(os.path.abspath(os.getcwd()), dir_name, file_name + '.txt'), "a") as file:
        file.write("0 " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")

def create_yolo_labels(is_dummy:bool = False):
    df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "annotations", "all.csv"))
    delete_files_in_directory(all_labels_dir)

    for i in range(len(df_all)):
        x, y, w, h = get_aabb_from_string(df_all["region_shape_attributes"][i])
        x=x+(w/2)
        y=y+(h/2)
        test_bb(df_all["filename"][i], x, y, w, h)

        if(not is_dummy):
            image = cv2.imread(os.path.join(os.path.abspath(os.getcwd()), all_images_dir, df_all["filename"][i]))
            height, width, _ = image.shape
            x/=width; w/=width; y/=height; h/=height
        write_label(df_all["filename"][i], all_labels_dir, x, y, w, h)


def test_bb(file_name, x, y, w, h):
    im_path = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "vis", file_name)
    image = cv2.imread(im_path, cv2.IMREAD_COLOR)
    if(image is None): im_path = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "images", file_name)
    image = cv2.imread(im_path, cv2.IMREAD_COLOR)

    #To draw a rectangle, you need top-left corner and bottom-right corner of rectangle.
    cv2.rectangle(image, (int(x-(w/2)), int(y-(h/2))), (int(x+(w/2)), int(y+(h/2))), (0,255,0), 3)
    cv2.circle(image,(int(x-(w/2)), int(y-(h/2))), 10, (255,0,0), -1)
    if not cv2.imwrite(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", 'vis', file_name), image):
        print("imwrite failed")

##All functions for train/val/test splitting
def create_dummy_data(file_name, dir_name):
    file_name = re.sub(r'\.jpg$', '', file_name)

    with open(os.path.join(os.path.abspath(os.getcwd()), dir_name, file_name + '.jpg'), "w") as file:
        file.write("0 ")


all_data:dict = dict()
train:dict = dict()
test:dict = dict()
val:dict = dict()

train_info = SetInfo()
val_info = SetInfo()
test_info = SetInfo()


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

    if(open_value): sample_info.open_count += 1
    else: sample_info.closed_count += 1


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
        elif(pma < 32 or pma > 36):
            test[df_info["file"][i]] = 1
            # #TEMP
            # val[df_info["file"][i]] = df_info["annotated"][i]
        # else:
        #     if (pma != 0): pmas.append(pma)

        #     train[df_info["file"][i]] = df_info["annotated"][i]

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

    open_val = int(total_open_count * 0.2)
    open_train = total_open_count - open_val
    closed_val = int(total_closed_count * 0.2)
    closed_train = total_closed_count - closed_val

    val_open_count = 0
    val_closed_count = 0

    for key in train_val_dic:
        if(val_open_count < open_val and train_val_dic[key].open_count != 0):
            val[key] = 1
            val_open_count += train_val_dic[key].open_count
            val_closed_count += train_val_dic[key].closed_count
        elif(val_closed_count < closed_val and train_val_dic[key].closed_count != 0):
            val[key] = 1
            val_open_count += train_val_dic[key].open_count
            val_closed_count += train_val_dic[key].closed_count
        else:
            train[key] = 1
        
    print(len(train_val_dic))
    print('train={:d}, val={:d}, test={:d}'.format(len(train), len(val), len(test)))
        



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

    for i in range(len(df_all)):
        attributes = get_attributes_from_string(df_all["region_attributes"][i])
        copy_to_split(df_all["filename"][i], attributes)


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