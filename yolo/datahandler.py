import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import shutil

import pandas as pd
import settings

#directories
train_labels_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "train", "labels")
train_images_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "train", "images")
val_labels_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "val", "labels")
val_images_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "val", "images")
test_labels_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "test", "labels")
test_images_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "test", "images")
all_labels_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "labels")
all_images_dir = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "images")


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

def create_yolo_labels():
    df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "annotations", "all.csv"))
    delete_files_in_directory(all_labels_dir)

    for i in range(len(df_all)):
        x, y, w, h = get_aabb_from_string(df_all["region_shape_attributes"][i])
        write_label(df_all["filename"][i], all_labels_dir, x, y, w, h)


##All functions for train/val/test splitting
def create_dummy_data(file_name, dir_name):
    file_name = re.sub(r'\.jpg$', '', file_name)

    with open(os.path.join(os.path.abspath(os.getcwd()), dir_name, file_name + '.jpg'), "w") as file:
        file.write("0 ")

def copy_to_split(file_name:str, train_split:dict, val_split:dict, test_split:dict):
    label_file = re.sub(r'\.jpg$', '', file_name) + ".txt"
    key = re.sub(r'_\d+\.jpg$', '', file_name)

    if(train_split.get(key) != None):
        shutil.copy(os.path.join(all_images_dir, file_name), train_images_dir)
        shutil.copy(os.path.join(all_labels_dir, label_file), train_labels_dir)
    if(test_split.get(key) != None):
        shutil.copy(os.path.join(all_images_dir, file_name), test_images_dir)
        shutil.copy(os.path.join(all_labels_dir, label_file), test_labels_dir)
    if(val_split.get(key) != None):
        shutil.copy(os.path.join(all_images_dir, file_name), val_images_dir)
        shutil.copy(os.path.join(all_labels_dir, label_file), val_labels_dir)

def split_dataset():
    delete_files_in_directory(train_labels_dir)
    delete_files_in_directory(test_labels_dir)
    delete_files_in_directory(val_labels_dir)

    df_info = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "annotations", "info.csv"))

    all_data:dict = dict()
    train:dict = dict()
    test:dict = dict()
    val:dict = dict()

    for i in range(len(df_info)):
        all_data[df_info["file"][i]] = df_info["annotated"][i]
        if(df_info["annotated"][i]):
            test[df_info["file"][i]] = df_info["annotated"][i]
            #TEMP
            val[df_info["file"][i]] = df_info["annotated"][i]

        else:
            train[df_info["file"][i]] = df_info["annotated"][i]


    df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "annotations", "all.csv"))

    for i in range(len(df_all)):
        #get_attributes_from_string(df_all["region_attributes"][i])
        copy_to_split(df_all["filename"][i], train, val, test)

##########################################
