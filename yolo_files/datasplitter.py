
import cv2
from collections import defaultdict
import os
import pandas as pd
import re
import shutil

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
    is_filter = False #whether the dataset is filtered or not
    is_OC = True #whether eyes are labeled as open/closed or not

    train_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "train", "labels")
    train_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "train", "images")
    val_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "val", "labels")
    val_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type,"val", "images")
    test_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "test", "labels")
    test_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "test", "images")
    all_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "labels")
    all_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "images")

    delete_files_in_directory(train_labels_dir)
    delete_files_in_directory(test_labels_dir)
    delete_files_in_directory(val_labels_dir)
    delete_files_in_directory(train_images_dir)
    delete_files_in_directory(test_images_dir)
    delete_files_in_directory(val_images_dir)
    delete_files_in_directory(all_labels_dir)

    #dataframe consisting all annotation data of all frames
    df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "annotations", "aabb.csv"))

    train_ids = [137, 260, 416, 440, 524, 554, 614, 616, 701, 773, 777, 863, 867, 887, 901, 866, 704, 657, 778, 976]
    val_ids =  [4, 399, 875, 971, 43]
    test_ids = [228, 360, 417, 545, 663, 929]
    
    data_info = defaultdict(lambda:[[], []])

    for i in range(len(df_all)):
        attributes = get_attributes_from_string(df_all["region_attributes"][i])
        if attributes[2]: continue #side profiles are excluded

        #in case of no occlusions
        if(('none' in attributes[1]) or (len(attributes[1]) == 1 and attributes[1][0] == 'shadow')):
            data_info[df_all["filename"][i]][0].append(attributes[0])
        #else, in case of occlusions
        else:
            data_info[df_all["filename"][i]][1].append(attributes[0])

        #get bounding box information, and save as YOLO format as txt
        x, y, w, h = get_aabb_from_string(df_all["region_shape_attributes"][i])
        x=x+(w/2); y=y+(h/2)
        image = cv2.imread(os.path.join(os.path.abspath(os.getcwd()), all_images_dir, df_all["filename"][i]))
        height, width, _ = image.shape  
        x/=width; w/=width; y/=height; h/=height
        class_label = "0"
        if(attributes[0] and is_OC): class_label = "1"
        write_aabb_label(df_all["filename"][i], all_labels_dir, x, y, w, h, class_label)


    for key in data_info:
        match = re.search(r'frame_(?:CG_)?(.*)', key)
        patient_id = int(match.group(1)[0:3])
        label_file = re.sub(r'\.jpg$', '', key) + ".txt" #save label as txt, according to YOLO
        #copy all raw annotation data into new folders with corresponding bounding box text files, according to YOLO standards
        if(len(data_info[key][1]) == 0 or not is_filter):
            if(patient_id in test_ids): copy_files(all_images_dir, all_labels_dir, test_images_dir, test_labels_dir, image_filename=key, label_filename=label_file)
            elif(patient_id in val_ids): copy_files(all_images_dir, all_labels_dir, val_images_dir, val_labels_dir, image_filename=key, label_filename=label_file)
            elif(patient_id in train_ids): copy_files(all_images_dir, all_labels_dir, train_images_dir, train_labels_dir, image_filename=key, label_filename=label_file)



create_splits("open-closed")
