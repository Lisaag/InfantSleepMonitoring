import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import re
import cv2
import pandas as pd
import settings
from collections import defaultdict

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

#write aabb label in YOLO format
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

def create_yolo_labels():
    global all_labels_dir
    all_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "labels", "aabb")
        #delete_files_in_directory(all_labels_dir)
        #delete_files_in_directory(vis_aabb_dir)
    df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "annotations", "aabb.csv"))
    stats = defaultdict(lambda: [0, 0, defaultdict(lambda: 0)])

    total_samples = 0
    total_samples_filtered = 0

    train_ids = []
    val_ids =  []
    test_ids = [228, 360, 417, 545, 663, 929]

    for i in range(len(df_all)):
        match = re.search(r'frame_(?:CG_)?(.*)', df_all["filename"][i])
        attributes = get_attributes_from_string(df_all["region_attributes"][i])
        if attributes[2]: continue
        
        patient_id = int(match.group(1)[0:3])
        if patient_id in test_ids: total_samples+=1

        if(attributes[0]):
            stats[patient_id][0] += 1
        else:
            stats[patient_id][1] += 1

        if('none' not in attributes[1]):
            for attribute in attributes[1]:
                stats[patient_id][2][attribute] += 1
        else:
            if patient_id in test_ids: total_samples_filtered+=1


    for key in stats:
        print(f'{key}:::  O: {stats[key][0]} - C: {stats[key][1]} -  {dict(stats[key][2])} ')

    print(f'TOTAL: {total_samples} TOTAL FILTERED: {total_samples_filtered}')

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