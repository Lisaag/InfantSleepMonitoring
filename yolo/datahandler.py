import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataclasses import dataclass
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
        x=x-(w/2)
        y=y-(h/2)
        test_bb(df_all["filename"][i], x, y, w, h)

        if(not is_dummy):
            image = cv2.imread(os.path.join(os.path.abspath(os.getcwd()), all_images_dir, df_all["filename"][i]))
            height, width, _ = image.shape
            x/=width; w/=width; y/=height; h/=height
        write_label(df_all["filename"][i], all_labels_dir, x, y, w, h)


def test_bb(file_name, x, y, w, h):
    im_path = os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "images", file_name)
    image = cv2.imread(im_path, cv2.IMREAD_COLOR)

    #To draw a rectangle, you need top-left corner and bottom-right corner of rectangle.
    #cv2.rectangle(image, (x-(w/2), y-(h/2)), (x+(w/2), y+(h/2)), (0,255,0), 3)

    if not cv2.imwrite(os.path.join(os.path.abspath(os.getcwd()), 'dataset', 'SLAPI', 'vis', file_name), image):
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

def update_set_properties(set_info:SetInfo, attributes):
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

def split_dataset():
    delete_files_in_directory(train_labels_dir)
    delete_files_in_directory(test_labels_dir)
    delete_files_in_directory(val_labels_dir)

    df_info = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "annotations", "info.csv"))

    group1_count = 0
    group2_count = 0
    pmas =[]

    for i in range(len(df_info)):
        all_data[df_info["file"][i]] = df_info["annotated"][i]
        if(df_info["annotated"][i]):
            test[df_info["file"][i]] = df_info["annotated"][i]
            #TEMP
            val[df_info["file"][i]] = df_info["annotated"][i]

        else:
            pma = int(df_info["PMA"][i][:2])
            if(pma !=  0): pmas.append(pma)
            if (pma < 35): group1_count+=1
            else: group2_count+=1
            train[df_info["file"][i]] = df_info["annotated"][i]

    df_pma = {'pma': pmas}
    df = pd.DataFrame(df_pma)
    df['pma'].hist(bins=6)
    plt.title('PMA frequencies')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.grid(False)
    plt.show()
    print(group1_count)
    print(group2_count)

    df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), settings.slapi_dir, "raw", "annotations", "all.csv"))


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