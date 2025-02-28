import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import re
import shutil
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

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

# def reduce_splits(train_split:Split, val_split:Split, test_split:Split, percentage:float):
#     percentage = min(percentage, 100) / 100 #cap to 100%

#     train_samples = train_split.open_samples[0:math.ceil(len(train_split.open_samples) * percentage)] + train_split.closed_samples[0:math.ceil(len(train_split.closed_samples) * percentage)]
#     val_samples = val_split.open_samples[0:math.ceil(len(val_split.open_samples) * percentage)] + val_split.closed_samples[0:math.ceil(len(val_split.closed_samples) * percentage)]
#     test_samples = test_split.open_samples[0:math.ceil(len(test_split.open_samples) * percentage)] + test_split.closed_samples[0:math.ceil(len(test_split.closed_samples) * percentage)]

#     print(f'reduced {percentage * 100}%: train: {len(train_samples)}, val: {len(val_samples)}, test: {len(test_samples)}, total: {len(train_samples)+len(val_samples)+len(test_samples)}')
#     print(f'FILECOUNT reduced {percentage * 100}%: train: {len(set(train_samples))}, val: {len(set(val_samples))}, test: {len(set(test_samples))}, total: {len(set(train_samples))+len(set(val_samples))+len(set(test_samples))}')

#     return(train_samples, val_samples, test_samples)

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


def plot_dataset_info(simple_train, simple_val, dif_train, dif_val, dif_test):
    ticks = [1, 2, 4, 5, 6]
    bar_names = ["train", "val", "train", "val", "test"]
    open = [lst[0] for lst in [simple_train, simple_val, dif_train, dif_val, dif_test]]
    closed = [lst[1] for lst in [simple_train, simple_val, dif_train, dif_val, dif_test]]

    sns.set_style("whitegrid")
    palette = sns.cubehelix_palette(start=.5, rot=-.5)

    plt.bar(ticks, open, color=palette[3], label='Open', width=1.0, edgecolor = '0.3')
    plt.bar(ticks, closed, bottom=open, color=palette[1], label='Closed', width=1.0, edgecolor = '0.3') 

    plt.xticks(ticks, bar_names)
    plt.gca().xaxis.grid(False)
    plt.legend(loc="upper right")

    group_divider = 3
    plt.axvline(x=group_divider, color='grey', linestyle='--', linewidth=0.8)
    
    plt.text(1.5, -120, "Simple dataset", ha='center', va='center', fontsize=10, color='0.3', weight=550)
    plt.text(5.0, -120, "Difficult dataset", ha='center', va='center', fontsize=10, color='0.3', weight=550)

    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"dataset.jpg"), dpi=500, format='jpg')   
   



#write aabb label in YOLO format
def write_aabb_label(file_name, dir_name, x, y, w, h, object_class):
    file_name = re.sub(r'\.jpg$', '', file_name)

    with open(os.path.join(os.path.abspath(os.getcwd()), dir_name, file_name + '.txt'), "a") as file:
        file.write(str(object_class) + " " + str(x) + " " + str(y) + " " + str(w) + " " + str(h) + "\n")

def create_splits(split_type):
    is_filter = False
    is_OC = True
    plot_info = False

    train_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "train", "labels")
    train_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "train", "images")
    val_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "val", "labels")
    val_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type,"val", "images")
    test_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "test", "labels")
    test_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", split_type, "test", "images")
    all_labels_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "labels")
    all_images_dir = os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "images")

    if(not plot_info):
        delete_files_in_directory(train_labels_dir)
        delete_files_in_directory(test_labels_dir)
        delete_files_in_directory(val_labels_dir)
        delete_files_in_directory(train_images_dir)
        delete_files_in_directory(test_images_dir)
        delete_files_in_directory(val_images_dir)
        delete_files_in_directory(all_labels_dir)

    df_all = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), "datasets", "SLAPI", "raw", "annotations", "aabb.csv"))

    train_ids = [137, 260, 416, 440, 524, 554, 614, 616, 701, 773, 777, 863, 867, 887, 901, 866, 704, 657, 778, 976]
    val_ids =  [4, 399, 875, 971, 43]
    test_ids = [228, 360, 417, 545, 663, 929]
    
    data_info = defaultdict(lambda:[[], []])

    s_train = [0, 0]; s_val = [0, 0]
    d_train = [0, 0]; d_val = [0, 0]; d_test = [0, 0]
    s_tr_p =set();s_va_p=set()
    d_tr_p =set();d_va_p=set();d_te_p=set()
    key_s_tr_p =set();key_s_va_p=set()
    key_d_tr_p =set();key_d_va_p=set();key_d_te_p=set()
    

    for i in range(len(df_all)):
        attributes = get_attributes_from_string(df_all["region_attributes"][i])
        if attributes[2]: continue

        if(('none' in attributes[1]) or (len(attributes[1]) == 1 and attributes[1][0] == 'shadow')):
            data_info[df_all["filename"][i]][0].append(attributes[0])
         
        else:
            data_info[df_all["filename"][i]][1].append(attributes[0])

        if(not plot_info):
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
        label_file = re.sub(r'\.jpg$', '', key) + ".txt"
        if(plot_info):
                for eye in data_info[key][0] + data_info[key][1]:
                    if(patient_id in train_ids):
                        d_tr_p.add(patient_id)
                        key_d_tr_p.add(key)
                        if(eye): d_train[0] +=1
                        else: d_train[1]+=1
                        if(len(data_info[key][1]) == 0):
                            s_tr_p.add(patient_id)
                            key_s_tr_p.add(key)
                            if(eye): s_train[0] +=1
                            else: s_train[1]+=1
                    elif(patient_id in val_ids):
                        d_va_p.add(patient_id)
                        key_d_va_p.add(key)
                        if(eye): d_val[0] +=1
                        else: d_val[1]+=1                        
                        if(len(data_info[key][1]) == 0):
                            s_va_p.add(patient_id)
                            key_s_va_p.add(key)
                            if(eye): s_val[0] +=1
                            else: s_val[1]+=1   
                    elif(patient_id in test_ids):
                        d_te_p.add(patient_id)
                        key_d_te_p.add(key)
                        if(eye): d_test[0] +=1
                        else: d_test[1]+=1                     
        else:
            if(len(data_info[key][1]) == 0 or not is_filter):
                if(patient_id in test_ids): copy_files(all_images_dir, all_labels_dir, test_images_dir, test_labels_dir, image_filename=key, label_filename=label_file)
                elif(patient_id in val_ids): copy_files(all_images_dir, all_labels_dir, val_images_dir, val_labels_dir, image_filename=key, label_filename=label_file)
                elif(patient_id in train_ids): copy_files(all_images_dir, all_labels_dir, train_images_dir, train_labels_dir, image_filename=key, label_filename=label_file)

                        
    if(plot_info):
        print(f'Simple set: train:{len(s_tr_p)}-{len(key_s_tr_p)}-{s_train[0]+s_train[1]}    val:{len(s_va_p)}-{len(key_s_va_p)}-{s_val[0]+s_val[1]}')
        print(f'Diff set: train:{len(d_tr_p)}-{len(key_d_tr_p)}-{d_train[0]+d_train[1]}    val:{len(d_va_p)}-{len(key_d_va_p)}-{d_val[0]+d_val[1]}     test:{len(d_te_p)}-{len(key_d_te_p)}-{d_test[0]+d_test[1]}')
        plot_dataset_info(s_train, s_val, d_train, d_val, d_test)


create_splits("OC")
