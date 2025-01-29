import os
import shutil
import cv2
import ast

def is_jpg(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension.lower() == '.jpg'

def delete_contents(directory):
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        # Remove files
        if os.path.isfile(item_path) or os.path.islink(item_path):
            try:
                os.unlink(item_path) 
                print(f"Deleted file: {item_path}")
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")

        # Recursively remove directories
        elif os.path.isdir(item_path):
            try:
                shutil.rmtree(item_path) 
                print(f"Deleted directory: {item_path}")
            except Exception as e:
                print(f"Failed to delete {item_path}: {e}")


def split_REM_set(val_patients:list):
    root_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags")
    REM_dataset_dir = os.path.join(os.path.abspath(os.getcwd()),"OREM-dataset")
    delete_contents(REM_dataset_dir)
    
    for patient in os.listdir(root_dir):
        patient_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags", patient)

        set_split = "train"
        patient_nr = int(patient[0:3])
        if patient_nr in val_patients:
            set_split = "val"

        for eye_state_dir in os.listdir(patient_dir):
            if(eye_state_dir == "O" or eye_state_dir == "OR"):
                data_dir =  os.path.join(root_dir, patient_dir, eye_state_dir, "data")
                if not os.path.exists(data_dir):
                    print(f'{data_dir} DOES NOT EXIST')
                    continue
                for fragment_dir in os.listdir(data_dir):
                    for eye_data_dir in os.listdir(os.path.join(data_dir, fragment_dir)):
                        if(os.path.isdir(os.path.join(data_dir, fragment_dir, eye_data_dir))):
                           frames_dir = os.path.join(data_dir, fragment_dir, eye_data_dir, "frames", fragment_dir)
                           for frame in os.listdir(frames_dir):
                                REM_dir = os.path.join(REM_dataset_dir, set_split, eye_state_dir, str(patient)+"-"+fragment_dir)
                                source_file = os.path.join(frames_dir, frame)
                                if not os.path.exists(REM_dir): os.makedirs(REM_dir)
                                destination_file = os.path.join(REM_dir, frame)
                                                                
                                if not is_jpg(source_file):
                                    print(f"{source_file} is not a JPG file.")
                                    continue

                                image = cv2.imread(source_file) 

                                resized_image = cv2.resize(image, dsize=(64, 64))

                                gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

                                cv2.imwrite(destination_file, gray_image) 


split_REM_set([554, 778])

