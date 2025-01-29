import os
import shutil
import cv2
import matplotlib.pyplot as plt
import numpy as np

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


def split_REM_set():
    root_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags")
    REM_dataset_dir = os.path.join(os.path.abspath(os.getcwd()),"OREM-dataset")
    delete_contents(REM_dataset_dir)

    sizes = []
    
    for patient in os.listdir(root_dir):
        patient_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags", patient)

        patient_nr = int(patient[0:3])

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
                                source_file = os.path.join(frames_dir, frame)
                                                                
                                if not is_jpg(source_file):
                                    print(f"{source_file} is not a JPG file.")
                                    continue

                                image = cv2.imread(source_file) 

               
                                sizes.append(len(image))

    average = np.mean(sizes)
    median = np.median(sizes)

    # Create a histogram
    plt.hist(sizes, bins=8, edgecolor='black', alpha=0.75)

    # Add labels and title
    plt.title('Frequency of image region sizes')
    plt.xlabel('Size')
    plt.ylabel('Frequency')

    text = f"Average: {average:.2f}\nMedian: {median:.2f}"
    plt.text(0.95, 0.95, text, transform=plt.gca().transAxes, fontsize=10, verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.5))

    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"sizes.jpg"), format='jpg')   



                                

split_REM_set()

