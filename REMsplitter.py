import os
import shutil
import cv2

def split_REM_set():
    root_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags")
    for patient in os.listdir(root_dir):
        patient_dir:str = os.path.join(os.path.abspath(os.getcwd()), "frags", patient)
        for eye_state_dir in os.listdir(patient_dir):
            if(eye_state_dir == "C" or eye_state_dir == "CR"):
                data_dir =  os.path.join(root_dir, patient_dir, eye_state_dir, "data")
                if not os.path.exists(data_dir):
                    print(f'{data_dir} DOES NOT EXIST')
                    continue
                for fragment_dir in os.listdir(data_dir):
                    for eye_data_dir in os.listdir(os.path.join(data_dir, fragment_dir)):
                        if(os.path.isdir(os.path.join(data_dir, fragment_dir, eye_data_dir))):
                           frames_dir = os.path.join(data_dir, fragment_dir, eye_data_dir, "frames", fragment_dir)
                           for frame in os.listdir(frames_dir):
                                REM_dir = os.path.join(os.path.abspath(os.getcwd()),"REM-dataset", eye_state_dir, str(patient)+"-"+fragment_dir)
                                source_file = os.path.join(frames_dir, frame)
                                if not os.path.exists(REM_dir): os.makedirs(REM_dir)
                                destination_file = os.path.join(REM_dir, frame)
                                                                # Load the image
                                image = cv2.imread(source_file)  # Replace with the path to your image
                                print(image)
                                # Resize the image to 64x64
                                resized_image = cv2.resize(image, dsize=(64, 64))

                                # Save or display the resized image
                                cv2.imwrite(destination_file, resized_image)  # Save the resized image

