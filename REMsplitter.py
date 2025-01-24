import os
import shutil

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
                                REM_dir = os.path.join(os.path.abspath(os.getcwd()),"REM-dataset", eye_state_dir, str(patient_dir)+"-"+fragment_dir)
                                source_file = os.path.join(frames_dir, frame)
                                if not os.path.exists(REM_dir): os.makedirs(REM_dir)
                                destination_file = os.path.join(REM_dir, frame)
                                new_file_name:str = str(fragment_dir)
                                print(os.path.join(os.path.abspath(os.getcwd()),"REM-dataset", eye_state_dir, new_file_name))
                                print(destination_file)
                                # try:
                                #     shutil.copy(source_file, destination_file)
                                #     print(f"File copied successfully as {destination_file}")
                                # except FileNotFoundError:
                                #     print("Source file not found. Please check the file path.")
                                # except PermissionError:
                                #     print("Permission denied. Please check your permissions.")
                                # except Exception as e:
                                #     print(f"An error occurred: {e}")
