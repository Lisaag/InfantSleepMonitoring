import os

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
                           frames_dir = os.path.join(data_dir, fragment_dir, eye_data_dir, eye_data_dir, "frames", fragment_dir)
                           print(frames_dir)
