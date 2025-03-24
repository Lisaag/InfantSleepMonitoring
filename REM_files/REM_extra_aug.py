import settings
import os
import cv2
import glob
import numpy as np
from scipy import ndimage
    
for patient in os.listdir(settings.data_dir):
    patient_dir:str = os.path.join(settings.data_dir, patient)
    patient_id:str = patient[0:3]
    if(patient != '773_02-11-2022'):
        continue
    #if(settings.is_OREM and patient_id == '440'): continue
    print(patient_id)
    for eye_state in os.listdir(patient_dir):
        eye_state_dir = os.path.join(patient_dir, eye_state)
        for sample in os.listdir(eye_state_dir):
            if(sample[-3:] == "AUG"): continue
            sample_dir = os.path.join(eye_state_dir, sample)
            images = list()
            frames = glob.glob(os.path.join(sample_dir, "*.jpg"))
            print(frames)


            #image_rotated1 = ndimage.rotate(image, 10)
            #image_rotated2 = ndimage.rotate(image, -10)
            #image_flipped = cv2.flip(image, 0) 

            # for idx in frame_indices:
            #     image = cv2.imread(os.path.join(sample_dir, sorted_frames[idx]), cv2.IMREAD_GRAYSCALE) 
