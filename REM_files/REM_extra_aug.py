import settings
import os
import cv2
from scipy import ndimage
    
cropped_dir = os.path.join(os.path.abspath(os.getcwd()), "REM", "tmp", "cropped")
data_dir = os.path.join(os.path.abspath(os.getcwd()),"REM", "raw", "cropped")
for frame_centering in os.listdir(data_dir):
    for patient in os.listdir(os.path.join(data_dir, frame_centering)):
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
                for frame in os.listdir(sample_dir):
                    if frame[-4:] != ".jpg":
                        continue
                    image = cv2.imread(os.path.join(sample_dir, frame), cv2.IMREAD_GRAYSCALE)
                    image_rotated1 = ndimage.rotate(image, 10)
                    image_rotated2 = ndimage.rotate(image, -10)
                    image_flipped = cv2.flip(image, 0) 
                    print(os.path.join(cropped_dir, frame_centering, patient, eye_state, sample, frame[:-4]+"ROT1AUG.jpg"))
                    cv2.imwrite(os.path.join(cropped_dir, frame_centering, patient, eye_state, sample, frame[:-4]+"ROT1AUG.jpg"), image_rotated1)
                    cv2.imwrite(os.path.join(cropped_dir, frame_centering, patient, eye_state, sample, frame[:-4]+"ROT2AUG.jpg"), image_rotated2)
                    cv2.imwrite(os.path.join(cropped_dir, frame_centering, patient, eye_state, sample, frame[:-4]+"FLIPAUG.jpg"), image_flipped)
   