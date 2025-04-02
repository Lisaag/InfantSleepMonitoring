import settings
import os
import cv2
from scipy import ndimage
    
cropped_dir = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cropped")
data_dir = os.path.join(os.path.abspath(os.getcwd()),"REM", "raw", "cropped")
for frame_centering in os.listdir(data_dir):
    for patient in os.listdir(os.path.join(data_dir, frame_centering)):
        patient_dir:str = os.path.join(settings.data_dir, patient)
        patient_id:str = patient[0:3]
        # if(patient != '773_02-11-2022'):
        #     continue
        #if(settings.is_OREM and patient_id == '440'): continue
        print(patient_id)
        for eye_state in os.listdir(patient_dir):
            eye_state_dir = os.path.join(patient_dir, eye_state)
            count = 0
            for sample in os.listdir(eye_state_dir):
                count +=1

            print(f'{frame_centering}-{patient}-{eye_state}  :  {count}')
                # if(sample[-3:] == "AUG"): continue
                # sample_dir = os.path.join(eye_state_dir, sample)
                # for frame in os.listdir(sample_dir):
                #     if frame[-4:] != ".jpg":
                #         continue

                #     image = cv2.imread(os.path.join(sample_dir, frame))
                #     image_rotated1 = ndimage.rotate(image, 5)
                #     image_rotated2 = ndimage.rotate(image, -5)
                #     image_flipped = cv2.flip(image, 1) 

                #     rot1_path = os.path.join(cropped_dir, frame_centering, patient, eye_state, sample+'ROT1AUG')
                #     if not os.path.exists(rot1_path): os.makedirs(rot1_path)
                #     rot2_path = os.path.join(cropped_dir, frame_centering, patient, eye_state, sample+'ROT2AUG')
                #     if not os.path.exists(rot2_path): os.makedirs(rot2_path)
                #     flip_path = os.path.join(cropped_dir, frame_centering, patient, eye_state, sample+'FLIPAUG')
                #     if not os.path.exists(flip_path): os.makedirs(flip_path)
                #     cv2.imwrite(os.path.join(rot1_path, frame), image_rotated1)
                #     cv2.imwrite(os.path.join(rot2_path, frame), image_rotated2)
                #     cv2.imwrite(os.path.join(flip_path, frame), image_flipped)
   