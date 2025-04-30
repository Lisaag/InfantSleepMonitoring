"""
This script is used to predict REM in full-length video's.
Output = a csv file containing the REM predictions over the whole video.
This script uses separate models for open REM and closed REM, unlike REM_predictor_combined.py, which uses a single model.

Author: Lisa Groen
Date: April 30, 2025
"""

import numpy as np
import os
import pandas as pd


import cv2
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from keras import backend as K

import settings

save_path = os.path.join(settings.eye_frag_path, settings.cur_vid[:-4])

def get_last_index(directory):
    existing_folders = []
    for dir in os.listdir(directory):
        if os.path.isdir(os.path.join(directory, dir)):
            if(dir.isdigit()):
                existing_folders.append(int(dir))
    
    return max(existing_folders, default=0)


def load_model_json(path):
    with open(path, "r") as json_file:
        loaded_model_json = json_file.read()

    return models.model_from_json(loaded_model_json)

def get_sample(fragment, frags_df):
    """
    Preprocess stack of images for 1.5 second fragment, to be used as input for the REM model

    Parameters:
    - fragment: fragment index
    - frags_df: dataframe consisting info on all fragments of full-length video
    Returns:
    stack of n images of fragment
    """
    images = []

    if not os.path.exists(os.path.join(save_path, str(fragment))):
        print(f"NO FRAGMENT AT INDEX {fragment}, {os.path.join(save_path, str(fragment))}")
        return [], 0
    for i in range(settings.frame_stack_count):
        image = cv2.imread(os.path.join(save_path, str(fragment), str(i)+".jpg"), cv2.IMREAD_GRAYSCALE) 
        image = cv2.resize(image, (settings.img_size, settings.img_size))
        image = image / 255
        images.append(image)

    expanded_stack = np.expand_dims(images, axis=-1) 
    stacked_images = np.stack(expanded_stack, axis=0)

    row =  frags_df[frags_df['idx'] == fragment]
    if row.empty:
        print(f'no fragment idx {fragment} found')
    open_count = row['open_count'].iloc[0]

    return np.stack([stacked_images], axis=0), open_count
    

def run_inference():
    """
    Use the REM model to predict REM for all fragments of a full-length video.
    Processes the video per fragment.
    """
    fragment_count = get_last_index(save_path)

    frags_df = pd.read_csv(os.path.join(settings.eye_frag_path, settings.cur_vid[:-4], "info.csv"), delimiter=';')

    if not os.path.exists(os.path.join(settings.predictions_path, settings.cur_vid)): os.makedirs(os.path.join(settings.predictions_path, settings.cur_vid))
    with open(os.path.join(settings.predictions_path, settings.cur_vid, "predictions_2.csv"), "w") as file:
        file.write("idx;predictions;class" + "\n")

    for i in range(fragment_count + 1):
        sample, open_count = get_sample(i, frags_df)

        if len(sample) == 0:
            continue

        if(open_count >= 3):
            model = load_model_json(os.path.join(settings.model_path, 'open', settings.model_filename))
            model.load_weights(os.path.join(settings.model_path, 'open', settings.checkpoint_filename))
        else:
            model = load_model_json(os.path.join(settings.model_path, 'closed', settings.model_filename))
            model.load_weights(os.path.join(settings.model_path, 'closed', settings.checkpoint_filename))
            
        prediction = model(sample, training=False)
        prediction = prediction.numpy().flatten().tolist()
        
        with open(os.path.join(settings.predictions_path, settings.cur_vid, "predictions_2.csv"), "a") as file:
            file.write(str(i)+";"+str(prediction[0])+";"+str("O"if open_count >= 3 else "C") + "\n")


run_inference()



            
    
    