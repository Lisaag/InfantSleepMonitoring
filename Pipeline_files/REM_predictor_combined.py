"""
This script is used to predict REM in full-length video's.
Output = a csv file containing the REM predictions over the whole video.
This script uses a single model for open REM and closed REM, unlike REM_predictor.py, which uses a 2 separate models (see paper).

Author: Lisa Groen
Date: April 30, 2025
"""

import numpy as np
import os

import cv2
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow.keras import models

import settings

save_path = os.path.join(settings.eye_frag_path, settings.cur_vid[:-4])
processing_batch_size = 40 #save 1 minute at a time

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


def get_all_samples(current_batch:int):
    """
    Preprocess stack of images for each 1.5 second fragment of a batch, to be used as input for the REM model

    Parameters:
    - current_batch: index of batch being processed
    Returns:
    stack of n images of each fragment of a batch, and corresponding fragment indices
    """
    all_samples = []
    indices = []
    for fragment in range(current_batch*processing_batch_size, current_batch*processing_batch_size+processing_batch_size):
        images = []
        if not os.path.exists(os.path.join(save_path, str(fragment))):
            print(f"NO FRAGMENT AT INDEX {fragment}, {os.path.join(save_path, str(fragment))}")
            continue
        for i in range(settings.frame_stack_count):
            image = cv2.imread(os.path.join(save_path, str(fragment), str(i)+".jpg"), cv2.IMREAD_GRAYSCALE) 
            image = cv2.resize(image, (settings.img_size, settings.img_size))
            image = image / 255
            images.append(image)

        expanded_stack = np.expand_dims(images, axis=-1) 
        stacked_images = np.stack(expanded_stack, axis=0)

        all_samples.append(stacked_images)

        indices.append(fragment)
    
    if(len(all_samples) == 0):
        return [], []
    return np.stack(all_samples, axis=0), indices

def run_inference():
    """
    Use the REM model to predict REM for all fragments of a full-length video.
    Processes the video per batch, where in our case the batch size is processing_batch_size (set to 40 to process minute by minute)
    """
    fragment_count = get_last_index(save_path) #fragment file names is index of fragment (e.g. 26.mp4), so the highest number file name equals number of fragments in a batch
    current_batch = 0 

    if not os.path.exists(os.path.join(settings.predictions_path, settings.cur_vid)): os.makedirs(os.path.join(settings.predictions_path, settings.cur_vid))
    with open(os.path.join(settings.predictions_path, settings.cur_vid, "predictions.csv"), "w") as file:
        file.write("idx;predictions" + "\n")

    while current_batch*processing_batch_size < fragment_count:
        #Preprocess data to give as input to the REM model
        all_samples, indices = get_all_samples(current_batch)

        #if no localized eyes in batch, continue to next batch
        if len(all_samples) == 0:
            current_batch += 1
            continue

        #Predict REM for all fragments in batch
        model = load_model_json(os.path.join(settings.model_path, settings.model_filename))
        model.load_weights(os.path.join(settings.model_path, settings.checkpoint_filename))
        predictions = model(all_samples, training=False)
        predictions = predictions.numpy().flatten().tolist()
        
        #Save all predictions of batch
        for i, idx in enumerate(indices):
            with open(os.path.join(settings.predictions_path, settings.cur_vid, "predictions.csv"), "a") as file:
                file.write(str(idx)+";"+str(predictions[i]) + "\n")

        print(f'Processed batch {current_batch}')
        current_batch += 1


run_inference()

