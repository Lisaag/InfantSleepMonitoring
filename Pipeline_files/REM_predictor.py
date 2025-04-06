import os
import numpy as np
import settings
import cv2
import pandas as pd

os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from keras import backend as K

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
    images = []

    if not os.path.exists(os.path.join(save_path, str(fragment))):
        print(f"NO FRAGMENT AT INDEX {fragment}, {os.path.join(save_path, str(fragment))}")
        return []
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

    return [stacked_images], open_count
    

def run_inference():
    fragment_count = get_last_index(save_path)
    print(f'{fragment_count+1} fragments detected from {save_path}')

    frags_df = pd.read_csv(os.path.join(settings.eye_frag_path, settings.cur_vid[:-4], "info.csv"), delimiter=';')

    if not os.path.exists(settings.predictions_path): os.makedirs(settings.predictions_path)
    with open(os.path.join(settings.predictions_path, "predictions_2.csv"), "w") as file:
        file.write("idx;predictions;class" + "\n")

    for i in range(fragment_count + 1):
        sample, open_count = get_sample(i, frags_df)

        if len(sample) == 0:
            continue

        if(open_count >= 3):
            print("OPEN")
            model = load_model_json(os.path.join(settings.model_path, 'open', settings.model_filename))
            model.load_weights(os.path.join(settings.model_path, 'open', settings.checkpoint_filename))
        else:
            print("CLOSED")
            model = load_model_json(os.path.join(settings.model_path, 'closed', settings.model_filename))
            model.load_weights(os.path.join(settings.model_path, 'closed', settings.checkpoint_filename))
            
        prediction = model(sample, training=False)
        prediction = prediction.numpy().flatten().tolist()
        
        with open(os.path.join(settings.predictions_path, "predictions.csv"), "a") as file:
            file.write(str(i)+";"+str(prediction),str("O"if open_count >= 3 else "C") + "\n")


run_inference()



            
    
    