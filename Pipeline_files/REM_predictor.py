import os
import numpy as np
import settings
import cv2

os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from keras import backend as K

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

def get_all_samples(current_batch):
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
    fragment_count = get_last_index(save_path)
    print(f'{fragment_count+1} fragments detected from {save_path}')
    current_batch = 0

    if not os.path.exists(settings.predictions_path): os.makedirs(settings.predictions_path)
    with open(os.path.join(settings.predictions_path, "predictions.csv"), "w") as file:
        file.write("idx;predictions" + "\n")

    while current_batch*processing_batch_size < fragment_count:
        all_samples, indices = get_all_samples(current_batch)

        if len(all_samples) == 0:
            current_batch += 1
            continue

        model = load_model_json(os.path.join(settings.model_path, settings.model_filename))
        model.load_weights(os.path.join(settings.model_path, settings.checkpoint_filename))
        predictions = model(all_samples, training=False)
        predictions = predictions.numpy().flatten().tolist()
        
        print(f'Processed minute {current_batch}')
        for i, idx in enumerate(indices):
            with open(os.path.join(settings.predictions_path, "predictions.csv"), "a") as file:
                file.write(str(idx)+";"+str(predictions[i]) + "\n")

        current_batch += 1


run_inference()



            
    
    