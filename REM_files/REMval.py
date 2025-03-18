import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from keras import backend as K

import tensorflow.keras as keras
import csv
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

import settings

def scale_to_01_range(x):

    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def load_model_json(path):
    with open(path, "r") as json_file:
        loaded_model_json = json_file.read()

    return models.model_from_json(loaded_model_json)

def get_validation_data():
    val_samples = list(); val_labels = list()

    for patient in os.listdir(settings.data_dir):
        patient_dir:str = os.path.join(settings.data_dir, patient)
        patient_id:str = patient[0:3]
        print(patient_id)
        for eye_state in os.listdir(patient_dir):
            if(settings.is_OREM and (eye_state == "C" or eye_state == "CR")): continue
            if(not settings.is_OREM and (eye_state == "O" or eye_state == "OR")): continue
            eye_state_dir = os.path.join(patient_dir, eye_state)
            for sample in os.listdir(eye_state_dir):
                if(patient_id not in settings.val_ids): continue
                if(patient_id in settings.val_ids and sample[-3:] == "AUG"): continue
                sample_dir = os.path.join(eye_state_dir, sample)
                images = list()
                for frame in os.listdir(sample_dir):
                    if frame.endswith(".jpg"):
                        image = cv2.imread(os.path.join(sample_dir, frame), cv2.IMREAD_GRAYSCALE) 
                        image = cv2.resize(image, (64, 64))
                        image = image / 255
                        images.append(image)
            
                expanded_stack = np.expand_dims(images, axis=-1) 
                stacked_images = np.stack(expanded_stack, axis=0)

                label = 0 if eye_state == "O" or eye_state == "C" else 1

                if(patient_id in settings.val_ids): 
                    print(f'from {patient_id} add to val')
                    val_samples.append(stacked_images)
                    val_labels.append(label)

    val_samples_stacked = np.stack(val_samples, axis=0)

    return val_samples_stacked, val_labels


model = load_model_json(settings.model_filepath)
model.load_weights(settings.checkpoint_filepath)


val_samples_stacked, val_labels = get_validation_data()

predictions = model.predict(val_samples_stacked)
predicted_labels = np.argmax(predictions, axis=1)

print(predicted_labels)

with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "predictions.txt"), 'w') as file:
    for label in predicted_labels:
        file.write(f"{label}\n")
with open(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "true_labels.txt"), 'w') as file:
    for label in val_labels:
        file.write(f"{label}\n")


model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
features = model2(val_samples_stacked)
tsne = TSNE(n_components=2).fit_transform(features)

tx = tsne[:, 0]
ty = tsne[:, 1]

tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)

colors = ['red', 'blue']
classes = ['O', 'OR'] if settings.is_OREM else ['C', 'CR']

# fig = plt.figure()
# ax = fig.add_subplot(111)
for idx, c in enumerate(colors):
    indices = [i for i, l in enumerate(predicted_labels) if idx == l]
    print(f'{classes[idx]} - {indices}')
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)
    plt.scatter(current_tx, current_ty, c=c, label=classes[idx])

plt.legend(loc='best')
plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"REM-results", "tsne.jpg"), format='jpg')  



