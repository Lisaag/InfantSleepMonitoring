import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from keras import backend as K

import tensorflow.keras as keras
import csv
import numpy as np
import cv2
import matplotlib.pyplot as plt
import glob
import re

from sklearn.manifold import TSNE

import settings

import REMmodelvis

from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, precision_recall_curve, average_precision_score

import statistics

def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

def extract_number(filename):
    match = re.search(r'(\d+)(?=\.jpg$)', filename)
    return int(match.group(1)) if match else float('inf')

def load_model_json(path):
    with open(path, "r") as json_file:
        loaded_model_json = json_file.read()

    return models.model_from_json(loaded_model_json)

def plot_tsne(model, path, val_samples_stacked, true_labels):
    #print('OUTPUT -4')
    #print(model.layers[-4].output)
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-4].output)
    features = model2(val_samples_stacked)
    #for f in features: print(f)
    #print('OUTPUT -1')
    #print(model.layers[-1].output)
    model3 = tf.keras.Model(inputs=model.input, outputs=model.layers[-1].output)
    features2 = model3(val_samples_stacked)
    #for f in features2: print(f)

    tsne = TSNE(n_components=2, perplexity=25.0).fit_transform(features)

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    colors = ['red', 'blue']
    classes = ['O', 'OR'] if settings.is_OREM else ['C', 'CR']
    
    plt.figure()
    for idx, c in enumerate(colors):
        indices = [i for i, l in enumerate(true_labels) if idx == l]
        print(f'{classes[idx]} - {indices}')
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        plt.scatter(current_tx, current_ty, c=c, label=classes[idx])

    plt.legend(loc='best')
    plt.savefig(os.path.join(path,"tsne.jpg"), format='jpg')  

def visualize_results(model, predicted_labels, true_labels, val_samples, path):
    with open(os.path.join(path, "predictions.txt"), 'w') as file:
        for label in predicted_labels:
            file.write(f"{label}\n")
    with open(os.path.join(path, "true_labels.txt"), 'w') as file:
        for label in true_labels:
            file.write(f"{label}\n")

    REMmodelvis.plot_confusion_matrix(path, true_labels, predicted_labels)
    plot_tsne(model, path, val_samples, true_labels)


def get_validation_data(fold):
    val_samples = list(); val_labels = list()

    for patient in os.listdir(settings.data_dir):
        patient_dir:str = os.path.join(settings.data_dir, patient)
        patient_id:str = patient[0:3]
        if(settings.is_OREM and patient_id == '440'): continue
        print(patient_id)
        for eye_state in os.listdir(patient_dir):
            if(settings.is_OREM and (eye_state == "C" or eye_state == "CR")): continue
            if(not settings.is_OREM and (eye_state == "O" or eye_state == "OR")): continue
            eye_state_dir = os.path.join(patient_dir, eye_state)
            for sample in os.listdir(eye_state_dir):
                if(patient_id not in settings.val_ids[fold]): continue
                if(patient_id in settings.val_ids[fold] and sample[-3:] == "AUG"): continue
                sample_dir = os.path.join(eye_state_dir, sample)
                images = list()

                frames = glob.glob(os.path.join(sample_dir, "*.jpg"))
                sorted_frames = sorted(frames, key=extract_number)

                frame_indices = np.linspace(0, len(sorted_frames) - 1, settings.frame_stack_count, dtype=int).tolist()

                for idx in frame_indices:
                    image = cv2.imread(os.path.join(sample_dir, sorted_frames[idx]), cv2.IMREAD_GRAYSCALE) 
                    image = cv2.resize(image, (64, 64))
                    image = image / 255
                    images.append(image)
            
                expanded_stack = np.expand_dims(images, axis=-1) 
                stacked_images = np.stack(expanded_stack, axis=0)

                label = 0 if eye_state == "O" or eye_state == "C" else 1

                if(patient_id in settings.val_ids[fold]): 
                    print(f'from {patient_id} add to val')
                    val_samples.append(stacked_images)
                    val_labels.append(label)

    val_samples_stacked = np.stack(val_samples, axis=0)

    return val_samples_stacked, val_labels

def validate_model(run, fold, path):
    print(path)
    model = load_model_json(os.path.join(path, settings.model_filename))
    model.load_weights(os.path.join(path, settings.checkpoint_filename))

    val_samples, true_labels = get_validation_data(fold)

    predictions = model(val_samples)

    precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)

    best_idx = np.argmax(precision[:-1])

    print(f'{len(precision)}, {len(thresholds)}')

    best_threshold = thresholds[best_idx]
    predicted_labels = [1 if x > best_threshold else 0 for x in predictions]

    ap = average_precision_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    with open(os.path.join(settings.results_dir, run, "metrics.csv"), "a") as file:
        file.write(f"{run},{fold},{accuracy},{precision},{recall},{ap}" + "\n")

    visualize_results(model, predicted_labels, true_labels, val_samples, path)

    return accuracy, precision, recall, ap


with open(os.path.join(settings.results_dir, "metrics.csv"), "w") as file:
    file.write("run,m_accuracy,m_precision,m_recall,m_AUC,sd_accuracy,sd_precision,sd_recall,sd_AUC" + "\n")

for run in os.listdir(settings.results_dir):
    if(not run.isdigit()): continue
    with open(os.path.join(settings.results_dir, run, "metrics.csv"), "w") as file:
        file.write("run,fold,accuracy,precision,recall,AP" + "\n")
    metrics = []
   
    for fold in range(len(settings.val_ids)):
        metrics.append(validate_model(run, fold, os.path.join(settings.results_dir, run, str(fold))))
   
    metrics = np.array(metrics).T
    with open(os.path.join(settings.results_dir, "metrics.csv"), "a") as file:
        file.write(f'{run},{statistics.mean(metrics[0])},{statistics.mean(metrics[1])},{statistics.mean(metrics[2])},{statistics.mean(metrics[3])},{statistics.stdev(metrics[0])},{statistics.stdev(metrics[1])},{statistics.stdev(metrics[2])},{statistics.stdev(metrics[3])}' + "\n")

