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
import seaborn as sns

import pandas as pd

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

def plot_pr_curve(precision, recall, best_threshold, best_idx, path):
    best_f1 = (2 * precision[best_idx] * recall[best_idx]) / (precision[best_idx] + recall[best_idx] + 1e-9)

    sns.set_style("whitegrid")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.scatter(recall[best_idx], precision[best_idx], s=50.0, color='red', label=f'Best threshold: {best_threshold:.2f}')

    # Labels and title
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(path,"prcurve.jpg"), format='jpg', dpi=500)  

def plot_tsne_both(model, path, samples, val_labels, train_labels):
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = model2(samples)

    sns.set_style("whitegrid", {'axes.grid' : False})
    tsne = TSNE(n_components=2, perplexity=25.0).fit_transform(features)

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    colors = ['#FFD9D9', '#D9ECFF', '#FF0000', '#0000FF']
    classes = ['-_t', 'R_t', '-', 'R']
    if(not settings.is_combined):
        classes = ['O_t', 'OR_t', 'O', 'OR'] if settings.is_OREM else ['C_t', 'CR_t', 'C', 'CR']

    val_labels = [2 if x == 0 else 3 for x in val_labels]
    all_labels = val_labels+train_labels
    
    plt.figure()
    for idx, c in enumerate(colors):
        indices = [i for i, l in enumerate(all_labels) if idx == l]
        print(f'{classes[idx]} - {indices}')
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        alpha = 0.4 if idx < 2 else 1
        plt.scatter(current_tx, current_ty, alpha=alpha, s=35.0, c=c, label=classes[idx])

    plt.legend(loc='best')
    plt.savefig(os.path.join(path,"tsne_both.jpg"), format='jpg', dpi=500)  

def plot_tsne(model, path, val_samples_stacked, true_labels):
    print('OUTPUT -2')
    print(model.layers[-2])
    #print(model.layers[-4].output)
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = model2(val_samples_stacked)

    sns.set_style("whitegrid", {'axes.grid' : False})
    tsne = TSNE(n_components=2, perplexity=25.0).fit_transform(features)

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    colors = ['red', 'blue']
    classes = ['-', 'R']
    if(not settings.is_combined):
        classes = ['O', 'OR'] if settings.is_OREM else ['C', 'CR']
    
    plt.figure()
    for idx, c in enumerate(colors):
        indices = [i for i, l in enumerate(true_labels) if idx == l]
        print(f'{classes[idx]} - {indices}')
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        plt.scatter(current_tx, current_ty, c=c, s=35.0, label=classes[idx])

    plt.legend(loc='best')
    plt.savefig(os.path.join(path,"tsne.jpg"), format='jpg', dpi=500)  

def plot_tsne_all(model, path, val_samples_stacked, all_labels):
    # Mapping dictionary
    mapping = {'O': 0, 'OR': 1, 'C': 2, 'CR': 3}

    # Replace using the mapping
    all_labels = [mapping[element] for element in all_labels]

    print('OUTPUT -2')
    print(model.layers[-2])
    #print(model.layers[-4].output)
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = model2(val_samples_stacked)

    sns.set_style("whitegrid", {'axes.grid' : False})
    tsne = TSNE(n_components=2, perplexity=25.0).fit_transform(features)

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    colors = ['red', 'blue', 'orange', 'green']
    
    classes = ['O', 'OR', 'C', 'CR']
    
    plt.figure()
    for idx, c in enumerate(colors):
        indices = [i for i, l in enumerate(all_labels) if idx == l]
        print(f'{classes[idx]} - {indices}')
        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)
        plt.scatter(current_tx, current_ty, c=c, s=35.0, label=classes[idx])

    plt.legend(loc='best')
    plt.savefig(os.path.join(path,"tsne_all.jpg"), format='jpg', dpi=500) 


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
    train_samples = list(); train_labels = list()
    all_labels = list()

    for patient in os.listdir(settings.data_dir):
        patient_dir:str = os.path.join(settings.data_dir, patient)
        patient_id:str = patient[0:3]
        if(patient_id == '440'): continue
        print(patient_id)
        for eye_state in os.listdir(patient_dir):
            if(not settings.is_combined):
                if(settings.is_OREM and (eye_state == "C" or eye_state == "CR")): continue
                if(not settings.is_OREM and (eye_state == "O" or eye_state == "OR")): continue
            eye_state_dir = os.path.join(patient_dir, eye_state)
            for sample in os.listdir(eye_state_dir):
                #if(patient_id not in settings.val_ids[fold]): continue
                if(sample[-3:] == "AUG"): continue
                sample_dir = os.path.join(eye_state_dir, sample)
                images = list()

                frames = glob.glob(os.path.join(sample_dir, "*.jpg"))
                sorted_frames = sorted(frames, key=extract_number)

                frame_indices = np.linspace(0, len(sorted_frames) - 1, settings.frame_stack_count, dtype=int).tolist()

                for idx in frame_indices:
                    image = cv2.imread(os.path.join(sample_dir, sorted_frames[idx]), cv2.IMREAD_GRAYSCALE) 
                    image = cv2.resize(image, (settings.img_size, settings.img_size))
                    image = image / 255
                    images.append(image)
            
                expanded_stack = np.expand_dims(images, axis=-1) 
                stacked_images = np.stack(expanded_stack, axis=0)

                label = 0 if eye_state == "O" or eye_state == "C" else 1

                if(patient_id in settings.val_ids[fold]): 
                    print(f'from {patient_id} add to val')
                    val_samples.append(stacked_images)
                    val_labels.append(label)
                    all_labels.append(eye_state)

                else:
                    train_samples.append(stacked_images)
                    train_labels.append(label)
                


                


    val_samples_stacked = np.stack(val_samples, axis=0)
    train_samples_stacked = np.stack(train_samples, axis=0)

    return val_samples_stacked, val_labels, train_samples_stacked, train_labels, all_labels

def validate_model(run, fold, path):
    print(path)
    model = load_model_json(os.path.join(path, settings.model_filename))
    model.load_weights(os.path.join(path, settings.checkpoint_filename))

    val_samples, true_labels, train_samples, train_labels, all_labels = get_validation_data(fold)

    predictions = model(val_samples, training=False)

    precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)

    best_idx = np.argmax(f1_scores[:-1])

    best_threshold = thresholds[max(0, best_idx -1)]
    predicted_labels = [1 if x > best_threshold else 0 for x in predictions]

    plot_pr_curve(precision, recall, best_threshold, best_idx, path)

    ap = average_precision_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels)
    recall = recall_score(true_labels, predicted_labels)

    with open(os.path.join(settings.results_dir, run, "metrics.csv"), "a") as file:
        file.write(f"{run},{fold},{accuracy},{precision},{recall},{ap},{auc}" + "\n")

    visualize_results(model, predicted_labels, true_labels, val_samples, path)
    plot_tsne_both(model, path, np.concatenate((val_samples, train_samples), axis=0), true_labels, train_labels)
    if(settings.is_combined): plot_tsne_all(model, path, val_samples, all_labels)
    plt.close('all')

    return accuracy, precision, recall, ap, auc


with open(os.path.join(settings.results_dir, "metrics.csv"), "w") as file:
    #file.write("run,m_accuracy,m_precision,m_recall,m_AUC,sd_accuracy,sd_precision,sd_recall,sd_AUC" + "\n")
    file.write("run,m_accuracy,m_precision,m_recall,m_AUC,auc" + "\n")



all_APs = []

for run in os.listdir(settings.results_dir):
    if(not run.isdigit()): continue
    with open(os.path.join(settings.results_dir, run, "metrics.csv"), "w") as file:
        file.write("run,fold,accuracy,precision,recall,AP,auc" + "\n")
    metrics = []
   
    for fold in range(len(settings.val_ids)):
        metrics.append(validate_model(run, fold, os.path.join(settings.results_dir, run, str(fold))))

   
    metrics = np.array(metrics).T
    all_APs.append([metrics[3]])

    with open(os.path.join(settings.results_dir, "metrics.csv"), "a") as file:
        file.write(f'{run},{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]},{metrics[4]}' + "\n")



#make_boxplot(all_APs, os.path.join(settings.results_dir,run,"box.jpg"))


        #file.write(f'{run},{statistics.mean(metrics[0])},{statistics.mean(metrics[1])},{statistics.mean(metrics[2])},{statistics.mean(metrics[3])},{statistics.stdev(metrics[0])},{statistics.stdev(metrics[1])},{statistics.stdev(metrics[2])},{statistics.stdev(metrics[3])}' + "\n")

