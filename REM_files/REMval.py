"""
This script is used to validate the REM model on the validation set.

Author: Lisa Groen
Date: May 9, 2025
"""

import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
from tensorflow.keras import models

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

#used for t-sne plot
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range

#get number from file name
def extract_number(filename):
    match = re.search(r'(\d+)(?=\.jpg$)', filename)
    return int(match.group(1)) if match else float('inf')

#load model saved as json
def load_model_json(path):
    with open(path, "r") as json_file:
        loaded_model_json = json_file.read()

    return models.model_from_json(loaded_model_json)

def plot_pr_curve(precision, recall, best_threshold, best_idx, path):
    """
    Plot pr curve over a range of sigmoid thresholds.

    Parameters:
    precision: list of precisions over range of threshold
    recall: list of recalls over range of threshold
    best_threshold: threshold with highest F1
    best_idx: idx with highest F1
    path: save path
    """
    sns.set_style("whitegrid")

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.')
    plt.scatter(recall[best_idx], precision[best_idx], s=50.0, color='red', label=f'Best threshold: {best_threshold:.2f}')

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(path,"prcurve.jpg"), format='jpg', dpi=500)  

def plot_tsne_both(model, path, samples, val_labels, train_labels):
    """
    Scatter plot using t-sne. This shows plot for both samples from training, and validation set

    Parameters:
    model: model, used to get the last dense layer, used to get the feature vectors that are used as tsne input
    path: save path
    samples: all samples to apply tsne to
    val_labels: class labels of validation set
    train_labels: class labels of train set
    """
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = model2(samples)

    sns.set_style("whitegrid", {'axes.grid' : False})
    tsne = TSNE(n_components=2, perplexity=25.0).fit_transform(features)

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    colors = ['#D9ECFF', '#FFD9D9', '#0000FF', '#FF0000']
    classes = ['-_t', 'REM_t', '-', 'REM']
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
    """
    Scatter plot using t-sne. This shows plot for validation samples, either for O/OR, C/CR, or -/REM

    Parameters:
    model: model, used to get the last dense layer, used to get the feature vectors that are used as tsne input
    path: save path
    val_samples_stacked: all samples to apply tsne to (all from validation set)
    true_labels: class labels of validation set
    """
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = model2(val_samples_stacked)

    sns.set_style("whitegrid", {'axes.grid' : False})
    tsne = TSNE(n_components=2, perplexity=25.0).fit_transform(features)

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    colors = ['blue', 'red']
    classes = ['-', 'REM']
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
    """
    Scatter plot using t-sne. This shows plot for all validation samples of the combined model, and color codes them by class O/OR/C/CR (see paper)

    Parameters:
    model: model, used to get the last dense layer, used to get the feature vectors that are used as tsne input
    path: save path
    val_samples_stacked: all samples to apply tsne to (all from validation set)
    all_labels: class labels of validation set, given as O/OR/C/CR (not only as -/REM)
    """
    mapping = {'O': 0, 'OR': 1, 'C': 2, 'CR': 3}
    all_labels = [mapping[element] for element in all_labels]

    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    features = model2(val_samples_stacked)

    sns.set_style("whitegrid", {'axes.grid' : False})
    tsne = TSNE(n_components=2, perplexity=25.0).fit_transform(features)

    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    colors = ['#02d1fa', '#faa302', '#026dfa', '#fa0202']
    
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

#Prepare validation data
def get_validation_data(fold):
    val_samples = list(); val_labels = list()
    train_samples = list(); train_labels = list()
    all_labels = list()

    for patient in os.listdir(settings.data_dir):
        patient_dir:str = os.path.join(settings.data_dir, patient)
        patient_id:str = patient[0:3]

        for eye_state in os.listdir(patient_dir):
            if(not settings.is_combined):
                if(settings.is_OREM and (eye_state == "C" or eye_state == "CR")): continue
                if(not settings.is_OREM and (eye_state == "O" or eye_state == "OR")): continue
            eye_state_dir = os.path.join(patient_dir, eye_state)
            for sample in os.listdir(eye_state_dir):
                if(sample[-3:] == "AUG"): continue
                sample_dir = os.path.join(eye_state_dir, sample)
                images = list()

                frames = glob.glob(os.path.join(sample_dir, "*.jpg"))
                sorted_frames = sorted(frames, key=extract_number)

                #get settings.frame_stack_count number of frames from the fragment, evenly spaced
                frame_indices = np.linspace(0, len(sorted_frames) - 1, settings.frame_stack_count, dtype=int).tolist()

                #some normalization steps
                for idx in frame_indices:
                    image = cv2.imread(os.path.join(sample_dir, sorted_frames[idx]), cv2.IMREAD_GRAYSCALE) 
                    image = cv2.resize(image, (settings.img_size, settings.img_size))
                    image = image / 255
                    images.append(image)
            
                expanded_stack = np.expand_dims(images, axis=-1) 
                stacked_images = np.stack(expanded_stack, axis=0)

                label = 0 if eye_state == "O" or eye_state == "C" else 1

                if(patient_id in settings.val_ids[fold]): 
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
    """
    run inference on test set, and get performance metrics

    Parameters:
    run: index of train run
    fold: index of fold (bc we use k-fold cross validation)
    path: save path

    Returns:
    accuracy, pr, rec, ap, auc, f1 performance metrics
    """
    model = load_model_json(os.path.join(path, settings.model_filename))
    model.load_weights(os.path.join(path, settings.checkpoint_filename))

    val_samples, true_labels, train_samples, train_labels, all_labels = get_validation_data(fold)

    predictions = model(val_samples, training=False)

    precision, recall, thresholds = precision_recall_curve(true_labels, predictions)
    f1_scores = (2 * precision * recall) / (precision + recall + 1e-9)

    best_idx = np.argmax(f1_scores[:-1])

    #best_threshold = thresholds[max(0, best_idx -1)]
    best_threshold = 0.5
    predicted_labels = [1 if x >= best_threshold else 0 for x in predictions]

    plot_pr_curve(precision, recall, best_threshold, best_idx, path)

    ap = average_precision_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, predictions)
    accuracy = accuracy_score(true_labels, predicted_labels) 
    pr = precision_score(true_labels, predicted_labels)
    rec = recall_score(true_labels, predicted_labels)
    f1 = (2 * pr * rec) / (pr + rec + 1e-9)

    with open(os.path.join(settings.results_dir, run, "metrics.csv"), "a") as file:
        file.write(f"{run},{fold},{accuracy},{pr},{rec},{ap},{auc},{f1}" + "\n")

    visualize_results(model, predicted_labels, true_labels, val_samples, path)
    plot_tsne_both(model, path, np.concatenate((val_samples, train_samples), axis=0), true_labels, train_labels)
    if(settings.is_combined): plot_tsne_all(model, path, val_samples, all_labels)
    plt.close('all')

    return accuracy, pr, rec, ap, auc, f1




with open(os.path.join(settings.results_dir, "metrics.csv"), "w") as file:
    file.write("run,m_accuracy,m_precision,m_recall,m_AUC,auc,mF1" + "\n")

#save performance metrics over multiple runs & multiple folds, so later the mean and stdv can be taken
all_APs = []
all_means = []
all_stds = []

#Get average metrics over train runs and over the 5 folds
for run in os.listdir(settings.results_dir):
    if(not run.isdigit()): continue
    with open(os.path.join(settings.results_dir, run, "metrics.csv"), "w") as file:
        file.write("run,fold,accuracy,precision,recall,AP,auc,F1" + "\n")
    metrics = []
   
    for fold in range(len(settings.val_ids)):
        metrics.append(validate_model(run, fold, os.path.join(settings.results_dir, run, str(fold))))

    metrics = np.array(metrics).T
    all_APs.append([metrics[3]])

    with open(os.path.join(settings.results_dir, "metrics.csv"), "a") as file:
        file.write(f'{run},{metrics[0]},{metrics[1]},{metrics[2]},{metrics[3]},{metrics[4]},{metrics[5]}' + "\n")

    all_means.append([statistics.mean(metrics[0]), statistics.mean(metrics[1]), statistics.mean(metrics[2]), statistics.mean(metrics[3]), statistics.mean(metrics[4]), statistics.mean(metrics[5])])
    all_stds.append([statistics.stdev(metrics[0]), statistics.stdev(metrics[1]), statistics.stdev(metrics[2]), statistics.stdev(metrics[3]), statistics.stdev(metrics[4]), statistics.stdev(metrics[5])])

all_means = np.array(all_means).T
all_stds = np.array(all_stds).T

#Write std between folds, std between train runs, and the total average metrics of all folds&trian runs
with open(os.path.join(settings.results_dir, "metrics.csv"), "a") as file:
    file.write(f'{"std/fold"},{statistics.mean(all_stds[0])},{statistics.mean(all_stds[1])},{statistics.mean(all_stds[2])},{statistics.mean(all_stds[3])},{statistics.mean(all_stds[4])},{statistics.mean(all_stds[5])}' + "\n")
    file.write(f'{"std/run"},{statistics.stdev(all_means[0])},{statistics.stdev(all_means[1])},{statistics.stdev(all_means[2])},{statistics.stdev(all_means[3])},{statistics.stdev(all_means[4])},{statistics.stdev(all_means[5])}' + "\n")
    file.write(f'{"mean/total"},{statistics.mean(all_means[0])},{statistics.mean(all_means[1])},{statistics.mean(all_means[2])},{statistics.mean(all_means[3])},{statistics.mean(all_means[4])},{statistics.mean(all_means[5])}' + "\n")


