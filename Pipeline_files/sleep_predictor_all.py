"""
This script is used to predict final sleep state of each minute in a full length video.
Outputs predictions and metrics for each minute in a video.

Author: Lisa Groen
Date: April 30, 2025
"""

import ast
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import os
import pandas as pd
import seaborn as sns


import cv2
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix

import settings


max_movement_fraction = 0.5 #maximum position difference of eye between frames, as a fraction of the eye's bounding box

CREM_threshold = 0.7 #threshold of when fragment is classified as REM
OREM_threshold = 0.7#threshold of when fragment is classified as REM
 
REM_threshold = 0.7 #threshold of when fragment is classified as REM
O_threshold = 3 * (settings.fragment_length//45) #threshold of O count when fragment is classified as O
AS_REM_count = 1#number of REMs in a minute to be classified as AS
W_O_count = 5 #number os O in am inute to be classified as Ws

frag_per_min = 40

def plot_pr_curve(precisionsAS, recallsAS, precisionsQS, recallsQS, precisionsW, recallsW, AS_baseline, QS_baseline, W_baseline):
    """
    Plots precision-recall curve of all sleep states
    Parameters:
    - precisions*: list of precisions over range of threshold, per sleep state
    - recalls*: list of recalls over range of threshold, per sleep state
    - *_baseline: baseline for performance reference (see paper)
    """

    sns.set_style("whitegrid")

    auc_pr_AS = auc(recallsAS, precisionsAS)
    auc_pr_QS = auc(recallsQS, precisionsQS)
    
    plt.figure(figsize=(8, 6))

    plt.plot(recallsAS, precisionsAS, color="#ff3333", marker='.', label=f"AS (AP = {round(auc_pr_AS, 2)})")
    plt.plot(recallsQS, precisionsQS, color="#87e087", marker='.', label=f"QS (AP = {round(auc_pr_QS, 2)})")

    plt.axhline(y=AS_baseline, color="#ff3333", linestyle=':', linewidth=2, label=f"AS baseline {round(AS_baseline, 2)}")
    plt.axhline(y=QS_baseline, color="#87e087", linestyle=':', linewidth=2, label=f"QS baseline {round(QS_baseline, 2)}")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title(f"Precision-Recall Curve (threshold={REM_threshold})", fontsize=14)

    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.yticks(ticks, ticks)
    plt.xticks(ticks, ticks)

    plt.legend()
    plt.savefig(os.path.join(settings.predictions_path,"prcurve.jpg"), format='jpg', dpi=500) 

def get_baseline(target_class, all_true_labels, all_predicted_labels):
    """
    Get baseline, important for class imbalance (see paper)
    Parameters:
    - target_class: AS, QS, or W
    - all_true_labels: list of ground truth sleep states per minute
    - all_predicted_labels: list of all predicted sleep states per minute
    """

    filtered_true_labels = []
    for i in (range(len(all_predicted_labels))):
        if all_predicted_labels[i] != 'reject' and all_true_labels[i] != 'reject':
            filtered_true_labels.append(all_true_labels[i])
            
    return all_true_labels.count(target_class)/len(filtered_true_labels)


def get_metrics(target_class, true_labels = list(), predicted_labels = list()):
    """
    Get precision and recall per sleep state
    Parameters:
    - target_class: AS, QS, or W
    - all_true_labels: list of ground truth sleep states per minute
    - predicted_labels: list of all predicted sleep states per minute
    """

    filtered_true_labels = []
    filtered_predicted_labels = []
    for i in (range(len(predicted_labels))):
        if predicted_labels[i] != 'reject' and true_labels[i] != 'reject':
            filtered_predicted_labels.append(predicted_labels[i])
            filtered_true_labels.append(true_labels[i])
    TP = sum((p == target_class and g == target_class) for p, g in zip(filtered_predicted_labels, filtered_true_labels))
    FP = sum((p == target_class and g != target_class) for p, g in zip(filtered_predicted_labels, filtered_true_labels))
    FN = sum((p != target_class and g == target_class) for p, g in zip(filtered_predicted_labels, filtered_true_labels))

    precision = TP/(TP+FP+1e-10)
    recall = TP/(TP+FN+1e-10)

    return precision, recall

def plot_confusion_matrix(true_labels = list(), predicted_labels = list()):
    """
    Generate confusion matrix over all full-length videos of the test set.
    Parameters:
    - true_labels: list of ground truth sleep states per minute
    - predicted_labels: list of all predicted sleep states per minute
    """
    filtered_true_labels = []
    filtered_predicted_labels = []
    for i in (range(len(predicted_labels))):
        if predicted_labels[i] != 'reject' and true_labels[i] != 'reject':
            filtered_predicted_labels.append(predicted_labels[i])
            filtered_true_labels.append(true_labels[i])

    cm = confusion_matrix(filtered_true_labels, filtered_predicted_labels, labels=['AS', 'QS', 'W'])

    plt.figure(figsize=(10, 7))
    h = sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.arange(3), yticklabels=np.arange(3), annot_kws={"size": 16})
    ticklabels = ['AS', 'QS', 'W']
    h.set_xticklabels(ticklabels)
    h.set_yticklabels(ticklabels)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')

    plt.savefig(os.path.join(settings.predictions_path, "confusion_matrix.jpg"), format='jpg', dpi=500)  

def show_prediction_bar(true_classes, prediction_classes, cur_vid, REM_counts):
    """
    Generate prediction bar, showing ground truth and predicted sleep states over a full length video (see paper)
    Parameters:
    - true_classes: list of ground truth sleep states per minute
    - prediction_classes: list of all predicted sleep states per minute
    - cur_vid: vid to draw prediciton bar for
    - REM_counts: number of REMs per minute
    """
    mapping = {
        'AS': 0,
        'QS': 1,
        'W': 2,
        'reject': 3
    }
    true_classes = [mapping[item] for item in true_classes]
    prediction_classes = [mapping[item] for item in prediction_classes]

    colors = {
        0: '#ff3333',
        1: '#87e087',
        2: '#7373ff',
        3: 'black'
    }

    fig, ax = plt.subplots(figsize=(12, 2))

    cmap = plt.get_cmap('Reds')
    norm = Normalize(vmin=min(REM_counts), vmax=max(REM_counts))


    for i, cls in enumerate(prediction_classes):
        ax.barh(0.2, 1, left=i, color=colors[cls], height=0.1)
    for i, cls in enumerate(prediction_classes):
        if cls > 2: continue
        ax.barh(0.125, 1, left=i, color=cmap(norm(REM_counts[i])), height=0.05)
    for i, cls in enumerate(true_classes):
        ax.barh(0, 1, left=i, color=colors[cls], height=0.1)

    ax.set_xlim(0, len(true_classes))
    yticks = [-0.05, 0.0, 0.05, 0.1, 0.15, 0.2]

    ytick_labels = ['' for _ in yticks]
    ytick_labels[yticks.index(0.0)] = 'True'
    ytick_labels[yticks.index(0.15)] = 'Predictions'

    plt.yticks(yticks, ytick_labels)
    ax.tick_params(axis='y', which='both', length=0)
    ax.tick_params(axis='x', which='both', length=0)

    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', label='AS'),
        Patch(facecolor=colors[1], edgecolor='black', label='QS'),
        Patch(facecolor=colors[2], edgecolor='black', label='W'),
        Patch(facecolor=colors[3], edgecolor='black', label='reject')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=4)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([]) 
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels(['0%', '100%'])
    cbar.set_label('REM count')

    plt.tight_layout()
    plt.savefig(os.path.join(settings.predictions_path,cur_vid,"plot.jpg"), dpi=500, format='jpg')  

def is_valid_movement(frag_idx, positions, cur_vid):
    """
    Determine if fragment is valid. If infant moves too much, fragment is denied.
    Parameters:
    - frag_idx: index of fragment
    - positions: all localized eye positions over the fragment
    - cur_vid: vid being processed

    Returns:
    True, if difference between eye positions across frames is less than max_movement
    """
    img_path = os.path.join(settings.eye_frag_path, cur_vid, str(frag_idx), "0.jpg")
    image = cv2.imread(img_path)
    height, width, channels = image.shape

    max_movement = max_movement_fraction * width

    positions = np.array(positions).T

    min_x = min(positions[0]); max_x = max(positions[0])
    min_y = min(positions[1]); max_y = max(positions[1])
    if (max_x - min_x > max_movement):
        return False
    if (max_y - min_y > max_movement):
        return False

    return True


def compute_sleep_states(cur_vid):
    """
    Compute sleep states for each minute over a full-length video
    Parameters:
    - cur_vid: current video being processed, path
    """

    #Get predictions either of combined model, or open-closed model
    if settings.is_combined:
        pred_df = pd.read_csv(os.path.join(settings.predictions_path,cur_vid, "predictions.csv"), delimiter=';')
    else:
        pred_df = pd.read_csv(os.path.join(settings.predictions_path,cur_vid, "predictions_2.csv"), delimiter=';')

    frags_df = pd.read_csv(os.path.join(settings.eye_frag_path, cur_vid, "info.csv"), delimiter=';')
    true_pred_df = pd.read_csv(os.path.join(settings.predictions_path,cur_vid, "true_predictions.csv"), delimiter=';')

    last_frag_idx = frags_df.iloc[-1]["idx"]
    minute_count = last_frag_idx // frag_per_min  

    true_classes = []
    prediction_classes = []

    #Save configurations of sleep predictions
    with open(os.path.join(settings.predictions_path,cur_vid, "configurations.csv"), "w") as file:
        file.write("max_movement_fraction;REM_threshold;CREM_threshold;OREM_threshold;AS_REM_count;O_threshold;W_O_count\n")
        file.write(str(max_movement_fraction) + ";" + str(REM_threshold) + ";" + str(CREM_threshold) + ";" + str(OREM_threshold) + ";" + str(AS_REM_count) + ";" + str(O_threshold) + ";" + str(W_O_count) + "\n")

    #Save eye states and sleep state per mintute of video
    with open(os.path.join(settings.predictions_path,cur_vid, "sleep_predictions.csv"), "w") as file:
        file.write("min;state;C;O;CR;OR" + "\n")

    REM_counts = [] #Saves number of REM per minute
    for minute in range(minute_count):

        O = 0; C = 0; O_R = 0; C_R = 0
        for fragment in range(minute*frag_per_min, minute*frag_per_min + frag_per_min):
            row =  frags_df[frags_df['idx'] == fragment]
            #If fragment does not contain any detections, continue
            if row.empty:
                continue

            #check if infant moves too much
            positions = row['positions'].apply(ast.literal_eval)
            if(not is_valid_movement(fragment, positions.iloc[0], cur_vid)):
                continue

            #Get number of frames where eyes are open
            open_count = row['open_count'].iloc[0]

            #when using combined model
            if(settings.is_combined):
                row =  pred_df[pred_df['idx'] == fragment]

                prediction = row['predictions'].iloc[0]

                is_REM = True if prediction >= REM_threshold else False

                #Determine eye state of fragment (OR, O, CR, or C)
                if open_count > O_threshold:
                    if is_REM: O_R += 1
                    else: O += 1
                else:
                    if is_REM: C_R += 1
                    else: C += 1

            #when using open-closed model
            else:
                row =  pred_df[pred_df['idx'] == fragment]

                prediction = float(row['predictions'].iloc[0])
                eye_class = row['class'].iloc[0]

                #Determine eye state of fragment (OR, O, CR, or C)
                if(eye_class == "O"):
                    if prediction >= OREM_threshold:
                        O_R += 1
                    else:
                        O += 1
                else:
                    if prediction >= CREM_threshold:
                        C_R += 1
                    else:
                        C += 1
        
        #When valid fragments of a minute is less than half, reject
        if(O+C+O_R+C_R < frag_per_min//2):
            sleep_state = "reject"
        #else, determinute sleep state based on eye states over the minute
        else:
            sleep_state = 'QS'
            if O >= W_O_count:
                sleep_state='W'
            elif O_R+C_R >= AS_REM_count:
                sleep_state='AS'

        REM_counts.append(O_R+C_R)

        row =  true_pred_df[true_pred_df['idx'] == minute]
        true_classes.append(row['state'].iloc[0])
        if row['state'].iloc[0] == "reject": sleep_state = "reject" #when ground truth = reject, set predicted to reject
        prediction_classes.append(sleep_state)  

        #Append minute sleep state prediction to csv
        with open(os.path.join(settings.predictions_path,cur_vid, "sleep_predictions.csv"), "a") as file:
            file.write(str(minute) + ";" + str(sleep_state) + ";" + str(C) + ";" + str(O)+ ";" + str(C_R)+ ";" + str(O_R) + "\n")


    show_prediction_bar(true_classes, prediction_classes, cur_vid, REM_counts)

    return true_classes, prediction_classes


all_true_classes = []
all_predicted_classes = []

#process all videos in test set
for vid in settings.all_vids:          
    true_classes, prediction_classes = compute_sleep_states(vid[0:-4])
    all_true_classes += true_classes
    all_predicted_classes += prediction_classes
    print(f"vid {vid} -  AS{true_classes.count('AS')}, QS {true_classes.count('QS')}, W {true_classes.count('W')}")

plot_confusion_matrix(all_true_classes, all_predicted_classes)