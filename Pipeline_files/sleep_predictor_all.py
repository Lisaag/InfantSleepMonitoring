#If movement is more than X --> discard

#if x REM detected --> AS
#if x O detected --> W
#else, QS

import settings

import numpy as np
import pandas as pd
import os
import ast

import cv2

import matplotlib.pyplot as plt

from matplotlib.patches import Patch

from sklearn.metrics import auc

import seaborn as sns
from sklearn.metrics import confusion_matrix

max_movement_fraction = 0.9

CREM_threshold = 0.55 #threshold of when fragment is classified as REM
OREM_threshold = 0.75#threshold of when fragment is classified as REM

REM_threshold = 0.5 #threshold of when fragment is classified as REM
O_threshold = 3 * (settings.fragment_length//45) #threshold of O count when fragment is classified as O
AS_REM_count = 0#number of REMs in a minute to be classified as AS
W_O_count = 5 #number os O in am inute to be classified as Ws

frag_per_min = 40

def plot_pr_curve(precisionsAS, recallsAS, precisionsQS, recallsQS, AS_baseline, QS_baseline):
    sns.set_style("whitegrid")

    auc_pr_AS = auc(recallsAS, precisionsAS)
    auc_pr_QS = auc(recallsQS, precisionsQS)
    
    plt.figure(figsize=(8, 6))
    plt.axhline(y=AS_baseline, color="#ff3333", linestyle=':', linewidth=2)
    plt.axhline(y=QS_baseline, color="#87e087", linestyle=':', linewidth=2)

    plt.plot(recallsAS, precisionsAS, color="#ff3333", marker='.', label=f"AS {auc_pr_AS}")
    plt.plot(recallsQS, precisionsQS, color="#87e087", marker='.', label=f"QS {auc_pr_QS}")

    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curve", fontsize=14)

    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.yticks(ticks, ticks)
    plt.xticks(ticks, ticks)

    plt.legend()
    plt.savefig(os.path.join(settings.predictions_path,"prcurve.jpg"), format='jpg', dpi=500) 

def get_baseline(target_class, all_true_labels, all_predicted_labels):
    filtered_true_labels = []
    for i in (range(len(all_predicted_labels))):
        if all_predicted_labels[i] != 'reject' and all_true_labels[i] != 'reject':
            filtered_true_labels.append(all_true_labels[i])
            
    return all_true_labels.count(target_class)/len(filtered_true_labels)


def get_metrics(target_class, true_labels = list(), predicted_labels = list()):
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
    print(true_labels)
    print(predicted_labels)
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

def show_prediction_bar(true_classes, prediction_classes, cur_vid):
    mapping = {
        'AS': 0,
        'QS': 1,
        'W': 2,
        'reject': 3
    }
    true_classes = [mapping[item] for item in true_classes]
    prediction_classes = [mapping[item] for item in prediction_classes]

    # Step 2: Define class colors
    colors = {
        0: '#ff3333',
        1: '#87e087',
        2: '#7373ff',
        3: 'black'
    }

    # Step 3: Create the plot
    fig, ax = plt.subplots(figsize=(12, 2))

    for i, cls in enumerate(prediction_classes):
        ax.barh(0.15, 1, left=i, color=colors[cls], height=0.1)
    for i, cls in enumerate(true_classes):
        ax.barh(0, 1, left=i, color=colors[cls], height=0.1)

    # Step 4: Aesthetics
    ax.set_xlim(0, len(true_classes))
    #ax.set_ylim(-0.5, 0.5)
    #ax.axis('off')  # Turn off axes for cleaner look
        # Your y-ticks
    yticks = [-0.05, 0.0, 0.05, 0.1, 0.15, 0.2]

    # Custom labels (empty strings for ticks you don't want labeled)
    ytick_labels = ['' for _ in yticks]
    ytick_labels[yticks.index(0.0)] = 'True'
    ytick_labels[yticks.index(0.15)] = 'Predictions'

    plt.yticks(yticks, ytick_labels)
    ax.tick_params(axis='y', which='both', length=0)
    ax.tick_params(axis='x', which='both', length=0)

    # Legend
    legend_elements = [
        Patch(facecolor=colors[0], edgecolor='black', label='AS'),
        Patch(facecolor=colors[1], edgecolor='black', label='QS'),
        Patch(facecolor=colors[2], edgecolor='black', label='W'),
        Patch(facecolor=colors[3], edgecolor='black', label='reject')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.4), ncol=4)

    plt.tight_layout()
    plt.savefig(os.path.join(settings.predictions_path,cur_vid,"plot.jpg"), dpi=500, format='jpg')  

def is_valid_movement(frag_idx, positions, cur_vid):
    img_path = os.path.join(settings.eye_frag_path, cur_vid, str(frag_idx), "0.jpg")
    image = cv2.imread(img_path)
    height, width, channels = image.shape

    max_movement = max_movement_fraction * width

    positions = np.array(positions).T

    min_x = min(positions[0]); max_x = max(positions[0])
    min_y = min(positions[1]); max_y = max(positions[1])
    if (max_x - min_x > max_movement):
        print(f"FRAG {frag_idx} TOO MUCH MOVEMENT ON X AXIS")
        return False
    if (max_y - min_y > max_movement):
        print(f"FRAG {frag_idx} TOO MUCH MOVEMENT ON y AXIS")
        return False

    return True


def compute_sleep_states(cur_vid):
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


    with open(os.path.join(settings.predictions_path,cur_vid, "configurations.csv"), "w") as file:
        file.write("max_movement_fraction;REM_threshold;CREM_threshold;OREM_threshold;AS_REM_count;O_threshold;W_O_count\n")
        file.write(str(max_movement_fraction) + ";" + str(REM_threshold) + ";" + str(CREM_threshold) + ";" + str(OREM_threshold) + ";" + str(AS_REM_count) + ";" + str(O_threshold) + ";" + str(W_O_count) + "\n")

    with open(os.path.join(settings.predictions_path,cur_vid, "sleep_predictions.csv"), "w") as file:
        file.write("min;state;C;O;CR;OR" + "\n")

    print(f"{minute_count} minutes detected")

    for minute in range(minute_count):
        print(f"processing minute {minute}")

        O = 0; C = 0; O_R = 0; C_R = 0
        for fragment in range(minute*frag_per_min, minute*frag_per_min + frag_per_min):
            row =  frags_df[frags_df['idx'] == fragment]
            if row.empty:
                print(f'no fragment idx {fragment} found')
                continue
            positions = row['positions'].apply(ast.literal_eval)
            if(not is_valid_movement(fragment, positions.iloc[0], cur_vid)):
                continue

            open_count = row['open_count'].iloc[0]


            if(settings.is_combined):
                row =  pred_df[pred_df['idx'] == fragment]

                prediction = row['predictions'].iloc[0]

                #TODO misschien andere threshold voor O_R vs C_R?
                is_REM = True if prediction >= REM_threshold else False

                if open_count > O_threshold:
                    if is_REM: O_R += 1
                    else: O += 1
                else:
                    if is_REM: C_R += 1
                    else: C += 1
            else:
                row =  pred_df[pred_df['idx'] == fragment]

                prediction = float(row['predictions'].iloc[0])
                eye_class = row['class'].iloc[0]
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

            
            print(f'O - {O}, OR - {O_R}, C - {C}, CR - {C_R} ')
        
        if(O+C+O_R+C_R < frag_per_min//2):
            sleep_state = "reject"
        else:
            sleep_state = 'QS'
            if O >= W_O_count:
                sleep_state='W'
            elif O_R+C_R >= AS_REM_count:
                sleep_state='AS'


    
        print(f'minute {minute} classified as {sleep_state}')

        row =  true_pred_df[true_pred_df['idx'] == minute]
        true_classes.append(row['state'].iloc[0])
        if row['state'].iloc[0] == "reject": sleep_state = "reject"
        prediction_classes.append(sleep_state)  


        with open(os.path.join(settings.predictions_path,cur_vid, "sleep_predictions.csv"), "a") as file:
            file.write(str(minute) + ";" + str(sleep_state) + ";" + str(C) + ";" + str(O)+ ";" + str(C_R)+ ";" + str(O_R) + "\n")


    show_prediction_bar(true_classes, prediction_classes, cur_vid)
    #plot_confusion_matrix(true_classes, prediction_classes)

    return true_classes, prediction_classes



precisionsAS = []; recallsAS = []
precisionsQS = []; recallsQS = []
for i in range(0, 40):
    AS_REM_count = i  

    all_true_classes = []
    all_predicted_classes = []
    for vid in settings.all_vids:          
        true_classes, prediction_classes = compute_sleep_states(vid[0:-4])
        all_true_classes += true_classes
        all_predicted_classes += prediction_classes

    plot_confusion_matrix(all_true_classes, all_predicted_classes)


    precisionAS, recallAS = get_metrics("AS", all_true_classes, all_predicted_classes)
    precisionQS, recallQS = get_metrics("QS", all_true_classes, all_predicted_classes)

    precisionsAS.append(precisionAS)
    recallsAS.append(recallAS)
    precisionsQS.append(precisionQS)
    recallsQS.append(recallQS)

with open(os.path.join(settings.predictions_path,"prs.txt"), "w") as file:
    file.write(f"precisions AS: {precisionsAS} \n")
    file.write(f"recalls AS: {recallsAS} \n")
    file.write(f"precisions QS: {precisionsQS} \n")
    file.write(f"recalls QS: {recallsQS} \n")

AS_baseline = get_baseline("AS", all_true_classes, all_predicted_classes)
QS_baseline = get_baseline("QS", all_true_classes, all_predicted_classes)
plot_pr_curve(precisionsAS, recallsAS, precisionsQS, recallsQS, AS_baseline, QS_baseline)





    


    



