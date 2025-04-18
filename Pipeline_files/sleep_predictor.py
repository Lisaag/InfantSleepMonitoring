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

import seaborn as sns
from sklearn.metrics import confusion_matrix

from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

max_movement_fraction = 0.5

CREM_threshold = 0.8 #threshold of when fragment is classified as REM
OREM_threshold = 0.8#threshold of when fragment is classified as REM


REM_threshold = 0.8 #threshold of when fragment is classified as REM
O_threshold = 0 * (settings.fragment_length//45) #threshold of O count when fragment is classified as O
AS_REM_count = 10#number of REMs in a minute to be classified as AS
W_O_count = 5 #number os O in am inute to be classified as W

frag_per_min = 40

 

def plot_confusion_matrix(true_labels = list(), predicted_labels = list()):
    print(true_labels)
    print(predicted_labels)
    filtered_true_labels = []
    filtered_predicted_labels = []
    for i in (range(len(predicted_labels))):
        if predicted_labels[i] != 'reject':
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

    plt.savefig(os.path.join(settings.predictions_path, settings.cur_vid[:-4], "confusion_matrix.jpg"), format='jpg', dpi=500)  

def show_prediction_bar(true_classes, prediction_classes, REM_counts):
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

    cmap = plt.get_cmap('Reds')
    norm = Normalize(vmin=min(REM_counts), vmax=max(REM_counts))


    for i, cls in enumerate(prediction_classes):
        ax.barh(0.2, 1, left=i, color=colors[cls], height=0.1)
    for i, cls in enumerate(prediction_classes):
        if cls > 1: continue
        ax.barh(0.125, 1, left=i, color=cmap(norm(REM_counts[i])), height=0.05)
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


    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([]) 
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_ticks([norm.vmin, norm.vmax])
    cbar.set_ticklabels(['0%', '100%'])
    cbar.set_label('REM count')

    plt.tight_layout()
    plt.savefig(os.path.join(settings.predictions_path,settings.cur_vid[:-4],"plot.jpg"), dpi=500, format='jpg')  

def is_valid_movement(frag_idx, positions):
    img_path = os.path.join(settings.eye_frag_path, settings.cur_vid[:-4], str(frag_idx), "0.jpg")
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


def compute_sleep_states():
    if settings.is_combined:
        pred_df = pd.read_csv(os.path.join(settings.predictions_path,settings.cur_vid[:-4], "predictions.csv"), delimiter=';')
    else:
        pred_df = pd.read_csv(os.path.join(settings.predictions_path,settings.cur_vid[:-4], "predictions_2.csv"), delimiter=';')

    frags_df = pd.read_csv(os.path.join(settings.eye_frag_path, settings.cur_vid[:-4], "info.csv"), delimiter=';')
    true_pred_df = pd.read_csv(os.path.join(settings.predictions_path,settings.cur_vid[:-4], "true_predictions.csv"), delimiter=';')

    last_frag_idx = frags_df.iloc[-1]["idx"]
    minute_count = last_frag_idx // frag_per_min  

    true_classes = []
    prediction_classes = []


    with open(os.path.join(settings.predictions_path,settings.cur_vid[:-4], "configurations.csv"), "w") as file:
        file.write("max_movement_fraction;REM_threshold;CREM_threshold;OREM_threshold;AS_REM_count;O_threshold;W_O_count\n")
        file.write(str(max_movement_fraction) + ";" + str(REM_threshold) + ";" + str(CREM_threshold) + ";" + str(OREM_threshold) + ";" + str(AS_REM_count) + ";" + str(O_threshold) + ";" + str(W_O_count) + "\n")

    with open(os.path.join(settings.predictions_path,settings.cur_vid[:-4], "sleep_predictions.csv"), "w") as file:
        file.write("min;state;C;O;CR;OR" + "\n")

    print(f"{minute_count} minutes detected")


    REM_counts = []
    for minute in range(minute_count):
        print(f"processing minute {minute}")

        O = 0; C = 0; O_R = 0; C_R = 0
        for fragment in range(minute*frag_per_min, minute*frag_per_min + frag_per_min):
            row =  frags_df[frags_df['idx'] == fragment]
            if row.empty:
                print(f'no fragment idx {fragment} found')
                continue
            positions = row['positions'].apply(ast.literal_eval)
            if(not is_valid_movement(fragment, positions.iloc[0])):
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

        REM_counts.append(O_R+C_R)
    
        print(f'minute {minute} classified as {sleep_state}')

        row =  true_pred_df[true_pred_df['idx'] == minute]
        true_classes.append(row['state'].iloc[0])
        if row['state'].iloc[0] == "reject": sleep_state = "reject"
        prediction_classes.append(sleep_state)  


        with open(os.path.join(settings.predictions_path,settings.cur_vid[:-4], "sleep_predictions.csv"), "a") as file:
            file.write(str(minute) + ";" + str(sleep_state) + ";" + str(C) + ";" + str(O)+ ";" + str(C_R)+ ";" + str(O_R) + "\n")


    show_prediction_bar(true_classes, prediction_classes, REM_counts)
    plot_confusion_matrix(true_classes, prediction_classes)


          
compute_sleep_states()





    


    



