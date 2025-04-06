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

max_movement_fraction = 0.6

REM_threshold = 0.3 #threshold of when fragment is classified as REM
O_threshold = 10 * (settings.fragment_length//45) #threshold of O count when fragment is classified as O
AS_REM_count = 5 #number of REMs in a minute to be classified as AS
W_O_count = 20 #number os O in am inute to be classified as W

frag_per_min = 40

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
    pred_df = pd.read_csv(os.path.join(settings.predictions_path, "predictions.csv"), delimiter=';')
    frags_df = pd.read_csv(os.path.join(settings.eye_frag_path, settings.cur_vid[:-4], "info.csv"), delimiter=';')

    last_frag_idx = frags_df.iloc[-1]["idx"]
    minute_count = last_frag_idx // frag_per_min  


    with open(os.path.join(settings.predictions_path, "sleep_predictions.csv"), "w") as file:
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
            if(not is_valid_movement(fragment, positions.iloc[0])):
                continue

            open_count = row['open_count'].iloc[0]

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
            
            print(f'O - {O}, OR - {O_R}, C - {C}, CR - {C_R} ')
        
        sleep_state = 'QS'
        if O_R+C_R >= AS_REM_count:
            sleep_state='AS'
        elif O >= W_O_count:
            sleep_state='W'

        print(f'minute {minute} classified as {sleep_state}')

        with open(os.path.join(settings.predictions_path, "sleep_predictions.csv"), "a") as file:
            file.write(str(minute) + ";" + str(sleep_state) + ";" + str(C) + ";" + str(O)+ ";" + str(C_R)+ ";" + str(O_R) + "\n")

            
compute_sleep_states()




    


    



