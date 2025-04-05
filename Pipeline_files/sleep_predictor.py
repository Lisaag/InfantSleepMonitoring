#If movement is more than X --> discard

#if x REM detected --> AS
#if x O detected --> W
#else, QS

import settings

import numpy as np
import pandas as pd
import os

REM_threshold = 0.5 #threshold of when fragment is classified as REM
O_threshold = 20 * (settings.fragment_length//45) #threshold of O count when fragment is classified as O
AS_REM_count = 5 #number of REMs in a minute to be classified as AS
W_O_count = 30 #number os O in am inute to be classified as W

pred_df = pd.read_csv(os.path.join(settings.predictions_path, "predictions.csv"), delimiter=';')

for i in range(pred_df.shape[0]):
    predictions =  pred_df[pred_df['idx'] == i]
    print(predictions)
