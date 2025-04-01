import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import settings

import numpy as np

def make_boxplot(data, classes):
    plt.figure()

    palette = sns.color_palette("husl", 5)

    df = pd.DataFrame({
    "Fold": classes,
    "AP": data
    })
    sns.boxplot(x="Fold", y="AP", data=df, palette=palette)
    sns.stripplot(x="Fold", y="AP", data=df, color="black", jitter=True, alpha=0.6)

    plt.title("AP per fold over 10 train runs")
    plt.savefig(os.path.join(settings.results_dir, "boxplot.jpg"), format='jpg', dpi=500)  


APs = []
classes = []

for run in os.listdir(settings.results_dir):
    if(not run.isdigit()): continue

    path = os.path.join(settings.results_dir, run, "metrics.csv")

    metrics = pd.read_csv(path)

    for fold in range(5):
        result = metrics.loc[metrics["fold"] == fold, "AP"]
        AP = result.iloc[0] if not result.empty else None
        APs.append(AP)
        classes.append(fold)
        print(fold)
        print(AP)

make_boxplot(APs, classes)



