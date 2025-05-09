"""
This script is used to show a box plot of difference in AP between train runs (see paper), for each train fold.

Author: Lisa Groen
Date: May 7, 2025
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

import settings


def make_boxplot(APs, folds):
    """
    Compute sleep states for each minute over a full-length video
    Parameters:
    - APs: all APs over train runs
    - folds: fold indices
    """
    plt.figure()
    sns.set_style("whitegrid")
    plt.ylim(0.4, 1.0)

    palette = sns.color_palette("husl", 5)

    df = pd.DataFrame({
    "Fold": folds,
    "AP": APs
    })

    sns.boxplot(x="Fold", y="AP", hue="Fold", data=df, palette=palette, width=0.6, legend=False)

    for i, fold in enumerate(df["Fold"].unique()):
        values = df[df["Fold"] == fold]["AP"]
        x_jitter = np.random.normal(loc=i, scale=0.05, size=len(values)) - 0.5 # Small jitter
        plt.scatter(x_jitter, values, alpha=0.6, color=palette[int(fold-1)], s=20)

    plt.title("AP per fold over 10 train runs")
    plt.savefig(os.path.join(settings.results_dir, "boxplot.jpg"), format='jpg', dpi=500)  


APs = []
folds = []

#run folder name is integer
for run in os.listdir(settings.results_dir):
    if(not run.isdigit()): continue

    #path to metrics csv, where APs are saved
    path = os.path.join(settings.results_dir, run, "metrics.csv")

    #get metrics over all train runs
    metrics = pd.read_csv(path)

    #5 = number of folds
    for fold in range(5):
        result = metrics.loc[metrics["fold"] == fold, "AP"]
        AP = result.iloc[0] if not result.empty else None
        APs.append(AP)
        folds.append(fold+1)

make_boxplot(APs, folds)



