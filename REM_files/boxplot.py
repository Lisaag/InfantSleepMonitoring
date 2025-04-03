import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import settings

import numpy as np

def make_boxplot(data, classes):
    plt.figure()
    sns.set_style("whitegrid")

    palette = sns.color_palette("husl", 5)

    df = pd.DataFrame({
    "Fold": classes,
    "AP": data
    })

    sns.boxplot(x="Fold", y="AP", hue="Fold", data=df, palette=palette, width=0.6, legend=False)
    #sns.stripplot(x="Folds", y="AP", hue="Folds", palette=palette,  data=df, color="black", jitter=True, alpha=0.6, legend=False)
    for i, fold in enumerate(df["Fold"].unique()):
        values = df[df["Fold"] == fold]["AP"]
        x_jitter = np.random.normal(loc=i, scale=0.05, size=len(values)) - 0.5 # Small jitter
        plt.scatter(x_jitter, values, alpha=0.6, color=palette[int(fold-1)], s=20)

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
        classes.append(fold+1)
        print(fold)
        print(AP)

make_boxplot(APs, classes)



