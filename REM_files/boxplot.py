import pandas as pd
import matplotlib as plt
import seaborn as sns

import os
import settings

import numpy as np

def make_boxplot(data, path):
    plt.figure()
    print(data)
    data = np.array(data).T
    print(data)
    length = len(data[0])
    data = data.flatten()
    print(data)
    df = pd.DataFrame({
    "Group": ["A"] * length + ["B"] * length + ["C"] * length + ["D"] * length + ["E"] * length,
    "Values": data
    })
    sns.boxplot(x="Category", y="Values", data=df)

    plt.title("Grouped Box Plots with Seaborn")
    plt.savefig(path, format='jpg', dpi=500)  

for run in os.listdir(settings.results_dir):
    if(not run.isdigit()): continue

    path = os.path.join(settings.results_dir, run, "metrics.csv")

    metrics = pd.read_csv(path)

    for fold in range(5):
        result = metrics[metrics['fold'] == fold]
        print(fold)
        print(result["AP"])



