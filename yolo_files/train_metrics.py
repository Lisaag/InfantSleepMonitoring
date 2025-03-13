import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_curve(train_losses, val_losses, filename, gridsize=10.0):
    epochs = range(1, len(train_losses) + 1)

    all_losses = [*train_losses, *val_losses]

    sns.set_style("whitegrid")
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', markersize=4)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and validation {filename} loss')
    plt.legend()
    plt.grid(True)
    print(min(all_losses))
    offset = (gridsize - min(all_losses)) * 0.05
    plt.ylim(min(all_losses) - offset, gridsize)
    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"train_plots", filename+".jpg"), dpi=500, format='jpg') 

def plot_metrics_curve(metric_vals, filename):
    epochs = range(1, len(metric_vals) + 1)

    sns.set_style("whitegrid")
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metric_vals, marker='o', markersize=4)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(filename)
    plt.grid(True)
    plt.ylim(0, metric_vals.max() + 0.01)
    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"train_plots", filename+".jpg"), dpi=500, format='jpg') 



train_metrics = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), "runs", "OCC", "occ3", "results.csv"))

plot_loss_curve(train_metrics["train/box_loss"], train_metrics["val/box_loss"], "box", 3.5)
plot_loss_curve(train_metrics["train/cls_loss"], train_metrics["val/cls_loss"], "cls", 1.7)
plot_loss_curve(train_metrics["train/dfl_loss"], train_metrics["val/dfl_loss"], "dfl", 1.3)

plot_metrics_curve(train_metrics["metrics/precision(B)"], "Precision")
plot_metrics_curve(train_metrics["metrics/recall(B)"], "Recall")
plot_metrics_curve(train_metrics["metrics/mAP50(B)"], "mAP50")
plot_metrics_curve(train_metrics["metrics/mAP50-95(B)"], "mAp50-95")



