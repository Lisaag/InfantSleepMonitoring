import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns


def plot_loss_curve(train_losses, val_losses, filename, gridsize=10.0):
    epochs = range(1, len(train_losses) + 1)

    sns.set_style("whitegrid")
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', markersize=4)
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s', markersize=4)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Training and validation {filename} loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, gridsize)
    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"train_plots", filename+".jpg"), dpi=500, format='jpg') 

def plot_metrics_curve(metric_vals, filename, gridsize=10.0):
    epochs = range(1, len(metric_vals) + 1)

    sns.set_style("whitegrid")
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, metric_vals, marker='o', markersize=4)
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(filename)
    plt.grid(True)
    plt.ylim(0, gridsize)
    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"train_plots", filename+".jpg"), dpi=500, format='jpg') 

  

def get_loss(train_metrics, train_box, i):
    if(i == 0): return 0
    elif(train_metrics[i]=="nan"): return train_box[i-1]
    else: return float(train_metrics[i])


train_metrics = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), "runs", "AUG", "no-aug3", "results.csv"))

#train/box_loss,train/cls_loss,train/dfl_loss
#val/box_loss,val/cls_loss,val/dfl_loss

train_box=[]
val_box=[]

for i in range(len(train_metrics)):
    train_box.append(get_loss(train_metrics["train/box_loss"], train_box, i))
    val_box.append(get_loss(train_metrics["val/box_loss"], val_box, i))

plot_loss_curve(train_box, val_box, "box", 6)

train_box=[]
val_box=[]

for i in range(len(train_metrics)):
    train_box.append(get_loss(train_metrics["train/cls_loss"], train_box, i))
    val_box.append(get_loss(train_metrics["val/cls_loss"], val_box, i))

plot_loss_curve(train_box, val_box, "cls", 4.5)
train_box=[]
val_box=[]

for i in range(len(train_metrics)):
    train_box.append(get_loss(train_metrics["train/dfl_loss"], train_box, i))
    val_box.append(get_loss(train_metrics["val/dfl_loss"], val_box, i))

plot_loss_curve(train_box, val_box, "dfl", 3)

plot_metrics_curve(train_metrics["metrics/precision(B)"], "Precision", 1)
plot_metrics_curve(train_metrics["metrics/recall(B)"], "Recall", 1)
plot_metrics_curve(train_metrics["metrics/mAP50(B)"], "mAP50", 1)
plot_metrics_curve(train_metrics["metrics/mAP50-95(B)"], "mAp50-95", 1)



