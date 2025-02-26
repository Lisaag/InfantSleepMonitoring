import pandas as pd
import os

import matplotlib.pyplot as plt

def plot_loss_curve(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"tmp.jpg"), format='jpg')   

def get_loss(loss, prev_loss):
    if(loss=="nan"): return float(prev_loss)
    else: return float(loss)


train_metrics = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), "runs", "AUG", "default-aug", "results.csv"))

#train/box_loss,train/cls_loss,train/dfl_loss
#val/box_loss,val/cls_loss,val/dfl_loss

train_box=[]
val_box=[]

for i in range(len(train_metrics)):
    train_box.append(get_loss(train_metrics["train/box_loss"][i], train_metrics["train/box_loss"][i-1]))
    val_box.append(get_loss(train_metrics["val/box_loss"][i], train_metrics["val/box_loss"][i-1]))
