import pandas as pd
import os

import matplotlib.pyplot as plt

def plot_loss_curve(train_losses, val_losses):
    epochs = range(1, len(train_losses) + 1)

    plt.style.use("seaborn-darkgrid") 
    
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(os.path.abspath(os.getcwd()),"tmp.jpg"), format='jpg')   

def get_loss(train_metrics, train_box, i):
    if(i == 0): return 0
    elif(train_metrics[i]=="nan"): return train_box[i-1]
    else: return float(train_metrics[i])


train_metrics = pd.read_csv(os.path.join(os.path.abspath(os.getcwd()), "runs", "AUG", "default-aug", "results.csv"))

#train/box_loss,train/cls_loss,train/dfl_loss
#val/box_loss,val/cls_loss,val/dfl_loss

train_box=[]
val_box=[]

for i in range(len(train_metrics)):
    train_box.append(get_loss(train_metrics["train/box_loss"], train_box, i))
    val_box.append(get_loss(train_metrics["val/box_loss"], val_box, i))

plot_loss_curve(train_box, val_box)
