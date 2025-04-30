import os

is_OREM = False
is_combined = False

val_ids = [['416', '778'],
           ['004', '704', '773'],
           ['399', '875', '440'],
           ['554', '866'],
           ['614', '657']] 

frame_stack_count = 6
img_size = 64

data_dir = os.path.join(os.path.abspath(os.getcwd()),"REM", "raw", "cropped", "center")
results_dir = os.path.join(os.path.abspath(os.getcwd()),"REM-results")

model_filename = "model_architecture.json"
checkpoint_filename = "checkpoint.model.keras"

train_batch_size = [16]
train_initial_lr = [0.00018]
train_l2 = [0.04]
train_dropout = [0.5]

seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
