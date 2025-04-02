import os

is_OREM = True

val_ids = [['416', '778'],
           ['004', '704', '773'],
           ['399', '875', '440'],
           ['554', '866'],
           ['614', '657']] 
#val_ids = [['416', '778']]

frame_stack_count = 6
img_size = 64

data_dir = os.path.join(os.path.abspath(os.getcwd()),"REM", "raw", "cropped", "center")
results_dir = os.path.join(os.path.abspath(os.getcwd()),"REM-results")

model_filename = "model_architecture.json"
checkpoint_filename = "checkpoint.model.keras"
#checkpoint_filepath = os.path.join(os.path.abspath(os.getcwd()),"REM-results","checkpoint.model.keras")
#model_filepath = os.path.join(os.path.abspath(os.getcwd()),"REM-results","model_architecture.json")


# train_batch_size = [2, 4, 8, 16]
# train_initial_lr = [0.001, 0.0001, 0.00001]
# train_l2 = [0.01, 0.1, 1.0]
# train_dropout = [0.3, 0.5, 0.7]

train_batch_size = [16]
train_initial_lr = [0.00018]
train_l2 = [0.08]
train_dropout = [0.5]

#seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9]
seeds = [0, 1, 2, 3]