import os

is_OREM = False #whether open model, or closed model is being trained/validated
is_combined = False #whether the combined model is being trained/valideated (see paper)

#patient id's of validation split for all folds
val_ids = [['416', '778'],
           ['004', '704', '773'],
           ['399', '875', '440'],
           ['554', '866'],
           ['614', '657']] 

frame_stack_count = 6 #number of images of a stack of images on which the REM model is trained
img_size = 64 #size of images

data_dir = os.path.join(os.path.abspath(os.getcwd()),"REM", "raw", "cropped", "center") #directory where processed data is stored (can be "center", "interpolate", "every")
results_dir = os.path.join(os.path.abspath(os.getcwd()),"REM-results") #directory where train results are stored

model_filename = "model_architecture.json"
checkpoint_filename = "checkpoint.model.keras"

#can be adjusted to find best configuration (grid search)
train_batch_size = [16]
train_initial_lr = [0.00018]
train_l2 = [0.04]
train_dropout = [0.5]

#seeds used for randomization, used to train multiple time with same configurations, but different seeds. Was used to find the difference in AP between train runs
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
