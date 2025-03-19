import os

is_OREM = True

#val_ids = ['554', '778'] #fold1
#val_ids = ['004', '866'] #fold2
#val_ids = ['399', '657'] #fold3
#val_ids = ['416', '440'] #fold4
#val_ids = ['614', '704'] #fold5

#val_ids = ['416', '773'] #REM goed, OREM goed
val_ids = ['004', '778', '875'] #REM acceptabel/goed, OREM acceptabel/goed
#val_ids = ['399', '704', '866'] #REM goed , OREM goed
#val_ids = ['554', '778'] #REM goed, OREM goed
#val_ids = ['614', '657'] #REM acceptabel, OREM heel goed


frame_stack_count = 6

data_dir = os.path.join(os.path.abspath(os.getcwd()),"REM", "raw", "cropped", "center")
checkpoint_filepath = os.path.join(os.path.abspath(os.getcwd()),"REM-results","checkpoint.model.keras")
model_filepath = os.path.join(os.path.abspath(os.getcwd()),"REM-results","model_architecture.json")
