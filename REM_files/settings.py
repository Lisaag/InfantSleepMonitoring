import os

is_OREM = True

#val_ids = ['416', '773'] #REM goed, OREM goed
#val_ids = ['004', '778', '875'] #REM acceptabel/goed, OREM heel goed
#val_ids = ['399', '704', '440'] #REM goed , OREM goed
#val_ids = ['554', '866'] #REM goed, OREM goed  
#val_ids = ['614', '657'] #REM acceptabel, OREM heel goed

val_ids = [['416', '773'],
           ['004', '778', '875'],
           ['399', '704', '440'],
           ['554', '866'],
           ['614', '657']] 

#val_ids = ['416', '875'] #<-- SLECHTE FOLD EXAMPLE



frame_stack_count = 6

data_dir = os.path.join(os.path.abspath(os.getcwd()),"REM", "raw", "cropped", "center")
results_dir = os.path.join(os.path.abspath(os.getcwd()),"REM-results")

model_filename = "model_architecture.json"
checkpoint_filename = "checkpoint.model.keras"
#checkpoint_filepath = os.path.join(os.path.abspath(os.getcwd()),"REM-results","checkpoint.model.keras")
#model_filepath = os.path.join(os.path.abspath(os.getcwd()),"REM-results","model_architecture.json")
