import os
from collections import defaultdict

import settings



def get_dataset_statistics():
    ps = defaultdict(lambda: {'O': 0, 'OR':0, 'C':0, 'CR':0})

    for patient in os.listdir(settings.data_dir):
        patient_dir:str = os.path.join(settings.data_dir, patient)
        patient_id:str = patient[0:3]
        if(settings.is_OREM and patient_id == '440'): continue
        print(patient_id)
        for eye_state in os.listdir(patient_dir):
            eye_state_dir = os.path.join(patient_dir, eye_state)
            for sample in os.listdir(eye_state_dir):
                if(patient_id in settings.val_ids and sample[-3:] == "AUG"): continue
                ps[patient_id][eye_state] += 1

    print(dict(ps))

get_dataset_statistics()

