import os

fragment_dir = os.path.join(os.path.abspath(os.getcwd()), "frags")

for dir in os.listdir(fragment_dir):
    patient_nr = int(dir[0:3])
    print(dir)
    print(patient_nr)
