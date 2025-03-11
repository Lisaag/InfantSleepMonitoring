import os
import re

def get_fragment_info(fragment_filname):
    match = re.match(r'(frag\d+)T([\d.]+)\.mp4', fragment_filname)

    if match:
        filename = match.group(1)
        timestamp = match.group(2)
        return(filename, timestamp)
    else:
        print("CANNOT GET FRAGMENT INFO")
        return(None, None)  

fragment_dir = os.path.join(os.path.abspath(os.getcwd()), "frags")
csv_dir = os.path.join(os.path.abspath(os.getcwd()), "REMinfo.csv")

if os.path.isfile(csv_dir):
    os.remove(csv_dir)

with open(csv_dir, "w") as file:
    file.write("id,"+ "class,"+"filename,"+"timestamp"+"\n")

for patient in os.listdir(fragment_dir):
    #patient_nr = int(patient[0:3])
    print(patient)
    #print(patient_nr)

    for REM_class in os.listdir(os.path.join(fragment_dir, patient)):
        for fragment in os.listdir(os.path.join(fragment_dir, patient, REM_class, "raw")):
            frag_filename, frag_timestamp = get_fragment_info(fragment)

            with open(csv_dir, "a") as file:
                file.write(str(patient)+","+str(REM_class)+","+str(frag_filename)+","+str(frag_timestamp)+"\n")
