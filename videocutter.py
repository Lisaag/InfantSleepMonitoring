from moviepy.video.io.VideoFileClip import VideoFileClip
import os

import pandas as pd


def cut_video(patient_id:str, REM_class:str, file_name:str, start_time:str, input_file:str):
    #print('a')
    #input_file =  os.path.join(os.path.abspath(os.getcwd()), "vid", "frag", "IN", file_name+".mp4")
    #output_dir = os.path.join(os.path.abspath(os.getcwd()), "vid", "frag", patient_id)
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    try:
        video = VideoFileClip(input_file)
        start_time=float(start_time)

        end_time = start_time + 1.5  # 1 second after the start time

        if start_time < 0 or end_time > video.duration:
            raise ValueError("Start time is out of the video duration range.")

        cut_clip = video.subclip(start_time, end_time)


        output_file = output_dir = os.path.join(output_dir, file_name+"T"+str(start_time)+".mp4")
        cut_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

        print(f"Video segment saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        video.close()


#TODO write function to get input file directory and file name.

csv_dir = os.path.join(os.path.abspath(os.getcwd()), "REMinfo.csv")
fragments_dir = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "fragments")

df_all = pd.read_csv(csv_dir)
for i in range(len(df_all)):
    frag_class_dirs = next(os.walk(os.path.join(fragments_dir, df_all["id"][i] )))[1]
    class_dir:str = ""
    if(df_all["class"][i] in frag_class_dirs): class_dir = df_all["class"][i]
    elif((df_all["class"][i] == "OR" or df_all["class"][i] == "CR") and "OR-CR" in frag_class_dirs): class_dir = "OR-CR"
    else: continue



    input_file = os.path.join(fragments_dir, df_all["id"][i], class_dir, df_all["filename"][i]+".mp4")
    print(f"{i} - {input_file}")

    #cut_video(df_all["id"][i], df_all["class"][i], df_all["filename"][i], df_all["timestamp"][i], input_file)
    