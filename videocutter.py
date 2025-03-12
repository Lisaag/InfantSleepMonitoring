from moviepy.video.io.VideoFileClip import VideoFileClip
import os

import pandas as pd


def cut_video(file_name:str, start_time:str, input_file:str, output_dir:str):
    try:
        video = VideoFileClip(input_file)

        fragment_length = 1.5
        aug_offset = 0.1

        start_time = float(start_time)
        end_time = start_time + fragment_length  # 1 second after the start time

        if start_time < 0 or end_time > video.duration:
            raise ValueError("Start time is out of the video duration range.")

        cut_clip = video.subclip(start_time - aug_offset, end_time + aug_offset)

        output_file = os.path.join(output_dir, file_name+"T"+str(start_time)+".mp4")
        cut_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

        print(f"Video segment saved to {output_file}")

    except Exception as e:
        print(f"An error occurred when cutting fragment: {e}")
    finally:
        video.close()

csv_dir = os.path.join(os.path.abspath(os.getcwd()), "REMinfo.csv")
fragments_dir = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "fragments")

df_all = pd.read_csv(csv_dir)
for i in range(len(df_all)):
    frag_class_dirs = next(os.walk(os.path.join(fragments_dir, df_all["id"][i] )))[1]
    class_dir:str = ""
    if(df_all["class"][i] in frag_class_dirs): class_dir = df_all["class"][i]
    elif((df_all["class"][i] == "OR" or df_all["class"][i] == "CR") and "OR-CR" in frag_class_dirs): class_dir = "OR-CR"
    else:
        print("NO FRAGMENT FILE FOUND")
        continue

    output_dir = os.path.join(os.path.abspath(os.getcwd()), "REM", "raw", "cutout", df_all["id"][i], df_all["class"][i])
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_file = os.path.join(fragments_dir, df_all["id"][i], class_dir, df_all["filename"][i]+".mp4")
    print(f"{i} - {input_file}")

    cut_video(df_all["filename"][i], df_all["timestamp"][i], input_file, output_dir)
    