from moviepy.video.io.VideoFileClip import VideoFileClip
import os

def cut_video(file_name:str, patient_id:str, start_time:str):
    print('a')
    # input_file =  os.path.join(os.path.abspath(os.getcwd()), "vid", "frag", "IN", file_name)
    # output_file = os.path.join(os.path.abspath(os.getcwd()), "vid", "frag", patient_id, file_name)
    # if not os.path.exists(output_file):
    #     os.makedirs(output_file)
    # try:
    #     video = VideoFileClip(input_file)

    #     end_time = start_time + 1  # 1 second after the start time

    #     if start_time < 0 or end_time > video.duration:
    #         raise ValueError("Start time is out of the video duration range.")

    #     cut_clip = video.subclip(start_time, end_time)

    #     cut_clip.write_videofile(output_file, codec="libx264", audio_codec="aac")

    #     print(f"Video segment saved to {output_file}")

    # except Exception as e:
    #     print(f"An error occurred: {e}")
    # finally:
    #     video.close()