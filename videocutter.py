from moviepy.video.io.VideoFileClip import VideoFileClip
import os

def cut_video(file_name:str, patient_id:str, start_time:str):
    print('a')
    input_file =  os.path.join(os.path.abspath(os.getcwd()), "vid", "frag", "IN", file_name+".mp4")
    output_dir = os.path.join(os.path.abspath(os.getcwd()), "vid", "frag", patient_id)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
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