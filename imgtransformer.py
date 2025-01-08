from moviepy.editor import VideoFileClip
import os

def rotate_img(file_path, file_name, rotation):
    input_path = os.path.join(os.path.abspath(os.getcwd()), file_path, file_name)
    output_path = os.path.join(os.path.abspath(os.getcwd()), file_path, 'ROT'+file_name)
    angle = int(rotation)
    try:
        # Load the video file
        clip = VideoFileClip(input_path)

        # Rotate the video
        rotated_clip = clip.rotate(angle)

        # Write the rotated video to the output file
        rotated_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

        print(f"Video successfully rotated and saved to {output_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
