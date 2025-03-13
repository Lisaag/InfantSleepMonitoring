import os
import cv2

root_directory = os.path.join(os.path.abspath(os.getcwd()), "vid", "all")
out_directory = os.path.join(os.path.abspath(os.getcwd()), "vid", "samples")


def extract_frames(file_name, video_path, output_folder, frame_interval, saved_count):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        print(f"Error: Unable to open video file {video_path}")
        return

    frame_count = 0

    while True:
        success, frame = video_capture.read()

        if not success:
            print("End of video or error reading the video.")
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{file_name}-{saved_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            print(f"Saved: {frame_filename}")

        frame_count += 1

    video_capture.release()
    return saved_count


for subdir, dirs, files in os.walk(root_directory):
    subdir_name = os.path.basename(subdir)
    outdir_name = os.path.join(out_directory, subdir_name)
    saved_count = 0
    print(subdir_name)
    for file in files:
        file_path = os.path.join(subdir, file)
        saved_count = extract_frames(subdir_name, file_path, outdir_name, 600, saved_count)
