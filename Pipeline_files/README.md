1. Use eye_localizer.py to localize the eyes over a video
2. Then, use video_cutter to crop the eyes from the video.
3. Then, use REM_predictor_combined.py for the combined pipeline, or REM_predictor.py for the open/closed pipeline, to predict the eye states.
4. Finally, use sleep_predictor_all.py to determine the sleep state per minute, for all video's of the dataset.