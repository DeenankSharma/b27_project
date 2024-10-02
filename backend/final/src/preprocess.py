import os
from transformers import VideoFeatureExtractor

extractor = VideoFeatureExtractor.from_pretrained('facebook/timesformer-base-finetuned')

def preprocess_video(video_path):
    # Implement video frame extraction here
    video_frames = extract_frames(video_path)  # Custom function to extract frames
    return extractor(video_frames, return_tensors="pt")

def get_video_paths(data_folder):
    video_paths = []
    for folder in ['Celeb-real', 'YouTube-real', 'Celeb-synthesis']:
        folder_path = os.path.join(data_folder, folder)
        for video_file in os.listdir(folder_path):
            video_paths.append(os.path.join(folder_path, video_file))
    return video_paths
