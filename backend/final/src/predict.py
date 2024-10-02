from transformers import VideoClassificationModel
import torch

from preprocess import preprocess_video

model = VideoClassificationModel.from_pretrained('../model/your_model_directory')

video_tensor = preprocess_video('../Celeb-DF/Celeb-real/your_video.mp4')
prediction = model(video_tensor)

probability = prediction.item()

if probability > 0.5:
    print(f"Prediction: Deepfake with probability {probability}")
else:
    print(f"Prediction: Real with probability {1 - probability}")
