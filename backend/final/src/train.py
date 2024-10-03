import os
from transformers import VideoClassificationModel, Trainer, TrainingArguments
import torch
from datasets import load_dataset

from preprocess import preprocess_video, get_video_paths

# Define a dataset class
class CelebDFDataset(torch.utils.data.Dataset):
    def __init__(self, video_paths, labels):
        self.video_paths = video_paths
        self.labels = labels

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        video_tensor = preprocess_video(video_path)
        label = self.labels[idx]
        return video_tensor, label

# Create a list of video paths and their labels
data_folder = '../Celeb-DF-v2/'
real_video_paths = get_video_paths(os.path.join(data_folder, 'Celeb-real')) + get_video_paths(os.path.join(data_folder, 'YouTube-real'))
synthesized_video_paths = get_video_paths(os.path.join(data_folder, 'Celeb-synthesis'))

video_paths = real_video_paths + synthesized_video_paths
labels = [0] * len(real_video_paths) + [1] * len(synthesized_video_paths)  # 0 for real, 1 for deepfake

# Create dataset object
dataset = CelebDFDataset(video_paths, labels)

# Split into train and test (assuming you've already defined this split in List_of_testing_videos.txt)
train_dataset = dataset[:int(0.8 * len(dataset))]  # 80% train
test_dataset = dataset[int(0.8 * len(dataset)):]   # 20% test

# Load model
model = VideoClassificationModel.from_pretrained('facebook/timesformer-base-finetuned')

# Modify classifier for binary classification
model.classifier = torch.nn.Sequential(
    torch.nn.Linear(in_features=768, out_features=256),
    torch.nn.ReLU(),
    torch.nn.Linear(256, 1),
    torch.nn.Sigmoid()
)

# Set training arguments
training_args = TrainingArguments(
    output_dir="../results",
    per_device_train_batch_size=4,
    evaluation_strategy="epoch",
    num_train_epochs=3,
    logging_dir="../logs",
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# Train the model
trainer.train()
model.save_pretrained('../model/your_model_directory')  # Save the trained model
