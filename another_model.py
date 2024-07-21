from PIL import Image
import torch
from torchvision import models

# Load the pretrained I3D model
model = models.video.r3d_18(pretrained=True)

# Set the model to evaluation mode
model.eval()

import cv2
import numpy as np
from torchvision import transforms

# Define the preprocessing transforms
preprocess = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to load video and apply preprocessing
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = Image.fromarray(frame)
        frame = preprocess(frame)
        frames.append(frame)
    cap.release()
    
    # Stack frames to create a tensor of shape (C, T, H, W)
    video_tensor = torch.stack(frames, dim=1)
    return video_tensor

video_path = r'videos\riding.mp4'
video_tensor = preprocess_video(video_path)

# Add a batch dimension
video_tensor = video_tensor.unsqueeze(0)

# Make predictions
with torch.no_grad():
    outputs = model(video_tensor)

# Get the predicted class
_, predicted = torch.max(outputs, 1)
print(f'Predicted class: {predicted.item()}')

# Load Kinetics-400 class labels
KINETICS_400_LABELS = r'kinetics_400_labels.csv'

def load_labels(label_file):
    with open(label_file, 'r') as f:
        labels = [line.strip() for line in f.readlines()]
    return labels

labels = load_labels(KINETICS_400_LABELS)
predicted_label = labels[predicted.item()]
print(f'Predicted action: {predicted_label}')



