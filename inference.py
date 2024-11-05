import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
import cv2
import numpy as np
from torchvision.models.video import swin3d_t, Swin3D_T_Weights

from lime_3d.utils import preprocess_video
from tester import predict_fn

# Load the pretrained model
model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
model.eval()

# Load and preprocess the input video
video_path = r"selected_videos\-Cfav7t1B3I_000002_000012.mp4"
frames, _ ,_ = preprocess_video(video_path)
_, outputs = predict_fn(frames)

top_scores, top_labels = torch.topk(torch.from_numpy(outputs), 5, dim=1)

# Load the Kinetics-400 action names
def load_kinetics_labels(label_file):
    with open(label_file, 'r') as f:
        kinetics_labels = [line.strip() for line in f.readlines()]
    return kinetics_labels

kinetics_labels = load_kinetics_labels('kinetics_400_labels.csv')

# Print the top-5 predictions with confidence scores
for i in range(top_scores.size(1)):
    action_label = kinetics_labels[top_labels[0, i].item() + 1]
    confidence = top_scores[0, i].item()
    print(f"Action: {action_label}, Confidence: {confidence:.4f}")
