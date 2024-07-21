import torch
import torchvision.transforms as transforms
from torchvision.models.video import r3d_18
import cv2
import numpy as np
from torchvision.models.video import R3D_18_Weights

# Load the pretrained model
model = r3d_18(weights=R3D_18_Weights.DEFAULT)
model.eval()

# Function to preprocess the input video
def preprocess_video(video_path, num_frames=300):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened() and len(frames) < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (112, 112))
        frames.append(frame)
    
    cap.release()
    
    # Ensure we have the correct number of frames
    if len(frames) < num_frames:
        frames = frames + [frames[-1]] * (num_frames - len(frames))
    frames = np.array(frames)
    
    # Transform the video frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.225, 0.225, 0.225]),
    ])
    
    frames = [transform(frame) for frame in frames]
    frames = torch.stack(frames)
    
    # Reshape to (B, C, T, H, W) format
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)
    
    return frames

# Load and preprocess the input video
video_path = r"videos\riding.mp4"
input_frames = preprocess_video(video_path)

# Perform inference
with torch.no_grad():
    outputs = model(input_frames)
    
# Get the confidence scores
confidence_scores = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
top_scores, top_labels = torch.topk(torch.from_numpy(confidence_scores), 5, dim=1)

# Load the Kinetics-400 action names
def load_kinetics_labels(label_file):
    with open(label_file, 'r') as f:
        kinetics_labels = [line.strip() for line in f.readlines()]
    return kinetics_labels

kinetics_labels = load_kinetics_labels('kinetics_400_labels.csv')

# Print the top-5 predictions with confidence scores
for i in range(top_scores.size(1)):
    action_label = kinetics_labels[top_labels[0, i].item()]
    confidence = top_scores[0, i].item()
    print(f"Action: {action_label}, Confidence: {confidence:.4f}")
