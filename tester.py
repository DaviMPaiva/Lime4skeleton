import torch
from lime_3d.lime_3d import VideoPerturbationAnalyzer
from torchvision.models.video import swin3d_t, Swin3D_T_Weights
import torchvision.transforms as transforms
import numpy as np

rows, cols = 5, 5
num_matrix = 20
video_path = r""
desired_action = 

def predict_fn(frames):
    
    # Transform the video frames
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    transformed_frames = []
    for frame in frames:
        transformed_frame = transform(frame)  # Transform each frame individually
        transformed_frames.append(transformed_frame)
    frames = torch.stack(transformed_frames)
    
    # Reshape to (B, C, T, H, W) format
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0)

    frames = frames.to("cuda")
    model.to("cuda")
    with torch.no_grad():
        outputs = model(frames)
    scores = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    return scores[0][desired_action], scores

# Load the pretrained model
model = swin3d_t(weights=Swin3D_T_Weights.DEFAULT)
model.eval()

if __name__ == '__main__':
    # Example usage:
    analyzer = VideoPerturbationAnalyzer()
    analyzer.explain_instance(model_function=predict_fn, video_path=video_path,
                              num_matrix=num_matrix, output_folder="output.mp4")
