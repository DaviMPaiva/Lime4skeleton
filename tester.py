import torch
from lime_3d import VideoPerturbationAnalyzer
from torchvision.models.video import r3d_18, R3D_18_Weights


rows, cols = 5, 5
num_matrix = 30
video_path = r'videos\riding.mp4'
desired_action = 273

def predict_fn(video):
    with torch.no_grad():
        outputs = model(video)
    return torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()

# Load the pretrained model
model = r3d_18(weights=R3D_18_Weights.DEFAULT)
model.eval()
# Example usage:
analyzer = VideoPerturbationAnalyzer('out_video.mp4', num_matrix, rows, cols)
analyzer.explain_instance(model_function=predict_fn,
                          desired_action=141,
                          video_path=video_path)