

import random

import torch

from utils import perturbe_frame, preprocess_video
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.linear_model import LinearRegression


rows, cols = 5, 5
num_matrix = 30
video_path = r'videos\sport.mp4'
desired_action = 308

# Load the pretrained model
model = r3d_18(weights=R3D_18_Weights.DEFAULT)
model.eval()

pert_matrixs = []
for matrix in range(num_matrix):
    matrix_buffer = []
    for i in range(rows):
        line_buffer = []
        for j in range(cols):
            line_buffer.append(random.randint(0,1) == 1)
        matrix_buffer.append(line_buffer)
    pert_matrixs.append(matrix_buffer)

raw_frames, width, height = preprocess_video(video_path)
preds = []
desired_action_scores = []
for asd, pert in enumerate(pert_matrixs):
    pert_frames = perturbe_frame(raw_frames, pert, cols, rows, width, height, asd)
    #make inference o pert video
    # Perform inference
    with torch.no_grad():
        outputs = model(pert_frames)
        
    # Get the confidence scores
    confidence_scores = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    disered_action_score = confidence_scores[0][desired_action]
    desired_action_scores.append(disered_action_score)

Y_dataset = desired_action_scores
X_dataset = []
for matrix in pert_matrixs:
    buffer = []
    for line in matrix:
        for i in line:
            buffer.append(i)
    X_dataset.append(buffer)


#pred on linear model
simpler_model = LinearRegression()
simpler_model.fit(X=X_dataset, y=Y_dataset)
coeff = simpler_model.coef_
print(coeff)







