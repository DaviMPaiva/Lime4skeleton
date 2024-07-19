

import random

import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from utils import heat_map_over_img, perturbe_frame, preprocess_video
from torchvision.models.video import r3d_18, R3D_18_Weights
from sklearn.linear_model import LinearRegression


rows, cols = 5, 5
num_matrix = 30
video_path = r'videos\golf.mp4'
desired_action = 141

# Load the pretrained model
model = r3d_18(weights=R3D_18_Weights.DEFAULT)
model.eval()

all_matrixs = []
for _ in range(num_matrix):
    pert_matrixs = []
    for matrix in range(300):
        matrix_buffer = []
        for i in range(rows):
            line_buffer = []
            for j in range(cols):
                line_buffer.append(random.randint(0,1) == 1)
            matrix_buffer.append(line_buffer)
        pert_matrixs.append(matrix_buffer)
    all_matrixs.append(pert_matrixs)

raw_frames, width, height, real_width, real_height = preprocess_video(video_path)
preds = []
desired_action_scores = []
for asd, pert in tqdm(enumerate(all_matrixs)):
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
for matrix_dect in all_matrixs:
    buffer = []
    for matrix in matrix_dect:
        for line in matrix:
            for i in line:
                buffer.append(i)
    X_dataset.append(buffer)


#pred on linear model
simpler_model = LinearRegression()
simpler_model.fit(X=X_dataset, y=Y_dataset)
coeff = simpler_model.coef_
coeff = (coeff - np.min(coeff)) / (np.max(coeff) - np.min(coeff))

heat_maps = []
#make heat map for each coeficient matrix
for coef_idx in tqdm(range(0, len(coeff), rows*cols)):
    coeff_matrix = coeff[coef_idx : coef_idx + rows*cols]
    heat_maps.append(heat_map_over_img(coeff_matrix, real_height, real_width, rows, cols))

# Define the codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output_video.mp4', fourcc, 20.0, (real_width, real_height))

# Load the video
cap = cv2.VideoCapture(video_path)

cell_width = width // cols
cell_height = height // rows

idx = 0 
while cap.isOpened() and idx < len(heat_maps):
    ret, frame = cap.read()
    if not ret:
        break

    # Overlay the heatmap on the image
    overlay = cv2.addWeighted(frame, 0.7, heat_maps[idx], 0.3, 0)
    idx += 1

    # Write the frame to the output video
    out.write(overlay)
    #cv2.imwrite(f"aqui_o{idx}.jpg", overlay)

# Release everything when done
cap.release()
out.release()
cv2.destroyAllWindows()
