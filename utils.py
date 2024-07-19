
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from matplotlib.colors import LinearSegmentedColormap

def preprocess_video(video_path, num_frames=300):
    cap = cv2.VideoCapture(video_path)
    frames = []

    real_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
    
    return frames, 112, 112, real_width, real_height

def tranform_frames(frames):
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

def perturbe_frame(frames, pert_matrix, cols, rows, width, height):
    cell_width = width // cols
    cell_height = height // rows

    pert_frames = []
    for idx, frame in enumerate(frames):
        frame_buf = frame.copy()
        for i in range(cols):
            for j in range(rows):
                if pert_matrix[idx][i][j]:
                    start_x = j * cell_width
                    start_y = i * cell_height
                    end_x = start_x + cell_width
                    end_y = start_y + cell_height
                    frame_buf[start_y:end_y, start_x:end_x] = 0  # Make the cell black

        pert_frames.append(frame_buf)
        #cv2.imwrite(f"aqui_o{asd}_{idx}.jpg", frame_buf)

    return tranform_frames(pert_frames)

def heat_map_over_img(matrix_coeff, height, width, rows, cols):
    # Resize the matrix to the size of the image
    heatmap = np.zeros((height, width)) 
    step_row = math.ceil(height / rows)
    step_col = math.ceil(width / cols)
    for idx_row, row in enumerate(range(0, height, step_row)):
        for idx_col, col in enumerate(range(0, width, step_col)):
            value = matrix_coeff[idx_row*cols + idx_col] 
            heatmap[row:row+step_row, col:col+step_col] = value if value > 0.8 else 0

    # Create a custom colormap (low values red, high values green)
    colors = [(1, 0, 0), (1, 1, 0), (0, 1, 0)]  # Red -> Yellow -> Green
    n_bins = 100  # Discretize the colormap into 100 bins
    cmap_name = 'red_yellow_green'
    cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)

    # Apply colormap
    heatmap_colored = cmap(heatmap)[:, :, :3]  # Apply the colormap
    # Convert the normalized heatmap to a color map
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]  # Use the 'jet' colormap

    # Convert to uint8
    heatmap_colored = np.uint8(255 * heatmap_colored)

    return heatmap_colored
