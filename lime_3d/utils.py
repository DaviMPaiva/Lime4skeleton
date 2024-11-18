
import math
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
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
        
        frames.append(frame)
    
    cap.release()
    
    # Ensure we have the correct number of frames
    if len(frames) < num_frames:
        frames = frames + [frames[-1]] * (num_frames - len(frames))
    
    return frames, real_width, real_height

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
                    frame_buf[start_y:end_y, start_x:end_x] = (0,0,0)  

        pert_frames.append(frame_buf)
        #cv2.imwrite(f"aqui_o{asd}_{idx}.jpg", frame_buf)

    return pert_frames

def heat_map_over_video(raw_frames, matrix_coeff, height, width, rows, cols):
    heatmap = np.zeros((height, width)) 
    cell_row_size = height // rows
    cell_col_size = width // cols
    normalized_heatmaps = []
    global_min = min(heatmap for heatmap in matrix_coeff)
    global_max = max(heatmap for heatmap in matrix_coeff)

    for idx in range(len(raw_frames)):
        # Resize the matrix to the size of the image
        for i in range(rows):
            for j in range(cols):
                idx_coeff = int(idx/100)
                value = matrix_coeff[i + j*cols + idx_coeff*cols*rows] 
                heatmap[cell_row_size*i : (cell_row_size*i)+cell_row_size, 
                        cell_col_size*j : (cell_col_size*j)+cell_col_size] = value 

        # Normalize the heatmap to the range [0, 255] based on global min and max
        # heatmap_normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        heatmap_normalized = ((heatmap - global_min) / (global_max - global_min)) * 255
        heatmap_uint8 = heatmap_normalized.astype(np.uint8)

        # Apply color map if needed
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        normalized_heatmaps.append(heatmap_color)

    return normalized_heatmaps

def proof_of_concept_video(raw_frames, coeff, percentile, masks, masks_activation, segments, video_path):

    threshold = 1.5
    final_mask = np.zeros_like(segments, dtype=bool)
    for idx, value in enumerate(coeff):
        print(value < threshold * segments[segments == idx].mean())
        if value < percentile:
            final_mask |= (segments == idx)

    # frames_3d = np.stack(raw_frames, axis=0) 
    # pertubated_video = frames_3d.copy()  # Make a copy to preserve the original data
    # pertubated_video[final_mask] = [0, 0, 0]

    # Stack frames to create a 3D array
    frames_3d = np.stack(raw_frames, axis=0)
    pertubated_video = frames_3d.copy()  # Make a copy to preserve the original data

    # Define the darkening factor (e.g., 0.5 will make it 50% darker)
    darkening_factor = 0.2

    # Apply the darkening effect to the regions defined by `final_mask`
    pertubated_video[final_mask] = (pertubated_video[final_mask] * darkening_factor).astype(np.uint8)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_height, frame_width = raw_frames[0].shape[:2]
    out = cv2.VideoWriter(video_path, fourcc, 10.0, (frame_width, frame_height))
    pert_frames = []
    for pertubated_frame in pertubated_video:
        out.write(pertubated_frame)
    out.release()

    return pert_frames
