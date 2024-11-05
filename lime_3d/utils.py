
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
        avg_color = np.mean(frame_buf, axis=(0, 1)).astype(int)
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

def proof_of_concept_video(raw_frames, matrix_coeff, height, width, rows, cols, threshold, video_path):
    cell_width = width // cols
    cell_height = height // rows

    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_folder.mp4', fourcc, 10.0, (width, height))

    pert_frames = []
    for idx in range(len(raw_frames)):
        ret, frame = cap.read()
        if not ret:
            break
        frame_buf = frame.copy()
        avg_color = np.mean(frame_buf, axis=(0, 1)).astype(int)
        frame_buf = cv2.resize(frame_buf, (width, height))
        for i in range(cols):
            for j in range(rows):
                idx_coeff = int(idx/100)
                if matrix_coeff[i + j*cols + idx_coeff*cols*rows] < threshold:
                    start_x = j * cell_width
                    start_y = i * cell_height
                    end_x = start_x + cell_width
                    end_y = start_y + cell_height
                    frame_buf[start_y:end_y, start_x:end_x] = (0,0,0)  

        pert_frames.append(frame_buf)
        out.write(frame_buf)
        # cv2.imwrite(f"frames/aqui_o_{idx}.jpg", frame_buf)
    out.release()
    cap.release()

    return pert_frames
