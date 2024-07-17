
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

def preprocess_video(video_path, num_frames=16):
    cap = cv2.VideoCapture(video_path)
    frames = []

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
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
    
    return frames, 112, 112

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

def perturbe_frame(frames, pert_matrix, cols, rows, width, height, asd):
    cell_width = width // cols
    cell_height = height // rows

    pert_frames = []
    for idx, frame in enumerate(frames):
        frame_buf = frame.copy()
        for i in range(cols):
            for j in range(rows):
                if pert_matrix[i][j]:
                    start_x = j * cell_width
                    start_y = i * cell_height
                    end_x = start_x + cell_width
                    end_y = start_y + cell_height
                    frame_buf[start_y:end_y, start_x:end_x] = 0  # Make the cell black

        pert_frames.append(frame_buf)
    cv2.imwrite(f"aqui_o{asd}.jpg", frame_buf)

    return tranform_frames(pert_frames)
