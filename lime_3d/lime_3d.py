import numpy as np
import random
import torch
import cv2
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from lime_3d.utils import heat_map_over_img, perturbe_frame, preprocess_video

class VideoPerturbationAnalyzer:
    def __init__(self, output_folder, num_matrix, rows, cols):
        self.num_matrix = num_matrix
        self.rows = rows
        self.cols = cols
        self.output_folder = output_folder
        self.simple_model = LinearRegression()

    def explain_instance(self, model_function, tranform_frames_function, desired_action, video_path):
        all_matrices = self._generate_perturbed_matrices()
        raw_frames, width, height, real_width, real_height = self._preprocess_video(video_path)
        X_dataset, Y_dataset = self._generate_dataset(model_function, tranform_frames_function, desired_action, 
                                                      all_matrices, raw_frames, width, height)
        coeff = self._train_model(X_dataset, Y_dataset)
        heat_maps = self._generate_heatmaps(coeff, real_height, real_width)
        self._create_output_video(heat_maps, real_width, real_height, video_path)

    def _generate_perturbed_matrices(self):
        all_matrices = []
        for _ in range(self.num_matrix):
            pert_matrixs = []
            for _ in range(300):
                matrix_buffer = []
                for _ in range(self.rows):
                    line_buffer = [random.randint(0, 1) == 1 for _ in range(self.cols)]
                    matrix_buffer.append(line_buffer)
                pert_matrixs.append(matrix_buffer)
            all_matrices.append(pert_matrixs)

        return all_matrices

    def _preprocess_video(self, video_path):
        return preprocess_video(video_path)

    def _generate_dataset(self, model_function, tranform_frames_function, 
                          desired_action, all_matrices, raw_frames, width, height):
        desired_action_scores = []
        for pert in tqdm(all_matrices):
            pert_frames = perturbe_frame(raw_frames, pert, tranform_frames_function, 
                                         self.cols, self.rows, width, height)
            confidence_scores = model_function(pert_frames)
            desired_action_score = confidence_scores[0][desired_action]
            desired_action_scores.append(desired_action_score)

        Y_dataset = desired_action_scores
        X_dataset = [self._flatten_matrix(matrix_dect) for matrix_dect in all_matrices]
        return X_dataset, Y_dataset
    
    def _train_model(self, X, y):
        self.simple_model.fit(X=X,y=y)
        coeff = self.simple_model.coef_
        return (coeff - np.min(coeff)) / (np.max(coeff) - np.min(coeff))

    def _flatten_matrix(self, matrix_dect):
        buffer = [i for matrix in matrix_dect for line in matrix for i in line]
        return buffer

    def _train_linear_model(self):
        simpler_model = LinearRegression()
        simpler_model.fit(X=self.X_dataset, y=self.Y_dataset)
        self.coeff = simpler_model.coef_
        self.coeff = (self.coeff - np.min(self.coeff)) / (np.max(self.coeff) - np.min(self.coeff))
        self.generate_heatmaps()

    def _generate_heatmaps(self, coeff, real_height, real_width):
        heat_maps = []
        for coef_idx in tqdm(range(0, len(coeff), self.rows * self.cols)):
            coeff_matrix = coeff[coef_idx:coef_idx + self.rows * self.cols]
            heat_maps.append(heat_map_over_img(coeff_matrix, real_height, real_width, self.rows, self.cols))

        return heat_maps

    def _create_output_video(self, heat_maps, real_width, real_height, video_path):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(self.output_folder, fourcc, 10.0, (real_width, real_height))
        cap = cv2.VideoCapture(video_path)

        idx = 0
        while cap.isOpened() and idx < len(heat_maps):
            ret, frame = cap.read()
            if not ret:
                break
            overlay = cv2.addWeighted(frame, 0.7, heat_maps[idx], 0.3, 0)
            idx += 1
            out.write(overlay)
        cap.release()
        out.release()



