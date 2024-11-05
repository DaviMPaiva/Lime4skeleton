import numpy as np
import random
import sklearn
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import cv2
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

from lime_3d.utils import heat_map_over_video, perturbe_frame, preprocess_video, proof_of_concept_video

class VideoPerturbationAnalyzer:
    def __init__(self, output_folder, num_matrix, rows, cols):
        self.num_matrix = num_matrix
        self.rows = rows
        self.cols = cols
        self.output_folder = output_folder
        self.simple_model = LinearRegression()

    def explain_instance(self, model_function, desired_action, video_path):
        raw_frames, real_width, real_height = self._preprocess_video(video_path)
        all_matrices = self._generate_repeted_perturbed_matrices(raw_frames)
        X_dataset, Y_dataset = self._generate_dataset(model_function, desired_action, 
                                                      all_matrices, raw_frames, real_width, real_height )
        coeff = self._train_model(X_dataset, Y_dataset)
        heat_maps = self._generate_heatmaps(raw_frames, coeff, real_height, real_width)
        self._create_proof_of_concept_video(raw_frames, coeff, real_height, real_width, video_path)
        self._create_output_video(heat_maps, real_width, real_height, video_path)

    def _generate_perturbed_matrices(self, raw_frames):
        all_matrices = []
        for _ in range(self.num_matrix):
            pert_matrixs = []
            for _ in range(len(raw_frames)):
                matrix_buffer = []
                for _ in range(self.rows):
                    line_buffer = [random.randint(0, 10) == 0 for _ in range(self.cols)]
                    matrix_buffer.append(line_buffer)
                pert_matrixs.append(matrix_buffer)
            all_matrices.append(pert_matrixs)

        return all_matrices
    
    def _help_generate_set(self):
        matrix_buffer = []
        for _ in range(self.rows):
            line_buffer = [random.randint(0, 10) < 6  for _ in range(self.cols)]
            matrix_buffer.append(line_buffer)
        return matrix_buffer

    def _generate_repeted_perturbed_matrices(self, raw_frames):
        all_matrices = []
        for _ in range(self.num_matrix):
            matrix_buffer = []
            pert_matrixs = []
            for idx in range(len(raw_frames)):
                if idx%int(len(raw_frames)/100) == 0:
                    matrix_buffer = self._help_generate_set()
                pert_matrixs.append(matrix_buffer)

            all_matrices.append(pert_matrixs)

        return all_matrices

    def _preprocess_video(self, video_path):
        return preprocess_video(video_path)

    def _generate_dataset(self, model_function, desired_action, all_matrices,
                           raw_frames, width, height):
        desired_action_scores = []
        base_confidence_score, _ = model_function(raw_frames)
        print(base_confidence_score)
        for pert in tqdm(all_matrices):
            pert_frames = perturbe_frame(raw_frames, pert, self.cols, self.rows, width, height)
            confidence_score, _ = model_function(pert_frames)
            print(confidence_score)
            desired_action_scores.append(confidence_score)

        Y_dataset = desired_action_scores
        X_dataset = [self._flatten_matrix(matrix_dect) for matrix_dect in all_matrices]
        # # Transformar o tensor PyTorch em um array numpy
        # pert_frames_np = np.array(crude_pert_frames_list).reshape(len(crude_pert_frames_list), -1)

        # # Transformar a lista raw_frames em um array numpy
        # raw_frames_np = np.array(raw_frames).reshape(1, -1)

        # # Calcular a distância de cosseno
        # distances = sklearn.metrics.pairwise_distances(pert_frames_np, raw_frames_np, metric='cosine').ravel()
        # kernel_width = 0.25
        # weights = np.sqrt(np.exp(-(distances**2)/kernel_width**2)) #Kernel function
        # weights.shape
        return X_dataset, Y_dataset
    
    def _train_model(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        self.simple_model.fit(X=X_train, y=y_train)
        y_pred = self.simple_model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        print("Mean Absolute Error (MAE):", mae)

        self.simple_model.fit(X=X,y=y)
        coeff = self.simple_model.coef_
        values = (coeff - np.min(coeff)) / (np.max(coeff) - np.min(coeff))
        return values

    def _flatten_matrix(self, matrix_dect):
        # [i for matrix in matrix_dect for line in matrix for i in line]
        flattened_list = []
        for matrix_idx in range(0, len(matrix_dect), 100) :
            for line in matrix_dect[matrix_idx]:
                for i in line:
                    flattened_list.append(i)

        return flattened_list

    def _train_linear_model(self):
        self.simpler_model.fit(X=self.X_dataset, y=self.Y_dataset)
        coeff = self.simpler_model.coef_
        self.coeff = (coeff - np.min(coeff)) / (np.max(coeff) - np.min(coeff))
        self.generate_heatmaps()

    def _generate_heatmaps(self, raw_frames, coeff, real_height, real_width):
        heat_maps = heat_map_over_video(raw_frames, coeff, real_height, real_width, self.rows, self.cols)
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

    def _create_proof_of_concept_video(self, raw_frames, coeff, real_height, real_width, video_path):
        percentile = np.percentile(coeff, 80)
        proof_of_concept_video(raw_frames, coeff, real_height, real_width, self.rows, self.cols, percentile, video_path)
        



