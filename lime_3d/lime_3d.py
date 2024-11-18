import numpy as np
import random
import sklearn
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import torch
import cv2
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from skimage import segmentation

from lime_3d.utils import heat_map_over_video, perturbe_frame, preprocess_video, proof_of_concept_video

class VideoPerturbationAnalyzer:
    def __init__(self):
        self.simple_model = LinearRegression()

    def explain_instance(self, model_function, video_path, num_matrix, output_folder):
        self.output_folder = output_folder
        self.num_matrix = num_matrix
        raw_frames, real_width, real_height = self._preprocess_video(video_path)
        masks, masks_activation, segments = self._generate_repeted_perturbed_matrices(raw_frames)
        X_dataset, Y_dataset = self._generate_dataset(model_function, raw_frames, masks, masks_activation)
        coeff = self._train_model(X_dataset, Y_dataset)
        # heat_maps = self._generate_heatmaps(raw_frames, coeff, real_height, real_width)
        self._create_proof_of_concept_video(raw_frames, coeff, masks, masks_activation, segments)
        # self._create_output_video(heat_maps, real_width, real_height, video_path)

    def _generate_repeted_perturbed_matrices(self, raw_frames):
        frames_3d = np.stack(raw_frames, axis=0) 
        print("segmenting img")
        segments = segmentation.slic(frames_3d, n_segments=20, compactness=20)

        cluster_size = len(np.unique(segments))
        all_masks = []
        all_masks_activation = []
        for _ in range(self.num_matrix):
            masks_activation = []
            mask = np.zeros_like(segments, dtype=bool)
            for idx in range(cluster_size):
                is_black = random.randint(0, 1) == 1
                masks_activation.append(is_black)
                if is_black:
                    mask |= (segments == idx)

            all_masks_activation.append(masks_activation)
            all_masks.append(mask)

        return all_masks, all_masks_activation, segments

    def _preprocess_video(self, video_path):
        return preprocess_video(video_path)

    def _generate_dataset(self, model_function, raw_frames, masks, mask_activation):
        desired_action_scores = []
        base_confidence_score, _ = model_function(raw_frames)
        print(base_confidence_score)
        frames_3d = np.stack(raw_frames, axis=0) 
        for mask in tqdm(masks):
            pertubated_video = frames_3d.copy()  # Make a copy to preserve the original data
            pertubated_video[mask] = [0, 0, 0]
            # pert_frames = perturbe_frame(raw_frames, pert, self.cols, self.rows, width, height)
            confidence_score, _ = model_function(pertubated_video)
            print(confidence_score)
            desired_action_scores.append(confidence_score)

        Y_dataset = desired_action_scores
        X_dataset = mask_activation
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
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # self.simple_model.fit(X=X_train, y=y_train)
        # y_pred = self.simple_model.predict(X_test)
        # mae = mean_absolute_error(y_test, y_pred)
        # print("Mean Absolute Error (MAE):", mae)

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

    def _create_proof_of_concept_video(self, raw_frames, coeff, masks, masks_activation, segments):
        percentile = np.percentile(coeff, 80)
        proof_of_concept_video(raw_frames, coeff, percentile, masks, masks_activation, segments, self.output_folder)
        



