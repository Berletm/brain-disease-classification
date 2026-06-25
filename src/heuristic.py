import numpy as np
import matplotlib.pyplot as plt
import os
from mpca import read_mri, normalize_image
from utils import *

class HeuristicSlicing():
    def __init__(self, n_components: int=3, slicing_axis: str="ax", mode: str="cov", partition: float=False):
        self.n_components = n_components
        self.axis = slicing_axis
        self.mode = mode
        self.partition = partition
        self.slice_indices = None
    
    def transpose_volume(self, volume: np.ndarray) -> np.ndarray:
        if self.axis == "sag":
            return np.transpose(volume, (1, 2, 0))
        elif self.axis == "ax":
            return np.transpose(volume, (0, 1, 2))
        elif self.axis == "front":
            return np.transpose(volume, (0, 2, 1))
        return volume   
        
    def score(self, slice: np.ndarray) -> float:
        if self.mode == "cov":
            cov_mat = (slice.T @ slice) / (slice.shape[0] - 1)
            
            return np.linalg.norm(cov_mat, ord="fro")
        else:
            return np.linalg.norm(slice, ord="fro")
    
    def extract_slices(self, volume: np.ndarray, n_comp: int=3) -> np.ndarray:
        scores = []
        for slice in volume:
            score = self.score(slice)
                
            scores.append(score)
                
        indices = np.argsort(scores)[-n_comp:][::-1] # from most to least informative n slices
        return indices
    
    def fit(self, volume: np.ndarray) -> None:
        volume = self.transpose_volume(volume)
        
        if not self.partition:
            self.slice_indices = self.extract_slices(volume, 3)
        else:
            n = len(volume)
            step = round((1/3) * n)
            
            top_part = (0, step)
            mid_part = (step, 2*step)
            bot_part = (2*step, n)
            
            slices = []
            
            for part in [top_part, mid_part, bot_part]:
                start, stop = part
                
                temp_ind = np.arange(start, stop)
                
                volume_part = volume[temp_ind]
                
                ind = self.extract_slices(volume_part, 1)
                slices.append(ind)
            self.slice_indices = slices
    
    def transform(self, volume: np.ndarray) -> np.ndarray:
        volume = self.transpose_volume(volume)
        return volume[self.slice_indices]
    
    def fit_transform(self, volume: np.ndarray) -> np.ndarray:
        self.fit(volume)
        return self.transform(volume)

def main() -> None:
    parkinson = read_mri(PARKINSON_DATASET_PATH)
    autism    = read_mri(AUTISM_DATASET_PATH)
    control   = read_mri(CONTROL_DATASET_PATH)
    control_ixi = read_mri(CONTROL_IXI_DATASET_PATH)
    alzheimer = read_mri(ALZHEIMER_DATASET_PATH)
    adhd      = read_mri(ADHD_DATASET_PATH)
    sclerosis  = read_mri(SCLEROSIS_DATASET_PATH)

    namings   = ["parkinson", "control", "control_ixi", "alzheimer", "adhd", "autism", "sclerosis"]
    dataset   = [parkinson, control, control_ixi, alzheimer, adhd, autism, sclerosis]

    data = zip(namings, dataset)
    
    model = HeuristicSlicing(partition=True, slicing_axis="ax", mode="cov")
    i = 0
    
    out_dir = "../heuristic_dataset"
    for name, ds in data:
        for vol in ds:
            img = model.fit_transform(vol)
            img = np.squeeze(img)
            img = np.transpose(img, (1, 2, 0))
            img = normalize_image(img)
            
            plt.imsave(os.path.join(out_dir, f"{name}_{i}.png"), img)
            i += 1
    
if __name__ == "__main__":
    main()