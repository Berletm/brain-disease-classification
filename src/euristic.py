import numpy as np

class HeuristicSlicing():
    def __init__(self, n_components: int=3, slicing_axis: str="axial", mode: str="cov", partition: float=False):
        self.n_components = n_components
        self.axis = slicing_axis
        self.mode = mode
        self.partition = partition
        self.slice_indices = None
    
    def transpose_volume(self, volume: np.ndarray) -> np.ndarray:
        pass
    
    def calculate_score(self, slice: np.ndarray) -> float:
        if self.mode == "cov":
            cov_mat = (slice.T @ slice) / (slice.shape[0] - 1)
            
            return np.linalg.norm(cov_mat, ord="fro")
        else:
            
            return np.linalg.norm(slice, ord="fro")
    
    def fit(self, volume: np.ndarray) -> None:
        scores = []
        
        if not self.partition:
            for slice in volume:
                score = self.calculate_score(slice)
                
                scores.append(score)
                
            indices = np.argsort(scores)[-3:][::-1] # from most to least informative 3 slices
            self.slice_indices = indices
        else:
            pass
        
    
    def transform(self, volume: np.ndarray) -> np.ndarray:
        pass
    
    def fit_transform(self, volume: np.ndarray) -> np.ndarray:
        pass

def main() -> None:
    pass


if __name__ == "__main__":
    main()