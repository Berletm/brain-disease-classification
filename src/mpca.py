import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from typing import Dict
from skimage.transform import resize
from utils import *

class MPCA:
    def __init__(self, max_iter: int, ranks: tuple, projection_mode: str="axial", tol: float = 1e-6):
        self.max_iter = max_iter
        
        self.ranks = np.array(ranks)
        
        self.axial_mat   = None
        self.frontal_mat = None
        self.sagital_mat = None
        
        self.mean_tensor = None        

        self.tol = tol
        self.projection_mode  = projection_mode
        self.initialized = False
        
    @staticmethod
    def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
        return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

    @staticmethod
    def fold(tensor: np.ndarray, mode: int, shape: tuple) -> np.ndarray:
        full_shape = list(shape)
        mode_dim = full_shape.pop(mode)
        full_shape.insert(0, mode_dim)
        return np.moveaxis(np.reshape(tensor, full_shape), 0, mode)
    
    @staticmethod
    def n_mode_prod(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
        res = np.dot(matrix, MPCA.unfold(tensor, mode))

        new_shape = list(tensor.shape)
        new_shape[mode] = matrix.shape[0]

        return MPCA.fold(res, mode, tuple(new_shape))

    def fit_transform(self, dataset: np.ndarray) -> np.ndarray:
        self.fit(dataset)
        return self.transform(dataset)

    def transform(self, dataset: np.ndarray) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Model should be fitted first")
        
        X = dataset
        n = len(dataset)
        mats = [self.axial_mat, self.frontal_mat, self.sagital_mat]
        
        projected_all = []
        for i in range(n):
            proj = X[i]
            if self.projection_mode == "sagital":
                proj = self.n_mode_prod(proj, self.sagital_mat.T, 2)
            elif self.projection_mode == "frontal":
                proj = self.n_mode_prod(proj, self.frontal_mat.T, 1)
            elif self.projection_mode == "axial":
                proj = self.n_mode_prod(proj, self.axial_mat.T, 0)
            elif self.projection_mode == "full":
                for m in range(3):
                    proj = self.n_mode_prod(proj, mats[m].T, m)
            projected_all.append(proj)

        res = np.array(projected_all)
        return res

    def fit(self, dataset: np.ndarray) -> None:
        n, i_dim, j_dim, k_dim = dataset.shape
        r1, r2, r3 = self.ranks
        orig_dims = [i_dim, j_dim, k_dim]
        
        r1, r2, r3 = self.ranks
        # projection matrices U_i = I
        self.axial_mat   = np.eye(i_dim)[:, :r1]
        self.frontal_mat = np.eye(j_dim)[:, :r2]
        self.sagital_mat = np.eye(k_dim)[:, :r3]
        
        mats = [self.axial_mat, self.frontal_mat, self.sagital_mat]

        # data centering
        self.mean_tensor = np.mean(dataset, axis=0)
        X = dataset

        # metric
        prev_total_variance = -1.0
        for it in range(1, self.max_iter + 1):
            for m in range(3):
                others = [idx for idx in range(3) if idx != m]
                covariance_mat = np.zeros((orig_dims[m], orig_dims[m]))

                for i in range(n):
                    temp_proj = X[i]
                    for o in others:
                        temp_proj = self.n_mode_prod(temp_proj, mats[o].T, o)

                    unfolded = self.unfold(temp_proj, m)
                    covariance_mat += unfolded @ unfolded.T

                eig_vals, eig_vecs = np.linalg.eigh(covariance_mat)
                sort_indices = np.argsort(eig_vals)[::-1]
                target_rank = self.ranks[m]
                mats[m] = eig_vecs[:, sort_indices[:target_rank]]
            
            total_variance = 0.0
            for i in range(n):
                proj = X[i]
                for m in range(3):
                    proj = self.n_mode_prod(proj, mats[m].T, m)
                total_variance += np.linalg.norm(proj) ** 2
            print(f"Iter {it}| Total preserved variance: {total_variance:.6f}")
            
            if abs(total_variance - prev_total_variance) < self.tol:
                print(f"Converged at iteration {it}")
                break
            prev_total_variance = total_variance

        self.axial_mat, self.frontal_mat, self.sagital_mat = mats
        self.initialized = True
    
    def load(self, pth: str) -> None:
        data = np.load(pth, allow_pickle=True)
        self.mean_tensor = data["mean_tensor"]
        self.axial_mat, self.frontal_mat, self.sagital_mat = data["axial_mat"], data["frontal_mat"], data["sagital_mat"]
        self.projection_mode = str(data["projection_mode"])
        self.ranks = data["ranks"]
        self.initialized = True
            
    def save(self, pth: str) -> None:
        if not self.initialized:
            raise RuntimeError("Can not save not initialized model")
        np.savez_compressed(pth, 
                            axial_mat=self.axial_mat, 
                            frontal_mat=self.frontal_mat, 
                            sagital_mat=self.sagital_mat,
                            mean_tensor=self.mean_tensor, 
                            ranks=self.ranks, 
                            projection_mode=self.projection_mode)
    
def read_mri(filepath: str) -> np.ndarray:
    data = []

    for file in os.listdir(filepath):
        img = nib.load(os.path.join(filepath, file))
        data.append(img.get_fdata())
    
    return np.array(data)

def normalize_image(img):
    vmin, vmax = np.percentile(img, (2, 98))

    img_norm = (img - vmin) / (vmax - vmin)

    img_norm = np.clip(img_norm, 0, 1)

    return img_norm

def generate_reduced_dataset(data: Dict[str, np.ndarray], plane: str="axial") -> None:
    dataset = list(data.values())
    namings = list(data.keys())

    s = [img.shape for img in dataset]
    _, *min_s = np.min(s, axis=0)
    
    temp = []
    for dset in dataset:
        shifted_dset = dset - np.mean(dset, axis=0)
        for img in shifted_dset:
            new_img = resize(img, min_s, order=1, preserve_range=True, anti_aliasing=True)
            temp.append(new_img)
    
    resized_dataset = np.array(temp)
    resized_dataset = np.array([normalize_image(x) for x in resized_dataset])

    w, h = min_s[:2]
    c = 3
    plane2shape = \
    {
        "sagital": (w, h, c),
        "frontal": (w, c, h),
        "axial"  : (c, w, h)
    }
    plane2axis = \
    {
        "sagital": (0, 1, 2),
        "frontal": (0, 2, 1),
        "axial"  : (1, 2, 0)
    }
    axis  = plane2axis[plane]
    shape = plane2shape[plane]
    
    mpca = MPCA(15, shape, plane)
    reduced_dataset = mpca.fit_transform(resized_dataset)
    
    if not os.path.exists(SAVED_MODELS_PATH):
        os.mkdir(SAVED_MODELS_PATH)
    pth = os.path.join(SAVED_MODELS_PATH, "mpca")
    if not os.path.exists(pth):
        os.mkdir(pth)
    mpca.save(os.path.join(pth, f"{plane}_mpca.npz"))

    sizes = [len(dset) for dset in dataset]
    ranges = []

    l, r = 0, 0
    for i in range(len(sizes)):
        r += sizes[i]
        ranges.append((l, r))
        l = r

    for i, img in enumerate(reduced_dataset):
        img = np.squeeze(img)
        img = np.transpose(img, axis)
        img = normalize_image(img)
        
        name_ind = 0
        for j, k in enumerate(ranges):
            left, right = k
            if left <= i < right:
                name_ind = j
                break

        name = namings[name_ind]
        pth = os.path.join(REDUCED_DATASET_PATH, plane)
        if not os.path.exists(pth):
            os.mkdir(pth)
        plt.imsave(os.path.join(pth, f"{name}_{i}.png"), img)

if __name__ == "__main__":
    parkinson = read_mri(PARKINSON_DATASET_PATH)
    autism    = read_mri(AUTISM_DATASET_PATH)
    control   = read_mri(CONTROL_DATASET_PATH)
    control_ixi = read_mri(CONTROL_IXI_DATASET_PATH)
    alzheimer = read_mri(ALZHEIMER_DATASET_PATH)
    adhd      = read_mri(ADHD_DATASET_PATH)

    namings   = ["parkinson", "control", "control_ixi", "alzheimer", "adhd", "autism"]
    dataset   = [parkinson, control, control_ixi, alzheimer, adhd, autism]

    data = dict(zip(namings, dataset))

    for plane in ["axial", "sagital", "frontal"]:
        print(plane)
        generate_reduced_dataset(data, plane)
        print("\n\n")
