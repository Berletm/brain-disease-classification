import numpy as np
import nibabel as nib
import os
import matplotlib.pyplot as plt
from typing import Dict
from skimage.transform import resize
from utils import *

class MPCA:
    def __init__(self, kernel: str = "linear", degree:int = 2, max_iter: int = 15, ranks: tuple = (), projection_mode: str="axial", tol: float = 1e-6):
        self.kernel = kernel
        self.max_iter = max_iter
        self.degree = degree
        
        self.ranks = np.array(ranks)
        
        self.axial_mat   = None
        self.frontal_mat = None
        self.sagital_mat = None
        self.orig_shape  = None
        
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
    
    def n_mode_prod(self, tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
        res = matrix @ MPCA.unfold(tensor, mode)
            
        new_shape = list(tensor.shape)
        new_shape[mode] = matrix.shape[0]

        return MPCA.fold(res, mode, tuple(new_shape))

    def fit_transform(self, dataset: np.ndarray) -> np.ndarray:
        self.fit(dataset)
        return self.transform(dataset)

    def transform(self, dataset: np.ndarray) -> np.ndarray:
        if not self.initialized:
            raise RuntimeError("Model should be fitted first")
        
        n = len(dataset)
        mats = [self.axial_mat, self.frontal_mat, self.sagital_mat]
        
        projected_all = []
        for i in range(n):
            proj = dataset[i]
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
    
    def __inner_prod(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        dots = lhs @ rhs
        
        if self.kernel == "linear":
            return dots
        
        elif self.kernel == "poly":
            return (dots + 1) ** self.degree
        
        elif self.kernel == "sigmoid":
            return np.tanh(dots)
        
        elif self.kernel == "rbf":
            lhs_norm = np.sum(lhs ** 2, axis=1)
            rhs_norm = np.sum(rhs ** 2, axis=0)
            
            dist_sq = lhs_norm[:, np.newaxis] + rhs_norm[np.newaxis, :] - 2 * (dots)
        
            dist_sq = np.maximum(dist_sq, 0.0)
        
            return np.exp(-dist_sq)
        
        elif self.kernel == "cosine":
            lhs_norm = np.sqrt(np.sum(lhs ** 2, axis=1))
            rhs_norm = np.sqrt(np.sum(rhs ** 2, axis=0))
            norms = np.maximum(lhs_norm * rhs_norm, 1e-10) 
            
            return dots / norms
        
        else:
            return dots
        
    def fit(self, dataset: np.ndarray) -> None:
        n, i_dim, j_dim, k_dim = dataset.shape
        r1, r2, r3 = self.ranks
        orig_dims = [i_dim, j_dim, k_dim]
        
        self.orig_shape = tuple(orig_dims)
        
        r1, r2, r3 = self.ranks
        # projection matrices U_i = I
        self.axial_mat   = np.eye(i_dim)[:, :r1]
        self.frontal_mat = np.eye(j_dim)[:, :r2]
        self.sagital_mat = np.eye(k_dim)[:, :r3]
        
        mats = [self.axial_mat, self.frontal_mat, self.sagital_mat]

        original_variance = sum(
           np.linalg.norm(dataset[i]) ** 2 
           for i in range(n)
        )

        # metric
        prev_total_variance = -1.0
        for it in range(1, self.max_iter + 1):
            for m in range(3):
                others = [idx for idx in range(3) if idx != m]
                covariance_mat = np.zeros((orig_dims[m], orig_dims[m]))

                for i in range(n):
                    temp_proj = dataset[i]
                    for o in others:
                        temp_proj = self.n_mode_prod(temp_proj, mats[o].T, o)

                    unfolded = self.unfold(temp_proj, m)
                    covariance_mat += self.__inner_prod(unfolded, unfolded.T)

                eig_vals, eig_vecs = np.linalg.eigh(covariance_mat)
                sort_indices = np.argsort(eig_vals)[::-1]
                target_rank = self.ranks[m]
                mats[m] = eig_vecs[:, sort_indices[:target_rank]]
            
            preserved_variance = 0.0
            for i in range(n):
                proj = dataset[i]
                for m in range(3):
                    proj = self.n_mode_prod(proj, mats[m].T, m)
                preserved_variance += np.linalg.norm(proj) ** 2                
                
            print(f"Iter {it}| Total preserved variance ratio: {preserved_variance / original_variance:.6f}")
            
            if abs(preserved_variance - prev_total_variance) < self.tol:
                print(f"Converged at iteration {it}")
                break
            prev_total_variance = preserved_variance

        self.axial_mat, self.frontal_mat, self.sagital_mat = mats
        self.initialized = True
    
    def load(self, pth: str) -> None:
        data = np.load(pth, allow_pickle=True)
        self.axial_mat, self.frontal_mat, self.sagital_mat = data["axial_mat"], data["frontal_mat"], data["sagital_mat"]
        self.projection_mode = str(data["projection_mode"])
        self.ranks = data["ranks"]
        self.orig_shape = tuple(data["orig_shape"])
        self.initialized = True

            
    def save(self, pth: str) -> None:
        if not self.initialized:
            raise RuntimeError("Can not save not initialized model")
        np.savez_compressed(pth, 
                            axial_mat=self.axial_mat, 
                            frontal_mat=self.frontal_mat, 
                            sagital_mat=self.sagital_mat,
                            ranks=self.ranks, 
                            projection_mode=self.projection_mode,
                            orig_shape=self.orig_shape)    
    
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

def unify_dataset(data: np.ndarray, shape: tuple=None) -> np.ndarray:
    if shape is None:
        s = [img.shape for img in data]
        _, *min_s = np.min(s, axis=0)
    else:
        min_s = shape
    
    temp = []
    for subset in data:
        for img in subset:
            new_img = resize(img, min_s, order=3, preserve_range=True, anti_aliasing=True)
            temp.append(new_img)
    
    resized_dataset = np.array(temp)
    resized_dataset = np.array([normalize_image(x) for x in resized_dataset])
    
    mean_tensor = np.mean(resized_dataset, axis=0)
    resized_dataset -= mean_tensor
    
    return resized_dataset

def generate_reduced_dataset(kernel: str, data: Dict[str, np.ndarray], plane: str="axial", precomputed: bool = False, other: tuple = None) -> None:
    dataset = tuple(data.values())
    namings = list(data.keys())
    unified_dataset, ranges = other
    
    w, h = unified_dataset[0].shape[:2]
    c = 3
        
    mpca_pth = os.path.join(SAVED_MODELS_PATH, "mpca", f"{plane}_mpca.npz")
    
    if not os.path.exists(os.path.dirname(mpca_pth)):
        os.makedirs(os.path.dirname(mpca_pth))
        
    if not precomputed:
        plane2shape = {"sagital": (w, h, c), "frontal": (w, c, h), "axial"  : (c, w, h)}
        shape = plane2shape[plane]
        
        mpca = MPCA(kernel=kernel, degree=3, max_iter=15, ranks=shape, projection_mode=plane)
        
        reduced_dataset = mpca.fit_transform(unified_dataset)
            
        mpca.save(mpca_pth)
    else:
        mpca = MPCA(15)
        mpca.load(mpca_pth)
        
        unified_dataset, _, _ = unify_dataset(dataset, shape=mpca.orig_shape)
        reduced_dataset = mpca.transform(unified_dataset)
        
    out_dir = os.path.join(REDUCED_DATASET_PATH, plane)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plane2axis  = {"sagital": (0, 1, 2), "frontal": (0, 2, 1), "axial"  : (1, 2, 0)}
    axis  = plane2axis[plane]
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
        plt.imsave(os.path.join(out_dir, f"{name}_{i}.png"), img)

def calculate_ranges(dataset: list[np.ndarray]) -> list:
    sizes  = [len(dset) for dset in dataset]
    ranges = []

    l, r = 0, 0
    for s in sizes:
        r += s
        ranges.append((l, r))
        l = r
        
    return ranges

if __name__ == "__main__":
    dataset = []
    namings = []
    
    for pth in [PARKINSON_DATASET_PATH, AUTISM_DATASET_PATH, OLD_ABIDE_CONTROL_DATASET_PATH, ABIDE_CONTROL_DATASET_PATH, IXI_CONTROL_DATASET_PATH, ALZHEIMER_DATASET_PATH, ADHD_DATASET_PATH, SCLEROSIS_DATASET_PATH]:
        dataset.append(read_mri(pth))
        d = pth.split("/")[-1]
        namings.append(d)
        
    unified_dataset = unify_dataset(dataset)
    ranges = calculate_ranges(dataset)
    
    data = dict(zip(namings, dataset))
    other = (unified_dataset, ranges)
    for plane in ["axial", "sagital", "frontal"]:
        print(plane)
        generate_reduced_dataset("linear", data, plane, precomputed=False, other = other)
        print("\n\n")
