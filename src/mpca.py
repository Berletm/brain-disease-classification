from pca import read_mri, resize_img
from utils import *
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict

# from mxnet-the-straight-dope
def unfold(tensor: np.ndarray, mode: int) -> np.ndarray:
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))

# from mxnet-the-straight-dope
def fold(tensor: np.ndarray, mode: int, shape: tuple) -> np.ndarray:
    full_shape = list(shape)
    mode_dim = full_shape.pop(mode)
    full_shape.insert(0, mode_dim)
    return np.moveaxis(np.reshape(tensor, full_shape), 0, mode)

# from mxnet-the-straight-dope
# aka tensor contraction
def n_mode_prod(tensor: np.ndarray, matrix: np.ndarray, mode: int) -> np.ndarray:
    res = np.dot(matrix, unfold(tensor, mode))

    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]

    return fold(res, mode, tuple(new_shape))

def mpca(dataset: np.ndarray, iterations: int = 15, ranks: tuple = (128, 128, 1), projection_mode: str = 'sagital') -> np.ndarray:
    """
    This function projects all tensors from dataset from initial shape to ranks shape
    :param dataset: array of tensors shaped 128x128x128
    :param iterations: number of iterations for calculating projection matrices
    :param ranks: output shape
    :param projection_mode: full, axial, sagital, frontal modes: full mode uses all three projection matrices, others uses only one
    depth projection matrix
    :return: returns projected dataset
    """
    n, i_dim, j_dim, k_dim = dataset.shape
    r1, r2, r3 = ranks
    orig_dims = [i_dim, j_dim, k_dim]

    # projection matrices U_i = I
    U1 = np.eye(i_dim)[:, :r1] # 128 x 128
    U2 = np.eye(j_dim)[:, :r2] # 128 x 128
    U3 = np.eye(k_dim)[:, :r3] # 128 x 3
    Us = [U1, U2, U3]

    # data centering
    mean_tensor = np.mean(dataset, axis=0)
    X = dataset - mean_tensor

    # metric
    prev_phi = 0.0
    for it in range(iterations):
        for m in range(3):
            others = [idx for idx in range(3) if idx != m]
            Phi = np.zeros((orig_dims[m], orig_dims[m]))

            for i in range(n):
                temp_proj = X[i]
                for o in others:
                    temp_proj = n_mode_prod(temp_proj, Us[o].T, o)

                unfolded = unfold(temp_proj, m)
                Phi += unfolded @ unfolded.T

            eig_vals, eig_vecs = np.linalg.eigh(Phi)
            sort_indices = np.argsort(eig_vals)[::-1]
            Us[m] = eig_vecs[:, sort_indices[:ranks[m]]]

            # explained variance
            used_eig_vals = sum(eig_vals[sort_indices[:ranks[m]]])
            all_eig_vals  = sum(eig_vals)
            print(f"Explained Variance {used_eig_vals/all_eig_vals} on {it} iter on mode {m}")

        current_phi = 0
        for i in range(n):
            proj = X[i]
            for m in range(3):
                proj = n_mode_prod(proj, Us[m].T, m)
            current_phi += np.linalg.norm(proj)

        if abs(current_phi - prev_phi) < 1e-6:
            print(f"Converged at iteration {it}")
            break
        prev_phi = current_phi

    projected_all = []
    for i in range(n):
        proj = dataset[i]
        if projection_mode == "sagital":
            proj = n_mode_prod(proj, Us[2].T, 2)
        elif projection_mode == "frontal":
            proj = n_mode_prod(proj, Us[1].T, 1)
        elif projection_mode == "axial":
            proj = n_mode_prod(proj, Us[0].T, 0)
        elif projection_mode == "full":
            for m in range(3):
                proj = n_mode_prod(proj, Us[m].T, m)
        projected_all.append(proj)

    return np.array(projected_all)

def normalize_image(img):
    vmin, vmax = np.percentile(img, (2, 98))

    img_norm = (img - vmin) / (vmax - vmin)

    img_norm = np.clip(img_norm, 0, 1)

    return img_norm

def generate_reduced_dataset(data: Dict[str, np.ndarray], plane: str="axial") -> None:
    dataset = list(data.values())
    namings = list(data.keys())

    s = np.array([img.shape for img in dataset])

    temp = []
    min_s = np.min(s, axis=0)[1:]
    for data in dataset:
        for img in data:
            new_img = resize_img(img, min_s)
            temp.append(new_img)
    
    resized_dataset = np.array(temp)

    w, h = np.min(s, axis=0)[0:2]
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
    reduced_dataset = mpca(resized_dataset, 15, shape, plane)

    sizes = [np.size(data, axis=0) for data in dataset]
    print(sizes)

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
        pth = os.path.join(REDUCED_DATASET_PATH, plane, f"{name}_{i}.png")
        plt.imsave(pth, img)

if __name__ == "__main__":
    parkinson = read_mri(PARKINSON_DATASET_PATH)
    autism    = read_mri(AUTISM_DATASET_PATH)
    control   = read_mri(CONTROL_DATASET_PATH)
    control_ixi = read_mri(CONTROL_IXI_DATASET_PATH)
    alzheimer = read_mri(ALZHEIMER_DATASET_PATH)
    adhd      = read_mri(ADHD_DATASET_PATH)
    # control_adhd = read_mri(CONTROL_ADHD_DATASET_PATH)

    namings   = ["parkinson", "control", "control_ixi", "alzheimer", "adhd", "autism"]
    dataset   = [parkinson, control, control_ixi, alzheimer, adhd, autism]

    data = dict(zip(namings, dataset))
    dataset = list(data.values())
    namings = list(data.keys())

    for plane in ["axial", "sagital", "frontal"]:
        print(plane)
        generate_reduced_dataset(data, plane)
        print("\n\n")
