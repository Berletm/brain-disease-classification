from pca import read_mri, resize_img
from utils import PARKINSON_DATASET_PATH, CONTROL_DATASET_PATH, AUTISM_DATASET_PATH, REDUCED_DATASET_PATH
import numpy as np
import matplotlib.pyplot as plt

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

def mpca(dataset: np.ndarray, iterations: int = 15, ranks: tuple = (128, 128, 1), projection_mode: str = 'depth') -> np.ndarray:
    """
    This function projects all tensors from dataset from initial shape to ranks shape
    :param dataset: array of tensors shaped 128x128x128
    :param iterations: number of iterations for calculating projection matrices
    :param ranks: output shape
    :param projection_mode: full or depth mode: full mode uses all three projection matrices, depth uses only one
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
        if projection_mode == "depth":
            proj = n_mode_prod(proj, Us[2].T, 2)
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


def main() -> None:
    parkinson = read_mri(PARKINSON_DATASET_PATH)
    autism    = read_mri(AUTISM_DATASET_PATH)
    control   = read_mri(CONTROL_DATASET_PATH)
    dataset   = np.vstack([parkinson, control, autism])
    dataset = np.array([resize_img(img, (128, 128, 128)) for img in dataset])

    dataset = mpca(dataset, 15,(128, 128, 3), 'depth')

    for i, img in enumerate(dataset):
        img = np.transpose(np.squeeze(img), (0, 1, 2))
        img = normalize_image(img)
        if i < 30:
            plt.imsave(REDUCED_DATASET_PATH + f"/sagital/parkinson_{i}.png", img)
        elif 30 <= i < 70:
            plt.imsave(REDUCED_DATASET_PATH + f"/sagital/control{i}.png", img)
        else:
            plt.imsave(REDUCED_DATASET_PATH + f"/sagital/autism_{i}.png", img)

if __name__ == "__main__":
    main()
