import os

import nibabel as nib
import numpy as np


class MPCA:
    def __init__(
        self,
        kernel: str = "linear",
        degree: int = 2,
        max_iter: int = 15,
        ranks: tuple = (),
        projection_mode: str = "axial",
        tol: float = 1e-6,
    ):
        self.kernel = kernel
        self.max_iter = max_iter
        self.degree = degree

        self.ranks = np.array(ranks)

        self.axial_mat = None
        self.frontal_mat = None
        self.sagital_mat = None
        self.orig_shape = None

        self.tol = tol
        self.projection_mode = projection_mode
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

        return np.array(projected_all)

    def __inner_prod(self, lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        dots = lhs @ rhs

        if self.kernel == "linear":
            return dots

        if self.kernel == "poly":
            return (dots + 1) ** self.degree

        if self.kernel == "sigmoid":
            return np.tanh(dots)

        if self.kernel == "rbf":
            lhs_norm = np.sum(lhs ** 2, axis=1)
            rhs_norm = np.sum(rhs ** 2, axis=0)

            dist_sq = lhs_norm[:, np.newaxis] + rhs_norm[np.newaxis, :] - 2 * (dots)

            dist_sq = np.maximum(dist_sq, 0.0)

            return np.exp(-dist_sq)

        if self.kernel == "cosine":
            lhs_norm = np.sqrt(np.sum(lhs ** 2, axis=1))
            rhs_norm = np.sqrt(np.sum(rhs ** 2, axis=0))
            norms = np.maximum(lhs_norm * rhs_norm, 1e-10)

            return dots / norms

        return dots

    def fit(self, dataset: np.ndarray) -> None:
        n, i_dim, j_dim, k_dim = dataset.shape
        orig_dims = [i_dim, j_dim, k_dim]

        self.orig_shape = tuple(orig_dims)

        r1, r2, r3 = self.ranks
        self.axial_mat = np.eye(i_dim)[:, :r1]
        self.frontal_mat = np.eye(j_dim)[:, :r2]
        self.sagital_mat = np.eye(k_dim)[:, :r3]

        mats = [self.axial_mat, self.frontal_mat, self.sagital_mat]

        original_variance = sum(
            np.linalg.norm(dataset[i]) ** 2
            for i in range(n)
        )

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

            print(
                f"Iter {it}| Total preserved variance ratio: "
                f"{preserved_variance / original_variance:.6f}"
            )

            if abs(preserved_variance - prev_total_variance) < self.tol:
                print(f"Converged at iteration {it}")
                break
            prev_total_variance = preserved_variance

        self.axial_mat, self.frontal_mat, self.sagital_mat = mats
        self.initialized = True

    def load(self, pth: str) -> None:
        data = np.load(pth, allow_pickle=True)
        self.axial_mat, self.frontal_mat, self.sagital_mat = (
            data["axial_mat"], data["frontal_mat"], data["sagital_mat"],
        )
        self.projection_mode = str(data["projection_mode"])
        self.ranks = data["ranks"]
        self.orig_shape = tuple(data["orig_shape"])
        self.initialized = True

    def save(self, pth: str) -> None:
        if not self.initialized:
            raise RuntimeError("Can not save not initialized model")

        np.savez_compressed(
            pth,
            axial_mat=self.axial_mat,
            frontal_mat=self.frontal_mat,
            sagital_mat=self.sagital_mat,
            ranks=self.ranks,
            projection_mode=self.projection_mode,
            orig_shape=self.orig_shape,
        )


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
