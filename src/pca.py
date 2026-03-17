import nibabel as nib
import numpy as np
import os

from skimage.transform import resize
from sklearn.decomposition import PCA

from utils import PARKINSON_DATASET_PATH, CONTROL_DATASET_PATH, AUTISM_DATASET_PATH


def read_mri(filepath: str) -> np.ndarray:
    data = []

    for file in os.listdir(filepath):
        img = nib.load(os.path.join(filepath, file))
        data.append(img.get_fdata())
    
    return np.array(data)

def resize_img(img: np.ndarray, target_size: tuple[int, int, int]) -> np.ndarray:
    return resize(img,
                  target_size,
                  order=3,
                  preserve_range=True,
                  anti_aliasing=True
                  )

def reduce_dim(data: np.ndarray, explained_var: float) -> np.ndarray:
    """resizing + MRI flatten + default PCA"""

    resized_data = []

    for img in data:
        resized_img = resize_img(img, (128, 128, 128))

        resized_data.append(resized_img)

    resized_data = np.array(resized_data)

    flattened_data = []

    for img in resized_data:
        flattened_data.append(img.flatten())

    pca = PCA(n_components=explained_var, svd_solver="full")
    reduced_vectors = pca.fit_transform(flattened_data)

    return reduced_vectors

def save_reduced_data(filename: str, reduced_data: np.ndarray) -> None:
    np.savetxt(filename, reduced_data)

def load_reduced_data(filename: str) -> np.ndarray:
    return np.loadtxt(filename)

def main() -> None:
    parkinson = read_mri(PARKINSON_DATASET_PATH)
    autism    = read_mri(AUTISM_DATASET_PATH)
    control   = read_mri(CONTROL_DATASET_PATH)
    dataset   = np.vstack([parkinson, control, autism])
    reduced_parkinson = reduce_dim(dataset, 0.99)
    save_reduced_data("../reduced_dataset.txt", reduced_parkinson)

    # print(reduced_parkinson)

if __name__ == "__main__":
    main()

