import numpy as np


class HeuristicSlicing:
    def __init__(
        self,
        n_components: int = 3,
        slicing_axis: str = "ax",
        mode: str = "cov",
        partition: float = False,
    ):
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
            cov_mat = (slice @ slice.T) / (slice.shape[0] - 1)

            return np.linalg.norm(cov_mat, ord="fro")
        else:
            return np.linalg.norm(slice, ord="fro")

    def extract_slices(self, volume: np.ndarray, n_comp: int = 3) -> np.ndarray:
        scores = []
        for image_slice in volume:
            scores.append(self.score(image_slice))

        indices = np.argsort(scores)[-n_comp:][::-1]
        return indices

    def fit(self, volume: np.ndarray) -> None:
        volume = self.transpose_volume(volume)

        if not self.partition:
            self.slice_indices = self.extract_slices(volume, 3)
        else:
            n = len(volume)
            step = round((1 / 3) * n)

            top_part = (0, step)
            mid_part = (step, 2 * step)
            bot_part = (2 * step, n)

            slices = []

            for part in [top_part, mid_part, bot_part]:
                start, stop = part

                partition_indices = np.arange(start, stop)
                volume_part = volume[partition_indices]
                selected_indices = self.extract_slices(volume_part, 1)
                slices.append(selected_indices)

            self.slice_indices = slices

    def transform(self, volume: np.ndarray) -> np.ndarray:
        volume = self.transpose_volume(volume)
        return volume[self.slice_indices]

    def fit_transform(self, volume: np.ndarray) -> np.ndarray:
        self.fit(volume)
        return self.transform(volume)
