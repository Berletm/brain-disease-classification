import json
from pathlib import Path

import numpy as np
from skimage.transform import resize

from src.preprocessing.mpca import MPCA, normalize_image


class FoldMPCA:
    def __init__(self, shape, ranks, plane='axial', max_iter=15):
        self.shape = tuple(shape)
        self.ranks = tuple(ranks)
        self.plane = plane
        self.max_iter = max_iter

        self.mean = None
        self.model = None
        self.train_ids = ()

    def _prepare(self, volumes):
        values = []

        for volume in volumes:
            if volume.ndim != 3 or not np.isfinite(volume).all():
                raise ValueError('Expected finite 3D MRI')

            image = resize(
                volume,
                self.shape,
                order=3,
                preserve_range=True,
                anti_aliasing=True,
            )

            lo, hi = np.percentile(image, (2, 98))
            if hi <= lo:
                raise ValueError('Degenerate intensity range')

            values.append(normalize_image(image))

        if not values:
            raise ValueError('Empty MRI collection')

        return np.array(values)

    def fit(self, volumes, sample_ids):
        if self.model is not None:
            raise RuntimeError('Create a new pipeline for each training fold')

        ids = tuple(sample_ids)
        if len(ids) != len(volumes) or len(set(ids)) != len(ids):
            raise ValueError('Expected one unique sample ID per volume')

        values = self._prepare(volumes)
        mean = values.mean(axis=0)

        model = MPCA(
            kernel='linear',
            ranks=self.ranks,
            projection_mode=self.plane,
            max_iter=self.max_iter,
            degree=3,
        )

        model.fit(values - mean)

        self.mean, self.model, self.train_ids = mean, model, ids

        return self

    def transform(self, volumes):
        if self.model is None:
            raise RuntimeError('Pipeline must be fitted on training data first')

        return self.model.transform(self._prepare(volumes) - self.mean)

    def save(self, directory):
        if self.model is None:
            raise RuntimeError('Pipeline is not fitted')

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=False)

        self.model.save(directory / 'mpca.npz')
        np.save(directory / 'mean.npy', self.mean, allow_pickle=False)

        (directory / 'metadata.json').write_text(
            json.dumps({
                'version': 1,
                'shape': self.shape,
                'ranks': self.ranks,
                'plane': self.plane,
                'max_iter': self.max_iter,
                'train_ids': self.train_ids,
            })
        )

    @classmethod
    def load(cls, directory):
        directory = Path(directory)
        metadata = json.loads((directory / 'metadata.json').read_text())

        if metadata.pop('version') != 1:
            raise ValueError('Unsupported preprocessing version')

        train_ids = metadata.pop('train_ids')
        result = cls(**metadata)

        result.mean = np.load(directory / 'mean.npy', allow_pickle=False)
        result.model = MPCA(
            kernel='linear',
            degree=3,
            max_iter=result.max_iter,
            ranks=result.ranks,
            projection_mode=result.plane,
        )

        result.model.load(directory / 'mpca.npz')
        result.train_ids = tuple(train_ids)

        return result
