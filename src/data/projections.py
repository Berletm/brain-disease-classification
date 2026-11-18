import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as tv

from src.preprocessing.mpca import normalize_image


def projection_image(values):
    values = normalize_image(values)

    if not np.isfinite(values).all():
        raise ValueError('Non-finite projected intensities')

    return Image.fromarray(np.uint8(values * 255))


class ProjectionImages:
    def __init__(self, paths, shape):
        self.paths = tuple(paths)
        self.shape = tuple(shape)

        if len(self.shape) != 4 or self.shape[0] != len(self.paths) or self.shape[-1] != 3:
            raise ValueError('PNG paths must contain aligned RGB projections')

    def __len__(self):
        return len(self.paths)

    def load(self, index):
        with Image.open(self.paths[index]) as image:
            if (
                image.format != 'PNG'
                or image.mode != 'RGB'
                or image.size != (self.shape[2], self.shape[1])
            ):
                raise ValueError('Cached image is not a PNG RGB projection on the expected grid')

            return image.copy()


class ProjectionDataset(Dataset):
    def __init__(self, projections, labels, training=False, plane_count=3):
        labels = np.asarray(labels)

        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError('Expected a nonempty vector of explicit class labels')

        if labels.dtype.kind not in 'iu' or (labels < 0).any():
            raise ValueError('Class labels must be nonnegative integers')

        if plane_count not in (1, 3):
            raise ValueError('Expected one or three projection planes in the experiment')

        if len(projections) != plane_count:
            expected = 'three' if plane_count == 3 else 'one'
            raise ValueError(f'Expected {expected} MRI projection planes')

        for plane in projections:
            if isinstance(plane, ProjectionImages):
                aligned = len(plane) == len(labels)
            else:
                aligned = plane.ndim == 4 and plane.shape[0] == len(labels) and plane.shape[-1] == 3

            if not aligned:
                raise ValueError('Projection planes and labels must contain aligned RGB samples')

        self.projections, self.labels = projections, labels

        operations = []
        if training:
            operations = [
                tv.RandomAffine(
                    degrees=(-7, 7),
                    translate=(.08, .08),
                    scale=(.92, 1.10),
                    shear=(-7, 7),
                    interpolation=tv.InterpolationMode.BICUBIC,
                ),
                tv.RandomHorizontalFlip(.5),
            ]

        self.transforms = tv.Compose(
            operations + [
                tv.ToTensor(),
                tv.Resize((224, 224)),
                tv.Normalize([.485, .456, .406], [.229, .224, .225]),
            ]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        inputs = []
        for plane in self.projections:
            if isinstance(plane, ProjectionImages):
                image = plane.load(index)
            else:
                image = projection_image(plane[index])

            inputs.append(self.transforms(image))

        return tuple(inputs), int(self.labels[index])
