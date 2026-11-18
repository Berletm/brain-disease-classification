
import numpy as np
import pytest
from torch.utils.data import DataLoader

from src.data.projections import ProjectionDataset


def projections():
    rng = np.random.default_rng(9)
    return [rng.normal(size=(2, 4, 5, 3)) for _ in range(3)]


@pytest.mark.parametrize('labels', [[], [[0, 1]], [-1, 0], [0.5, 1.0]])
def test_missing_or_invalid_explicit_labels_are_rejected(labels):
    with pytest.raises(ValueError, match='labels'):
        ProjectionDataset(projections(), labels)


def test_misaligned_planes_are_rejected_before_loading():
    planes = projections()
    planes[1] = planes[1][:1]

    with pytest.raises(ValueError, match='aligned'):
        ProjectionDataset(planes, [0, 1])


def test_missing_plane_is_rejected():
    with pytest.raises(ValueError, match='three'):
        ProjectionDataset(projections()[:2], [0, 1])


def test_train_transforms_do_not_modify_evaluation_dataset():
    planes = projections()
    labels = np.array([5, 0])

    training = ProjectionDataset(planes, labels, training=True)
    evaluation = ProjectionDataset(planes, labels, training=False)

    from torchvision.transforms import RandomAffine, RandomHorizontalFlip

    assert any(isinstance(t, RandomAffine) for t in training.transforms.transforms)
    assert any(isinstance(t, RandomHorizontalFlip) for t in training.transforms.transforms)
    assert not any(isinstance(t, (RandomAffine, RandomHorizontalFlip))
                   for t in evaluation.transforms.transforms)


def test_loader_preserves_explicit_label_order():
    from torch import int64

    dataset = ProjectionDataset(projections(), [5, 0])
    _, labels = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))

    assert labels.tolist() == [5, 0]
    assert labels.dtype == int64
