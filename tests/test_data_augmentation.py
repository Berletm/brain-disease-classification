import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from data_augmentation import AxisHolder, SliceHolder


def make_dataset(tmp_path, monkeypatch, holder, names):
    directories = ("axial", "frontal", "sagital") if holder is AxisHolder else ("",)
    for plane, directory in enumerate(directories):
        target = tmp_path / directory
        target.mkdir(exist_ok=True)
        for name in names:
            Image.new("RGB", (2, 3), (10 + plane, 20, 30)).save(target / name)
    # Control only filesystem enumeration; exercise actual loading and transforms.
    monkeypatch.setattr("data_augmentation.os.listdir", lambda _: list(names))
    return holder(str(tmp_path), ToTensor())


@pytest.mark.parametrize("holder", [AxisHolder, SliceHolder])
def test_class_labels_counts_and_enumeration_order(tmp_path, monkeypatch, holder):
    names = ["autism_0.png", "control_0.png", "sclerosis_0.png",
             "adhd_0.png", "alzheimer_0.png", "parkinson_0.png", "control_1.png"]
    dataset = make_dataset(tmp_path, monkeypatch, holder, names)
    assert dataset.labels == [5, 0, 4, 3, 2, 1, 0]
    assert dataset.counts == [2, 1, 1, 1, 1, 1]
    assert len(dataset) == 7
    paths = [entry.axial if holder is AxisHolder else entry for entry in dataset.images]
    assert [Path(path).name for path in paths] == names


@pytest.mark.parametrize("holder", [AxisHolder, SliceHolder])
@pytest.mark.parametrize("name,label", [
    ("autism_control.png", 0),
    ("parkinson_alzheimer.png", 1),
    ("alzheimer_adhd.png", 2),
    ("adhd_sclerosis.png", 3),
    ("sclerosis_autism.png", 4),
])
def test_ambiguous_filenames_preserve_label_priority(tmp_path, monkeypatch, holder, name, label):
    dataset = make_dataset(tmp_path, monkeypatch, holder, [name])
    assert dataset.labels == [label]
    assert dataset.counts[label] == 1


@pytest.mark.parametrize("holder", [AxisHolder, SliceHolder])
def test_unknown_filename_keeps_image_without_label(tmp_path, monkeypatch, holder):
    dataset = make_dataset(tmp_path, monkeypatch, holder, ["unknown.png"])
    assert len(dataset) == 1
    assert dataset.labels == []
    assert dataset.counts == [0] * 6
    with pytest.raises(IndexError):
        dataset[0]


@pytest.mark.parametrize("holder", [AxisHolder, SliceHolder])
def test_getitem_values_shapes_label_dtype_and_plane_order(tmp_path, monkeypatch, holder):
    dataset = make_dataset(tmp_path, monkeypatch, holder, ["adhd.png"])
    images, label = dataset[0]
    assert label.shape == torch.Size([])
    assert label.dtype == torch.long
    assert label.device.type == "cpu"
    assert label.item() == 3
    if holder is AxisHolder:
        assert isinstance(images, tuple)
        assert len(images) == 3
    else:
        images = (images,)
    for plane, image in enumerate(images):
        assert image.shape == (3, 3, 2)
        assert image.dtype == torch.float32
        assert image.device.type == "cpu"
        expected = torch.tensor([10 + plane, 20, 30], dtype=torch.float32) / 255
        torch.testing.assert_close(image, expected[:, None, None].expand(3, 3, 2), rtol=0, atol=0)


@pytest.mark.parametrize("holder", [AxisHolder, SliceHolder])
def test_transform_calls_and_rgb_conversion(tmp_path, monkeypatch, holder):
    dataset = make_dataset(tmp_path, monkeypatch, holder, ["control.png"])
    paths = [dataset.images[0].axial, dataset.images[0].frontal, dataset.images[0].sagital] if holder is AxisHolder else dataset.images
    for path in paths:
        Image.new("L", (2, 3), 17).save(path)
    calls = []

    def transform(image):
        assert image.mode == "RGB"
        calls.append(image.getpixel((0, 0)))
        return len(calls)

    dataset.x_transforms = transform
    images, _ = dataset[0]
    assert images == ((1, 2, 3) if holder is AxisHolder else 1)
    assert calls == [(17, 17, 17)] * (3 if holder is AxisHolder else 1)
