from dataclasses import dataclass

from torch.utils.data import Dataset
import torch
from typing import Tuple
import torchvision.transforms as tv
import os
from PIL import Image

def _label_from_filename(filename: str):
    """Return the first matching class label, or None for unknown names."""
    if "control" in filename:
        return 0
    elif "parkinson" in filename:
        return 1
    elif "alzheimer" in filename:
        return 2
    elif "adhd" in filename:
        return 3
    elif "sclerosis" in filename:
        return 4
    elif "autism" in filename:
        return 5
    return None

@dataclass
class Axis:
    axial:   str
    frontal: str
    sagital: str

class AxisHolder(Dataset):
    def __init__(self, dataset_dir: str, x_transforms: tv.Compose):
        self.images: list[Axis] = []
        self.labels = []
        self.dir = dataset_dir
        self.x_transforms = x_transforms


        for img in os.listdir(self.dir + "/axial"):
            ax_dir = os.path.join(self.dir + "/axial", img)
            front_dir = os.path.join(self.dir + "/frontal", img)
            sag_dir = os.path.join(self.dir + "/sagital", img)
            img_ = Axis(ax_dir, front_dir, sag_dir)
            self.images.append(img_)

            label = _label_from_filename(img)
            if label is not None:
                self.labels.append(label)

        self.counts = [self.labels.count(i) for i in range(6)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        img_path = self.images[idx]

        ax_img    = Image.open(img_path.axial).convert("RGB")
        front_img = Image.open(img_path.frontal).convert("RGB")
        sag_img   = Image.open(img_path.sagital).convert("RGB")
        label = self.labels[idx]

        ax_img    = self.x_transforms(ax_img)
        front_img = self.x_transforms(front_img)
        sag_img   = self.x_transforms(sag_img)
        label = torch.tensor(label, dtype=torch.long)

        return (ax_img, front_img, sag_img), label

class SliceHolder(Dataset):
    def __init__(self, dataset_dir: str, x_transforms: tv.Compose):
        self.images: list[str] = []
        self.labels: list[int] = []
        self.dir: str = dataset_dir
        self.x_transforms: tv.Compose = x_transforms


        for img in os.listdir(self.dir):
            img_pth = os.path.join(self.dir, img)
            self.images.append(img_pth)

            label = _label_from_filename(img)
            if label is not None:
                self.labels.append(label)

        self.counts = [self.labels.count(i) for i in range(6)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]

        img   = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        img   = self.x_transforms(img)
        label = torch.tensor(label, dtype=torch.long)

        return img, label
