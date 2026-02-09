from dataclasses import dataclass

from torch.utils.data import Dataset
import torch
from typing import Tuple
import torchvision.transforms as tv
import os
from PIL import Image

@dataclass
class Axis:
    axial:   str
    frontal: str
    sagital: str

class Holder(Dataset):
    def __init__(self, dataset_dir: str, x_transforms: tv.Compose):
        self.dir = dataset_dir
        self.x_transforms = x_transforms

        self.images = []
        self.labels = []

        for img in os.listdir(self.dir):
            self.images.append(img)

            if "control"     in img: self.labels.append(0)
            elif "parkinson" in img: self.labels.append(1)
            elif "autism"    in img: self.labels.append(2)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path = self.images[idx]

        img    = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        img    = self.x_transforms(img)
        label = torch.tensor(label, dtype=torch.long)

        return img, label

class AxisHolder(Holder):
    def __init__(self, dataset_dir: str, x_transforms: tv.Compose):
        super().__init__(dataset_dir, x_transforms)

        self.images: list[Axis] = []

        for img in os.listdir(self.dir + "/axial"):
            ax_dir = os.path.join(self.dir + "/axial", img)
            front_dir = os.path.join(self.dir + "/frontal", img)
            sag_dir = os.path.join(self.dir + "/sagital", img)
            img_ = Axis(ax_dir, front_dir, sag_dir)
            self.images.append(img_)

            if "control"     in img: self.labels.append(0)
            elif "parkinson" in img: self.labels.append(1)
            elif "autism"    in img: self.labels.append(2)

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