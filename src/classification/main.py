import numpy as np
import os
from PIL import Image
from typing import Tuple

from data_augmentation import AxisHolder
from torch.utils.data import random_split, DataLoader
from classification.clf import MultiCLF, train_multi
from utils.utils import REDUCED_DATASET_PATH
import torchvision.transforms as tv

def read_dataset(path: str) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []

    for file in os.listdir(path):
        img = Image.open(path + "/" + file).convert('RGB')
        X.append(img)
        if   "control"   in file: y.append(0)
        elif "autism"    in file: y.append(1)
        elif "parkinson" in file: y.append(2)

    return np.array(X), np.array(y)

def main() -> None:
    x_base_transforms = tv.Compose(
    [
        tv.ToTensor(),
        tv.Resize(224),
        tv.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    ds = AxisHolder(REDUCED_DATASET_PATH, x_base_transforms)

    train_ds, val_ds = random_split(ds, [80, 20])

    train_transforms = tv.Compose([
        tv.ToTensor(),
        tv.Resize(224),
        tv.RandomRotation(30),
        tv.ElasticTransform(),
        tv.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        tv.RandomVerticalFlip(),
        tv.RandomHorizontalFlip(),
        tv.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.9, 1.1), shear=8),
        tv.RandomErasing(p=0.4, scale=(0.02, 0.25)),
        tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds.x_transforms = train_transforms

    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(val_ds,  batch_size=1, shuffle=True, num_workers=4, pin_memory=True)

    model = MultiCLF()

    model = train_multi(n_epoch=100, model=model, train_loader=train_loader, val_loader=test_loader)



if __name__ == "__main__":
    main()