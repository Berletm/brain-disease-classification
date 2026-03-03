import numpy as np
import os
from PIL import Image
from typing import Tuple

from classification.data_augmentation import AxisHolder
from torch.utils.data import random_split, DataLoader
from classification.clf import MultiCLF, train_multi, validate
from utils.utils import REDUCED_DATASET_PATH, SAVED_MODELS_PATH
import torchvision.transforms as tv
import torchvision.transforms.v2 as tv2
import torch
torch.manual_seed(0)

from preprocessing.mpca import generate_reduced_dataset

def main() -> None:
    x_base_transforms = tv.Compose(
    [
        tv.ToTensor(),
        tv.Resize(224),
        tv.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    ds = AxisHolder(REDUCED_DATASET_PATH, x_base_transforms)

    weights = 1 / np.array(ds.counts)

    train_ds, val_ds = random_split(ds, [0.8, 0.2])

    train_transforms = tv.Compose([
        tv.RandomAffine(
            degrees=(-7, 7),         
            translate=(0.08, 0.08),  
            scale=(0.92, 1.10),       
            shear=(-7, 7),           
            interpolation=tv.InterpolationMode.BICUBIC,
            fill=0
        ),

        tv.ElasticTransform(
            alpha=120.,              
            sigma=8.,                 
            interpolation=tv.InterpolationMode.BICUBIC,
            fill=0
        ),

        tv.RandomHorizontalFlip(p=0.5),
        tv.RandomVerticalFlip(p=0.15),   

        tv.RandomApply([
            tv.ColorJitter(
                brightness=(0.7, 1.4),
                contrast=(0.75, 1.35),
                saturation=0.,      
                hue=0.
            )
        ], p=0.45),

        tv.RandomApply([tv2.GaussianNoise(sigma=0.015)], p=0.25),
        tv.RandomApply([tv.GaussianBlur(kernel_size=3, sigma=(0.4, 1.4))], p=0.20),

        tv.Resize((224, 224), interpolation=tv.InterpolationMode.BICUBIC),
        tv.ToTensor(),

        tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds.x_transforms = train_transforms

    train_loader = DataLoader(train_ds, batch_size=20, shuffle=True, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(val_ds,  batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

    model = MultiCLF()

    model = torch.load(SAVED_MODELS_PATH + "/v4/multi.pth", "cuda", weights_only=False)
    model = train_multi(n_epoch=200, model=model, train_loader=train_loader, val_loader=test_loader, weights=weights)
    
    pth = os.path.join(SAVED_MODELS_PATH, "multi.pth")
    torch.save(model, pth)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    loss, metrics, conf = validate(model, criterion, test_loader)
    
    acc, f1, recall, precision = metrics

    print(f"loss: {loss:.4f} | acc: {acc:.4f}")
    print(f"confusion mat: {conf}")

if __name__ == "__main__":
    # for plane in ["axial", "sagital", "frontal"]:
    #     print(plane)
    #     generate_reduced_dataset(plane)
    #     print("\n\n")
    main()