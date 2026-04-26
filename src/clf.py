from typing import Tuple, List, Callable

from utils import REDUCED_DATASET_PATH, SAVED_MODELS_PATH
from data_augmentation import AxisHolder
import torchvision.transforms as tv
import torchvision.transforms.v2 as tv2
import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import SAVED_MODELS_PATH
from logger import Tee
import optuna
import gc

optuna.logging.set_verbosity(optuna.logging.INFO)

import os
from tqdm import tqdm
import sys
torch.manual_seed(0)

os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CrossAttention(nn.Module):
    def __init__(self, feature_dim:int=512, heads_num:int=4):
        super().__init__()

        self.feature_dim = feature_dim
        self.heads_num = heads_num
        self.head_dim  = feature_dim // self.heads_num
        self.planes    = 3

        self.q_proj = nn.Linear(feature_dim, feature_dim)
        self.k_proj = nn.Linear(feature_dim, feature_dim)
        self.v_proj = nn.Linear(feature_dim, feature_dim)

        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, q_features: torch.Tensor, kv_features: torch.Tensor) -> torch.Tensor:
        batch_size = q_features.size(0)

        Q = self.q_proj(q_features)  # (512 x 512) * (512 x 1) = (512 x 1)
        Q = Q.view(batch_size, self.heads_num, self.head_dim) # (4 x 128) ~ 4 vec 128 x 1
        
        K = self.k_proj(kv_features) # (512 x 512) * (512 x 3) = (512 x 3)
        K = K.view(batch_size, self.planes, self.heads_num, self.head_dim) # 3 x 4 x 128 = 3 x (4 x 128) ~ 3 mat 4 vecs 128 x 1
        
        V = self.v_proj(kv_features) # (512 x 512) * (512 x 3) = (512 x 3)
        V = V.view(batch_size, self.planes, self.heads_num, self.head_dim) # 3 x 4 x 128 = 3 x (4 x 128) ~ 3 mat 4 vecs 128 x 1

        attn_scores = torch.einsum('bhd,bkhd->bhk', Q, K) # (4 x 3) ~ 4 vecs w scalars for planes
        attn_scores = attn_scores / (Q.shape[-1] ** 0.5)  # normalization from "Attetion is all you need"
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.dropout(attn_probs)

        out = torch.einsum('bhk,bkhd->bhd', attn_probs, V) # (3 x (4 x 128)) * (3 x 4) = 4 x 128 ~ 4 vec 128 x 1
        out = out.reshape(batch_size, -1) # flatten
        
        return self.out_proj(out) + q_features

class MultiCLF(nn.Module):
    def __init__(self, base_model:str="resnet18", hidden_dim:int=256, num_classes:int=4, attention_heads:int=4):
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim  = hidden_dim
        self.feature_dim = None
        
        if "resnet" in base_model:
            if base_model == "resnet18":
                self.model_ax = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.model_front = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.model_sag = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            elif base_model == "resnet34":
                self.model_ax = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
                self.model_front = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
                self.model_sag = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            elif base_model == "resnet50":
                self.model_ax = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                self.model_front = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
                self.model_sag = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                self.model_ax = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.model_front = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                self.model_sag = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

            features_dim = self.model_ax.fc.in_features

            self.model_ax.fc = nn.Identity()
            self.model_front.fc = nn.Identity()
            self.model_sag.fc = nn.Identity()

            for model in [self.model_ax, self.model_sag, self.model_front]:
                for name, param in model.named_parameters():
                    if name not in ["layer4", "fc"]:
                        param.requires_grad = False

        elif "convnext" in base_model:
            if base_model == "convnext_small":
                self.model_ax = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
                self.model_front = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
                self.model_sag = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
            elif base_model == "convnext_tiny":
                self.model_ax = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
                self.model_front = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
                self.model_sag = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
            elif base_model == "convnext_base":
                self.model_ax = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
                self.model_front = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
                self.model_sag = models.convnext_base(weights=models.ConvNeXt_Base_Weights.DEFAULT)
            else:
                self.model_ax = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
                self.model_front = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)
                self.model_sag = models.convnext_small(weights=models.ConvNeXt_Small_Weights.DEFAULT)

            features_dim = self.model_ax.classifier[2].in_features 
            self.feature_dim = features_dim

            self.model_ax.classifier[2] = nn.Identity()
            self.model_front.classifier[2] = nn.Identity()
            self.model_sag.classifier[2] = nn.Identity()

            for model in [self.model_ax, self.model_sag, self.model_front]:
                for param in model.parameters():
                    param.requires_grad = False

                for param in model.features[-1].parameters():
                    param.requires_grad = True

        self.cross_attention = CrossAttention(features_dim, attention_heads)

        self.clf_head = nn.Sequential(
            nn.Linear(features_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

        self.softmax = nn.Softmax()

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        ax, front, sag = x

        ax_logits    = self.model_ax(ax)
        front_logits = self.model_front(front)
        sag_logits   = self.model_sag(sag)

        logits = torch.stack([ax_logits, front_logits, sag_logits], dim=1)

        ax_attention_logits = self.cross_attention(ax_logits, logits)
        front_attention_logits = self.cross_attention(front_logits, logits)
        sag_attention_logits = self.cross_attention(sag_logits, logits)
        
        attention_logits = torch.stack([ax_attention_logits, front_attention_logits, sag_attention_logits], dim=1)
        
        fused_logits = torch.mean(attention_logits, dim=1)
        
        return self.clf_head(fused_logits)

    def predict(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> np.ndarray:
        ax, front, sag = x

        ax_logits = self.model_ax(ax)
        front_logits = self.model_front(front)
        sag_logits = self.model_sag(sag)

        logits = torch.cat([ax_logits, front_logits, sag_logits], dim=1)

        logits = self.clf_head(logits)

        p = self.softmax(logits)

        return torch.argmax(p, dim=1).cpu().numpy()


def train(n_epoch:    int,
          model:      nn.Module,
          X:          np.ndarray,
          y:          np.ndarray,
          X_val:      np.ndarray,
          y_val:      np.ndarray,
          batch_size: int,
          shuffle:    bool) -> Tuple[nn.Module, np.ndarray]:
    X_tensor = torch.from_numpy(X).float().to(device)
    X_val_tensor = torch.from_numpy(X_val).float().to(device)
    y_tensor = torch.from_numpy(y).long().to(device)
    y_val_tensor = torch.from_numpy(y_val).long().to(device)

    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

    patience = 20
    counter  = 0
    best_val_acc = 0.0

    acc_hist = []

    for epoch in range(n_epoch):
        if counter > patience:
            print(f"Early stopping, best Val accuracy: {best_val_acc}")
            break
        if shuffle:
            permutation = torch.randperm(X_tensor.shape[0])
            X_tensor = X_tensor[permutation]
            y_tensor = y_tensor[permutation]

        model.train()
        mean_loss = 0.0
        for i in range(0, len(X_tensor), batch_size):
            optimizer.zero_grad()

            X_tensor_batch = X_tensor[i: i + batch_size]
            y_tensor_batch = y_tensor[i: i + batch_size]


            output = model(X_tensor_batch)
            loss   = criterion(output, y_tensor_batch)

            loss.backward()
            optimizer.step()
            mean_loss += loss.item()

        mean_loss = mean_loss / (len(X_tensor) / batch_size)
        model.eval()
        with torch.no_grad():
            # val accuracy
            output_val = model(X_val_tensor)
            val_loss = criterion(output_val, y_val_tensor).item()
            preds = torch.argmax(output_val, dim=1)
            accuracy = (preds == y_val_tensor).float().mean().item()
            scheduler.step(val_loss)
            acc_hist.append(accuracy)

            if accuracy > best_val_acc:
                best_val_acc = accuracy
                counter = 0
            else:
                counter += 1

            # train accuracy
            preds = torch.argmax(model(X_tensor), dim=1)
            train_accuracy = (preds == y_tensor).float().mean().item()

            print(f"Epoch {epoch + 1}/{n_epoch}, Train loss: {mean_loss:.4f} Val loss: {val_loss:.4f}, Val Accuracy: {accuracy:.4f}, "
                  f"Train Accuracy: {train_accuracy:.4f}")

    return model, np.array(acc_hist)

def train_torch(n_epoch:  int,
                model:    nn.Module,
                train_loader: DataLoader,
                val_loader  : DataLoader) -> Tuple[nn.Module, np.ndarray]:
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = CrossEntropyLoss(label_smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer)

    for epoch in range(n_epoch):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            loss, preds = train_step(images, labels, model, optimizer, criterion)

            train_loss += loss
            correct += (preds == labels).sum().item()
            total += len(labels)
        train_loss /= total
        train_acc   = correct / total

        val_loss, val_acc = validate(model, criterion, val_loader)
        scheduler.step(val_loss)
        print(f"Epoch: {epoch + 1}/{n_epoch} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

    return model

def train_multi(n_epoch:  int,
                model:    MultiCLF,
                lr: float,
                train_loader: DataLoader,
                val_loader  : DataLoader,
                weights:List|np.ndarray=[1.0, 1.0, 1.0, 1.0], save:bool=True) -> Tuple[nn.Module, np.ndarray]:
    model = model.to(device)
    optimizer = AdamW([
        {'params': model.clf_head.parameters(), 'lr': lr},
        {'params': [p for p in model.model_ax.parameters() if p.requires_grad], 'lr': 1e-6},
        {'params': [p for p in model.model_sag.parameters() if p.requires_grad], 'lr': 1e-6},
        {'params': [p for p in model.model_front.parameters() if p.requires_grad], 'lr': 1e-6},
    ], weight_decay=1e-4)
    patience = 5
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = CrossEntropyLoss(weight=weights)
    # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)
    scheduler2 = ReduceLROnPlateau(optimizer, patience=patience, min_lr=1e-6, factor=0.5)

    counter  = 0
    best_loss = float("inf")
    best_acc = 0.0
    best_metric = 0.0

    log_counter = 1
    f_name = f"../models/z{log_counter}"
    if not os.path.exists(f_name):
        os.mkdir(f_name)
        log_file = open(os.path.join(f_name, "training.log"), "w+")
    else:
        while True:
            f_name = f"../models/z{log_counter}"
            if not os.path.exists(f_name):
                os.mkdir(f"../models/z{log_counter}")
                log_file = open(os.path.join(f_name, "training.log"), "w+")
                break
            else: log_counter += 1; continue

    sys.stdout = Tee(log_file, sys.stdout)

    for epoch in range(n_epoch):
        if counter > patience:
            print(f"Early stopping at epoch: {epoch+1}/{n_epoch} with best val acc: {best_acc:.4f} and best val loss: {best_loss:4f}")
            break
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"epoch №{epoch+1}")

        for images, labels in progress_bar:

            ax, front, sag = images
            ax = ax.to(device)
            front = front.to(device)
            sag = sag.to(device)
            images = (ax, front, sag)
            labels = labels.to(device)

            loss, preds = train_step(images, labels, model, optimizer, criterion)
            batch_size = labels.size(0)

            train_loss += loss * batch_size
            correct += (preds == labels).sum().item()
            total += batch_size
            progress_bar.set_postfix(train_loss=f"{train_loss/total:.4f}", train_acc=f"{correct/total:.4f}")
        train_acc  = correct / total
        train_loss /= total

        val_loss, metrics, cm = validate(model, criterion, val_loader)

        val_acc, f1, recall, precision = metrics

        if f1 > best_metric:
            counter = 0
            best_loss = val_loss
            best_acc  = val_acc
            best_metric = f1
            if save:
                torch.save(model, os.path.join(SAVED_MODELS_PATH, f"z{log_counter}", "best_multi.pth"))
        else: counter += 1

        scheduler2.step(f1)
        print(f"Epoch: {epoch + 1}/{n_epoch} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | f1 weighted: {f1:.4f} | recall: {recall:.4f} | precision: {precision:.4f}")
    log_file.close()
    sys.stdout = sys.__stdout__
    torch.save(model, os.path.join(SAVED_MODELS_PATH, f"z{log_counter}", f"multi.pth"))
    return model

def train_step(x, y, model: nn.Module, optimizer, criterion: nn.Module) -> Tuple[float, torch.Tensor]:
    optimizer.zero_grad()

    output = model(x)
    loss = criterion(output, y)
    preds = torch.argmax(output, dim=1)

    loss.backward()
    optimizer.step()

    return loss.item(), preds

def validate(model: nn.Module, criterion: nn.Module, val_loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    with torch.no_grad():
        for images, labels in val_loader:
            ax, front, sag = images
            ax = ax.to(device)
            front = front.to(device)
            sag = sag.to(device)
            images = (ax, front, sag)
            labels = labels.to(device)

            output = model(images)
            preds = torch.argmax(output, dim=1)
            y_pred.extend(preds.cpu().numpy().tolist())
            y_true.extend(labels.cpu().numpy().tolist())
            batch_size = labels.size(0)

            correct += (preds == labels).sum().item()
            val_loss += criterion(output, labels).item() * batch_size
            total += batch_size
        val_acc = correct / total
        val_loss /= total

    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    recall = recall_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)

    return val_loss, (val_acc, f1, recall, precision), cm

def cross_validate_pytorch(
    dataset: AxisHolder, 
    model_class: Callable,
    model_params: dict,
    train_func: Callable, 
    n_splits: int = 5,
    batch_size: int = 16,
):
    labels = np.array(dataset.labels)
    indices = np.arange(len(dataset))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    results = \
    {
        'fold_f1': [],
        'fold_recall': [], 
        'fold_precision': [],
        'fold_accuracies': [],
        'fold_models': [],
        'all_predictions': [],
        'all_true_labels': []
    }

    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"Fold {fold + 1}/{n_splits}")

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        model = model_class(**model_params).to(device) 

        model = train_func(n_epoch=50, model=model, train_loader=train_loader, val_loader=val_loader, lr=0.007133483490519629)

        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = tuple(t.to(device) for t in inputs) 
                targets = targets.to(device)

                outputs = model(inputs)
                predictions = torch.argmax(outputs, dim=1)

                val_preds.extend(predictions.cpu().numpy())
                val_true.extend(targets.cpu().numpy())

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)

        val_acc = (val_preds == val_true).mean()
        f1 = f1_score(val_true, val_preds, average="weighted")
        recall = recall_score(val_true, val_preds, average="weighted")
        precision = precision_score(val_true, val_preds, average="weighted", zero_division=0)

        results['fold_f1'].append(f1)
        results['fold_recall'].append(recall)
        results['fold_precision'].append(precision)
        results['fold_accuracies'].append(val_acc)
        results['all_predictions'].extend(val_preds)
        results['all_true_labels'].extend(val_true)

        del model, train_loader, val_loader, train_subset, val_subset
        torch.cuda.empty_cache()
        gc.collect()

    return results

def objective(trial: optuna.Trial) -> float:
    base_model = trial.suggest_categorical(name="base_model", choices=["resnet18", "resnet34", "resnet50", "convnext_tiny", "convnext_small", "convnext_base"])
    hidden_dim = trial.suggest_categorical(name="hidden_dim", choices=[64, 128, 256, 512, 1024])
    attention_heads = trial.suggest_categorical(name="heads", choices=[2, 4, 8, 16])
    lr = trial.suggest_float("learning_rate", low=1e-5, high=1e-2)

    x_base_transforms = tv.Compose(
    [
        tv.ToTensor(),
        tv.Resize((224, 224)),
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

    g = torch.Generator()
    g.manual_seed(0)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=1, pin_memory=True, generator=g)
    test_loader  = DataLoader(val_ds,  batch_size=8, shuffle=True, num_workers=1, pin_memory=True, generator=g)

    model = MultiCLF(base_model=base_model, hidden_dim=hidden_dim, num_classes=4, attention_heads=attention_heads)

    model = train_multi(n_epoch=75, model=model, lr=lr, train_loader=train_loader, val_loader=test_loader, weights=weights)

    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    loss, metrics, conf = validate(model, criterion, test_loader)
    
    acc, f1, recall, precision = metrics

    trial.set_user_attr("base_model", base_model)
    trial.set_user_attr("hidden_dim", hidden_dim)
    trial.set_user_attr("heads", attention_heads)
    trial.set_user_attr("lr", lr)
    print("\n\n")

    return f1


def main() -> None:
    x_base_transforms = tv.Compose(
    [
        tv.ToTensor(),
        tv.Resize((224, 224)),
        tv.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])

    ds = AxisHolder(REDUCED_DATASET_PATH, x_base_transforms)

    weights = 1 / np.array(ds.counts)
    print(ds.counts)

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

    g = torch.Generator()
    g.manual_seed(0)
    
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=1, pin_memory=True, generator=g)
    test_loader  = DataLoader(val_ds,  batch_size=4, shuffle=True, num_workers=1, pin_memory=True, generator=g)

    model = MultiCLF(base_model="convnext_base", num_classes=4, hidden_dim=64, attention_heads=16)

    # model_params = {"base_model": "convnext_base", "num_classes": 4, "hidden_dim": 64, "attention_heads": 16}

    # result = cross_validate_pytorch(dataset=ds, model_class=MultiCLF, train_func=train_multi, model_params=model_params, n_splits=5, batch_size=16)

    # with open("models/result.txt", "w+") as file:
    #     file.write(f"f1:        {np.mean(result['fold_f1']):.4f} +- {np.std(result['fold_f1']):.4f}\n")
    #     file.write(f"recall:    {np.mean(result['fold_recall']):.4f} +- {np.std(result['fold_recall']):.4f}\n")
    #     file.write(f"precision: {np.mean(result['fold_precision']):.4f} +- {np.std(result['fold_precision']):.4f}\n")
    #     file.write(f"accuracy:  {np.mean(result['fold_accuracies']):.4f} +- {np.std(result['fold_accuracies']):.4f}\n")

    model = train_multi(n_epoch=200, model=model, train_loader=train_loader, val_loader=test_loader, weights=weights, lr=0.007133483490519629)

    # study = optuna.create_study(direction='maximize')
    # study.optimize(objective, n_trials=100, timeout=4800)

    # df = study.trials_dataframe()
    # df.to_csv("optuna_results2.csv", index=False)

    # print(f"Best F1: {study.best_value}")
    # print(f"Best params: {study.best_params}")

if __name__ == "__main__":
    main()