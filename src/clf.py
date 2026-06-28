from typing import Tuple, List, Callable, Union

from utils import REDUCED_DATASET_PATH, SAVED_MODELS_PATH, HEURISTIC_DATASET_PATH
from data_augmentation import AxisHolder, SliceHolder
import torchvision.transforms as tv
import torchvision.transforms.v2 as tv2
import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from logger import Tee
import optuna
import gc
import copy
import seaborn as sns

optuna.logging.set_verbosity(optuna.logging.INFO)

import os
from tqdm import tqdm
import sys
torch.manual_seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

g = torch.Generator()
g.manual_seed(0)

class MultiHeadAttention(nn.Module):
    def __init__(self, feature_dim:int=512, attention_dim:int=512, heads_num:int=4, return_scores:bool=False):
        super().__init__()

        self.feature_dim = feature_dim
        self.attention_dim = attention_dim
        self.heads_num = heads_num
        self.head_dim  = attention_dim // self.heads_num
        self.scale = self.head_dim ** 0.5
        self.planes    = 3
        self.return_scores = return_scores

        self.q_proj = nn.Linear(feature_dim, attention_dim)
        self.k_proj = nn.Linear(feature_dim, attention_dim)
        self.v_proj = nn.Linear(feature_dim, attention_dim)

        self.out_proj = nn.Linear(attention_dim, feature_dim)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, q_features: torch.Tensor, kv_features: torch.Tensor) -> torch.Tensor:
        batch_size = q_features.size(0)

        Q = self.q_proj(q_features)  # (512 x 512) * (512 x 1) = (512 x 1)
        K = self.k_proj(kv_features) # (512 x 512) * (512 x 3) = (512 x 3)
        V = self.v_proj(kv_features) # (512 x 512) * (512 x 3) = (512 x 3)

        Q = Q.view(batch_size, self.heads_num, self.head_dim) # (4 x 128) ~ 4 vec 128 x 1
        K = K.view(batch_size, self.planes, self.heads_num, self.head_dim) # 3 x 4 x 128 = 3 x (4 x 128) ~ 3 mat 4 vecs 128 x 1
        V = V.view(batch_size, self.planes, self.heads_num, self.head_dim) # 3 x 4 x 128 = 3 x (4 x 128) ~ 3 mat 4 vecs 128 x 1

        attn_scores = torch.einsum('bhd,bkhd->bhk', Q, K) # (4 x 3) ~ 4 vecs w scalars for planes
        attn_scores /= self.scale  # normalization from "Attetion is all you need"
        attn_probs = self.softmax(attn_scores)
        attn_probs = self.dropout(attn_probs)

        out = torch.einsum('bhk,bkhd->bhd', attn_probs, V) # (3 x (4 x 128)) * (3 x 4) = 4 x 128 ~ 4 vec 128 x 1
        out = out.reshape(batch_size, -1) # flatten
        
        if self.return_scores:
            return self.out_proj(out) + q_features, attn_probs
        return self.out_proj(out) + q_features

class Baseline(nn.Module):
    def __init__(self, base_model: str, num_classes:int, hidden_dim: int):
        super().__init__()
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
                
            self.feature_dim = self.model_ax.fc.in_features

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

            self.feature_dim = self.model_ax.classifier[2].in_features

            self.model_ax.classifier[2] = nn.Identity()
            self.model_front.classifier[2] = nn.Identity()
            self.model_sag.classifier[2] = nn.Identity()

            for model in [self.model_ax, self.model_sag, self.model_front]:
                for param in model.parameters():
                    param.requires_grad = False

                for param in model.features[-1].parameters():
                    param.requires_grad = True
                    
        self.clf_head = nn.Sequential(
                    nn.Linear(self.feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_dim, num_classes)
                )
        self.softmax = nn.Softmax()
        
    def forward(self, x) -> torch.Tensor:
        raise NotImplementedError
    
    def predict(self, x) -> torch.Tensor:
        raise NotImplementedError

class MultiBranchAttention(Baseline):
    def __init__(self, base_model:str, hidden_dim:int, num_classes:int, attention_dim:int, attention_heads:int, return_attention_scores:bool=False):
        super().__init__(base_model, num_classes, hidden_dim)
        self.attention_heads = attention_heads
        self.attention_dim   = attention_dim
        self.clf_head[0] = nn.Linear(3 * self.feature_dim, hidden_dim, device=device)
        self.return_attention_scores = return_attention_scores
                
        self.multi_head_attention = MultiHeadAttention(self.feature_dim, self.attention_dim, self.attention_heads, return_attention_scores)

    def forward(self, ax: torch.Tensor, front: torch.Tensor, sag: torch.Tensor) -> torch.Tensor:
        ax_logits    = self.model_ax(ax)
        front_logits = self.model_front(front)
        sag_logits   = self.model_sag(sag)

        logits = torch.stack([ax_logits, front_logits, sag_logits], dim=1)
        
        if not self.return_attention_scores:
            ax_attention_logits = self.multi_head_attention(ax_logits, logits)
            front_attention_logits = self.multi_head_attention(front_logits, logits)
            sag_attention_logits = self.multi_head_attention(sag_logits, logits)
        else:
            ax_attention_logits, ax_scores = self.multi_head_attention(ax_logits, logits)
            front_attention_logits, front_scores = self.multi_head_attention(front_logits, logits)
            sag_attention_logits, sag_scores = self.multi_head_attention(sag_logits, logits)
        
        attention_logits = torch.cat([ax_attention_logits, front_attention_logits, sag_attention_logits], dim=1)
        
        if self.return_attention_scores:
            return self.clf_head(attention_logits), (ax_scores, front_scores, sag_scores)
        
        return self.clf_head(attention_logits)

    def predict(self, ax: torch.Tensor, front: torch.Tensor, sag: torch.Tensor) -> np.ndarray:
        ax_logits = self.model_ax(ax)
        front_logits = self.model_front(front)
        sag_logits = self.model_sag(sag)

        logits = torch.cat([ax_logits, front_logits, sag_logits], dim=1)

        logits = self.clf_head(logits)

        p = self.softmax(logits)

        return torch.argmax(p, dim=1).cpu().numpy()

    def return_scores(self) -> None:
        self.return_attention_scores            = True
        self.multi_head_attention.return_scores = True
    
    def hide_scores(self) -> None:
        self.return_attention_scores            = False
        self.multi_head_attention.return_scores = False
    
class SingleBranch(Baseline):
    def __init__(self, base_model, num_classes, hidden_dim) -> None:
        super().__init__(base_model, num_classes, hidden_dim)
        del self.model_front, self.model_sag
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.model_ax(x)
        
        return self.clf_head(features)

class MultiBranchConcat(Baseline):
    def __init__(self, base_model, num_classes, hidden_dim) -> None:
        super().__init__(base_model, num_classes, hidden_dim)
        self.clf_head[0] = nn.Linear(3 * self.feature_dim, hidden_dim, device=device)
        
    def forward(self, ax: torch.Tensor, front: torch.Tensor, sag: torch.Tensor) -> torch.Tensor:
        ax_logits    = self.model_ax(ax)
        front_logits = self.model_front(front)
        sag_logits   = self.model_sag(sag)

        logits = torch.cat([ax_logits, front_logits, sag_logits], dim=1)
        
        return self.clf_head(logits)

    def predict(self, ax: torch.Tensor, front: torch.Tensor, sag: torch.Tensor) -> np.ndarray:
        ax_logits = self.model_ax(ax)
        front_logits = self.model_front(front)
        sag_logits = self.model_sag(sag)

        logits = torch.cat([ax_logits, front_logits, sag_logits], dim=1)

        logits = self.clf_head(logits)

        p = self.softmax(logits)

        return torch.argmax(p, dim=1).cpu().numpy()
    
class MultiBranchMean(Baseline):
    def __init__(self, base_model, num_classes, hidden_dim):
        super().__init__(base_model, num_classes, hidden_dim)
        
    def forward(self, ax: torch.Tensor, front: torch.Tensor, sag: torch.Tensor) -> torch.Tensor:
        ax_logits    = self.model_ax(ax)
        front_logits = self.model_front(front)
        sag_logits   = self.model_sag(sag)

        logits = (ax_logits + front_logits + sag_logits) / 3.0
        
        return self.clf_head(logits)

    def predict(self, ax: torch.Tensor, front: torch.Tensor, sag: torch.Tensor) -> np.ndarray:
        ax_logits = self.model_ax(ax)
        front_logits = self.model_front(front)
        sag_logits = self.model_sag(sag)

        logits = (ax_logits + front_logits + sag_logits) / 3.0

        logits = self.clf_head(logits)

        p = self.softmax(logits)

        return torch.argmax(p, dim=1).cpu().numpy()

def train_multi(n_epoch: int,
                model: Union[MultiHeadAttention, MultiBranchConcat, MultiBranchMean],
                lr: float,
                train_loader: DataLoader,
                val_loader: DataLoader,
                weights: torch.Tensor, 
                save: bool=True,
                patience: int=5) -> Union[MultiHeadAttention, MultiBranchConcat, MultiBranchMean]:
    model = model.to(device)
    
    special_params_ids = (
        list(map(id, model.model_ax.parameters())) +
        list(map(id, model.model_sag.parameters())) +
        list(map(id, model.model_front.parameters()))
    )

    base_params = [
        p for p in model.parameters() 
        if p.requires_grad and id(p) not in special_params_ids
    ]
    
    optimizer = AdamW([
        {"params": base_params},
        {"params": [p for p in model.model_ax.parameters() if p.requires_grad],  "lr": 1e-6},
        {"params": [p for p in model.model_sag.parameters() if p.requires_grad],  "lr": 1e-6},
        {"params": [p for p in model.model_front.parameters() if p.requires_grad], "lr": 1e-6},
    ], lr=lr, weight_decay=1e-4)
    criterion = CrossEntropyLoss(weight=weights, label_smoothing=0.05)
    scheduler = ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    counter  = 0
    best_loss = float("inf")
    best_acc = 0.0
    best_metric = 0.0
    best_model = copy.deepcopy(model)
    
    if save:
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
        scheduler.step(val_loss)

        if f1 > best_metric:
            counter = 0
            best_loss = val_loss
            best_acc  = val_acc
            best_metric = f1
            del best_model
            best_model = copy.deepcopy(model)
            if save:
                torch.save(model, os.path.join(SAVED_MODELS_PATH, f"z{log_counter}", "best_multi.pth"))
        else: counter += 1

        print(f"Epoch: {epoch + 1}/{n_epoch} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | f1: {f1:.4f} | recall: {recall:.4f} | precision: {precision:.4f}")
    if save:
        torch.save(model, os.path.join(SAVED_MODELS_PATH, f"z{log_counter}", f"multi.pth"))
        log_file.close()
        sys.stdout = sys.__stdout__
    return best_model

def train_step(x, y, model: nn.Module, optimizer: torch.optim.Optimizer, criterion: nn.Module) -> Tuple[float, torch.Tensor]:
    optimizer.zero_grad()

    output = model(*x)
    loss = criterion(output, y)
    preds = torch.argmax(output, dim=1)

    loss.backward()
    optimizer.step()

    return loss.item(), preds

def train_single(n_epoch: int,
                model: SingleBranch,
                lr: float,
                train_loader: DataLoader,
                val_loader: DataLoader,
                weights: torch.Tensor,
                patience: int=5) -> SingleBranch:
    model = model.to(device)
    optimizer = AdamW([
        {'params': model.clf_head.parameters(), 'lr': lr},
        {'params': [p for p in model.model_ax.parameters() if p.requires_grad], 'lr': 1e-6},
    ], weight_decay=1e-4)
    criterion = CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    scheduler = ReduceLROnPlateau(optimizer, patience=2)

    counter  = 0
    best_loss = float("inf")
    best_acc = 0.0
    best_metric = 0.0

    for epoch in range(n_epoch):
        if counter > patience:
            print(f"Early stopping at epoch: {epoch+1}/{n_epoch} with best val acc: {best_acc:.4f} and best val loss: {best_loss:4f}")
            break
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc=f"epoch №{epoch+1}")

        for img, label in progress_bar:
            img = img.to(device)
            label = label.to(device)

            loss, preds = train_step(img, label, model, optimizer, criterion)
            batch_size = label.size(0)

            train_loss += loss * batch_size
            correct += (preds == label).sum().item()
            total += batch_size
            progress_bar.set_postfix(train_loss=f"{train_loss/total:.4f}", train_acc=f"{correct/total:.4f}")
        train_acc  = correct / total
        train_loss /= total

        val_loss, metrics, cm = validate(model, criterion, val_loader)

        val_acc, f1, recall, precision = metrics
        scheduler.step(val_loss)

        if f1 > best_metric:
            counter = 0
            best_loss = val_loss
            best_acc  = val_acc
            best_metric = f1
        else: counter += 1

        print(f"Epoch: {epoch + 1}/{n_epoch} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | f1: {f1:.4f} | recall: {recall:.4f} | precision: {precision:.4f}")
    return model

def validate(model: Union[SingleBranch, MultiBranchAttention, MultiBranchConcat, MultiBranchMean], criterion: nn.Module, val_loader: DataLoader) -> Tuple[float, Tuple[float, float, float, float], np.ndarray]:
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    y_pred = []
    y_true = []
    
    with torch.no_grad():
        for img, label in val_loader:
            if isinstance(model, (MultiBranchAttention, MultiBranchConcat, MultiBranchMean)):
                ax, front, sag = img
                ax = ax.to(device)
                front = front.to(device)
                sag = sag.to(device)
                output = model(ax, front, sag)
            else:
                img = img.to(device)
                output = model(img)
            label = label.to(device)

            preds = torch.argmax(output, dim=1)
            y_pred.extend(preds.cpu().numpy().tolist())
            y_true.extend(label.cpu().numpy().tolist())
            batch_size = label.size(0)

            correct += (preds == label).sum().item()
            val_loss += criterion(output, label).item() * batch_size
            total += batch_size
        val_acc = correct / total
        val_loss /= total

    cm = confusion_matrix(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    recall = recall_score(y_true, y_pred, average="macro")
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)

    return val_loss, (val_acc, f1, recall, precision), cm

def cross_validate_pytorch(
    dataset: Union[AxisHolder, SliceHolder], 
    model_class: Callable,
    model_params: dict,
    train_func: Callable, 
    n_splits: int = 5,
    batch_size: int = 8,
):
    labels = np.array(dataset.labels)
    indices = np.arange(len(dataset))
    weights = torch.tensor(1 / np.array(dataset.counts), dtype=torch.float32, device=device)

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

        model = train_func(n_epoch=50, model=model, train_loader=train_loader, val_loader=val_loader, lr=0.0006520366113221881, weights=weights)

        model.eval()
        val_preds = []
        val_true = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                if isinstance(dataset, AxisHolder):
                    inputs = tuple(t.to(device) for t in inputs) 
                    outputs = model(*inputs)
                else:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                targets = targets.to(device)

                predictions = torch.argmax(outputs, dim=1)

                val_preds.extend(predictions.cpu().numpy())
                val_true.extend(targets.cpu().numpy())

        val_preds = np.array(val_preds)
        val_true = np.array(val_true)

        val_acc = (val_preds == val_true).mean()
        f1 = f1_score(val_true, val_preds, average="macro")
        recall = recall_score(val_true, val_preds, average="macro")
        precision = precision_score(val_true, val_preds, average="macro", zero_division=0)

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

def objective(trial: optuna.Trial, train_ds: Subset) -> float:
    base_model = trial.suggest_categorical(name="base_model", choices=["resnet18", "resnet34", "convnext_tiny", "convnext_small"])
    hidden_dim = trial.suggest_categorical(name="hidden_dim", choices=[64, 128, 256, 512, 1024])
    attention_dim = trial.suggest_categorical(name="attention_dim", choices=[64, 128, 256, 512, 1024])
    attention_heads = trial.suggest_categorical(name="attention_heads", choices=[4, 8, 16, 32])
    lr = trial.suggest_float("lr", low=1e-5, high=5e-2, log=True)
    
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
    
    full_labels = np.array(train_ds.dataset.labels)
    train_indices = train_ds.indices
    labels_subset = full_labels[train_indices]
    
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    fold_f1_scores = []
    
    for inner_train_idx, inner_val_idx in skf.split(train_indices, labels_subset):
        actual_train_idx = train_indices[inner_train_idx]
        actual_val_idx = train_indices[inner_val_idx]
        
        inner_train_ds = Subset(train_ds.dataset, actual_train_idx)
        inner_val_ds = Subset(train_ds.dataset, actual_val_idx)
        
        inner_train_ds.x_transforms = train_transforms
        
        train_loader = DataLoader(inner_train_ds, batch_size=8, shuffle=True, num_workers=1, pin_memory=True)
        val_loader = DataLoader(inner_val_ds, batch_size=8, shuffle=False, num_workers=1, pin_memory=True)
        
        model = MultiBranchAttention(
            base_model=base_model, 
            num_classes=6, 
            attention_dim=attention_dim,
            hidden_dim=hidden_dim, 
            attention_heads=attention_heads
        )
        
        _, counts = np.unique(labels_subset, return_counts=True)
        weights = torch.tensor(1.0 / counts, dtype=torch.float32, device=device)
        
        model = train_multi(
            n_epoch=50, 
            model=model, 
            lr=lr, 
            train_loader=train_loader, 
            val_loader=val_loader, 
            weights=weights,
            save=False,
            patience=3
        )
        
        criterion = CrossEntropyLoss(weight=weights)
        val_loss, metrics, _ = validate(model, criterion, val_loader)
        _, f1, _, _ = metrics
        fold_f1_scores.append(f1)
        
        del model, train_loader, val_loader, inner_train_ds, inner_val_ds
        torch.cuda.empty_cache()

    return np.mean(fold_f1_scores)

def save_embeddings(model: Union[MultiBranchAttention, MultiBranchMean, MultiBranchConcat, SingleBranch], train_loader: DataLoader, test_loader: DataLoader) -> None:
    model.clf_head = nn.Identity()
    model.eval()
    for loader, name in zip([train_loader, test_loader], ["train", "test"]):
        embeddings = []
        labels = []
        with torch.no_grad():
            for batch, label in loader:
                if not isinstance(model, SingleBranch):
                    ax, front, sag = batch
                    ax = ax.to(device)
                    front = front.to(device)
                    sag = sag.to(device)
                    batch = (ax, front, sag)
                else:
                    batch = batch.to(device)
                embeddings_batch = model(batch)
                embeddings.extend(embeddings_batch.cpu().squeeze(0).tolist())
                labels.extend(label.cpu().squeeze(0).tolist())
        labels = np.array(labels)
        embeddings = np.array(embeddings)
        np.savetxt(f"../embeddings/{name}_embed.txt", embeddings)
        np.savetxt(f"../embeddings/{name}_label.txt", labels, fmt="%d")

def save_confusion_matrix(cm: np.ndarray, labels: list) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(cmap="Purples", ax=ax)
    plt.tight_layout()
    plt.savefig("../report/utils/conf_mat.png", dpi=700)

def plot_attention_scores_full(attention_data: dict) -> None:
    fig, axes = plt.subplots(6, 3, figsize=(16, 12))
    fig.suptitle(f'Attention Weights by Heads', fontsize=16)
    
    titles = ['Query: Axial', 'Query: Frontal', 'Query: Sagittal']
    
    class_labels = {
        "control": "Control",
        "parkinson": "Parkinson",
        "alzheimer": "Alzheimer",
        "adhd": "ADHD",
        "sclerosis": "Sclerosis",
        "autism": "Autism"
    }
    
    for row, (class_name, content) in enumerate(attention_data.items()):
        data = [content["ax"], content["front"], content["sag"]]
        class_display_name = class_labels.get(class_name, class_name.capitalize())
        
        for idx, (ax, title, d) in enumerate(zip(axes[row], titles, data)):
            if d is not None:
                sns.heatmap(d, annot=True, fmt=".2f", cmap='Blues', 
                            xticklabels=['KV: Ax', 'KV: Front', 'KV: Sag'],
                            yticklabels=[f"Head {i + 1}" for i in range(16)],
                            vmin=0, vmax=1, ax=ax)
                if idx == 0:
                    ax.set_title(f'{class_display_name} - {title}', fontweight='bold')
                else:
                    ax.set_title(title)
                ax.set_ylabel('Attention Head')
                ax.set_xlabel('Key-Value Projections')
            else:
                ax.text(0.5, 0.5, 'No samples', ha='center', va='center')
                if idx == 0:
                    ax.set_title(f'{class_display_name} - {title}', fontweight='bold')
                else:
                    ax.set_title(title)
            
    plt.tight_layout()
    plt.show()

def plot_aggregated_matrices(attention_data: dict) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Агрегированные веса внимания (усредненные по головам)', fontsize=16)
    
    class_labels = {
        "control": "Контрольная группа", "parkinson": "Паркинсон", "alzheimer": "Альцгеймер",
        "adhd": "СДВГ", "sclerosis": "Склероз", "autism": "РАС"
    }
    
    for idx, (class_name, content) in enumerate(attention_data.items()):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        ax_agg = content["ax"].mean(axis=0)      # [3]
        front_agg = content["front"].mean(axis=0) # [3]
        sag_agg = content["sag"].mean(axis=0)     # [3]
        
        flow_matrix = np.vstack([ax_agg, front_agg, sag_agg])
        
        sns.heatmap(flow_matrix, annot=True, fmt=".2f", cmap='YlOrRd',
                    xticklabels=['KV: Ax', 'KV: Front', 'KV: Sag'],
                    yticklabels=['Q: Ax', 'Q: Front', 'Q: Sag'],
                    vmin=0, vmax=1, ax=ax, cbar=False)
        
        ax.set_title(class_labels.get(class_name, class_name), fontweight='bold', fontsize=12)
        ax.set_xlabel('KV проекция')
        ax.set_ylabel('Query проекция')
    
    plt.tight_layout()
    plt.savefig("../report/utils/attention_agg.png", dpi=700)
    
def main() -> None:
    x_base_transforms = tv.Compose(
    [
        tv.ToTensor(),
        tv.Resize((224, 224)),
        tv.Normalize(mean=[0.485, 0.456, 0.406],
                     std=[0.229, 0.224, 0.225]),
    ])
    
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
     
    multi_ds  = AxisHolder(REDUCED_DATASET_PATH, x_base_transforms)
    
    indices = np.arange(len(multi_ds))
    labels  = np.array(multi_ds.labels)

    train_idx, test_idx, _, _ = train_test_split(
        indices, labels, test_size=0.2, stratify=labels, random_state=42
    )

    train_multi_ds = Subset(multi_ds, train_idx)
    test_multi_ds  = Subset(multi_ds, test_idx)
    train_multi_ds.x_transforms = train_transforms

    best_params = {'base_model': 'convnext_tiny', "num_classes":6, 'hidden_dim': 64, 'attention_dim': 512, 'attention_heads': 16, "lr": 0.0006520366113221881}

    attention_model: MultiBranchAttention = MultiBranchAttention(
        base_model=best_params["base_model"],
        num_classes=6,
        attention_dim=best_params["attention_dim"],
        hidden_dim=best_params["hidden_dim"],
        attention_heads=best_params["attention_heads"],
    )
    
    train_multi_loader = DataLoader(train_multi_ds, batch_size=8, shuffle=True, num_workers=1, pin_memory=True, generator=g)
    test_multi_loader  = DataLoader(test_multi_ds, batch_size=4, shuffle=True, num_workers=1, pin_memory=True, generator=g)
    
    _, counts = np.unique(np.array(multi_ds.labels)[train_idx], return_counts=True)
    final_weights = torch.tensor(1.0 / counts, dtype=torch.float32, device=device)

    # model: MultiBranchAttention = train_multi(100, attention_model, best_params["lr"], train_multi_loader, test_multi_loader, final_weights, True, 10)

    model: MultiBranchAttention = torch.load("../models/z3/best_multi.pth", weights_only=False)
    model = model.to(device)
    model.eval()
    model.return_scores()
    
    attention_scores = {"ax": [], "front": [], "sag": []}
    
    all_labels = []
    
    with torch.no_grad():
        for img, label in test_multi_loader:
            img   = [i.to(device) for i in img]
            label = label.to(device)

            _, (ax_scores, front_scores, sag_scores) = model(*img)
            
            all_labels.append(label.cpu())
            attention_scores["ax"].append(ax_scores.cpu())
            attention_scores["front"].append(front_scores.cpu())
            attention_scores["sag"].append(sag_scores.cpu())
            
    class_means = {"control": {}, "parkinson": {}, "alzheimer": {}, "adhd": {}, "sclerosis": {}, "autism": {}}
    class_names = {0: "control", 1: "parkinson", 2: "alzheimer", 3: "adhd", 4: "sclerosis", 5: "autism"}
    all_labels = torch.cat(all_labels)
    ax_scores_all = torch.cat(attention_scores["ax"])
    front_scores_all = torch.cat(attention_scores["front"])
    sag_scores_all = torch.cat(attention_scores["sag"])
    
    for class_id, class_name in class_names.items():
        mask = (all_labels == class_id)
        ax_class_probs = ax_scores_all[mask]
        front_class_probs = front_scores_all[mask]
        sag_class_probs = sag_scores_all[mask]
        

        class_means[class_name]["ax"]    = ax_class_probs.mean(dim=0).numpy()
        class_means[class_name]["sag"]   = sag_class_probs.mean(dim=0).numpy()
        class_means[class_name]["front"] = front_class_probs.mean(dim=0).numpy()
        
    
    # plot_attention_scores_full(class_means)
    plot_aggregated_matrices(class_means)
            
    
if __name__ == "__main__":
    main()