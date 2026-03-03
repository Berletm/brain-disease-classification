from typing import Tuple, List

import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score, precision_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from utils.utils import SAVED_MODELS_PATH
from utils.logger import Tee

from tqdm import tqdm
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SimpleCLF(nn.Module):
    def __init__(self, input_shape: int=100, output_shape: int=3):
        super().__init__()

        self.layer1 = nn.Linear(in_features=input_shape, out_features=256)
        self.layer2 = nn.Linear(in_features=256, out_features=512)
        self.layer3 = nn.Linear(in_features=512, out_features=output_shape)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.layer3(x)

        return x

class ConvCLF(nn.Module):
    def __init__(self, input_shape=(128, 128, 3)):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = self.gap(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        x_tensor = torch.tensor(x, dtype=torch.float32).to(device)
        output = self.softmax(self.forward(x_tensor))

        return torch.argmax(output, dim=1).cpu().numpy()

class Attention(nn.Module):
    def __init__(self, feature_dim:int=512, hidden_dim:int=256, num_classes:int=4):
        super().__init__()
        self.attn_pool = nn.Sequential(
            nn.Linear(feature_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Softmax(dim=-1)
        )

    def forward(self, features) -> torch.Tensor:
        weights = self.attn_pool(features)

        return weights

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
    def __init__(self, hidden_dim:int=256, num_classes:int=4):
        super().__init__()

        self.model_ax = models.resnet18(weights="DEFAULT")
        self.model_front = models.resnet18(weights="DEFAULT")
        self.model_sag = models.resnet18(weights="DEFAULT")

        for model in [self.model_ax, self.model_sag, self.model_front]:
            for name, param in model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

        features_dim = self.model_ax.fc.in_features

        self.model_ax.fc = nn.Identity()
        self.model_front.fc = nn.Identity()
        self.model_sag.fc = nn.Identity()

        self.cross_attention = CrossAttention(features_dim, 4)

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

        logits = torch.cat([ax_logits, front_logits, sag_logits], dim=0)

        fused_logits = self.cross_attention(ax_logits, logits)

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
                model:    nn.Module,
                train_loader: DataLoader,
                val_loader  : DataLoader,
                weights:List|np.ndarray=[1.0, 1.0, 1.0, 1.0]) -> Tuple[nn.Module, np.ndarray]:
    model = model.to(device)
    optimizer = AdamW([
        {'params': model.clf_head.parameters(), 'lr': 1e-3},
        {'params': [p for p in model.model_ax.parameters() if p.requires_grad], 'lr': 1e-6},
        {'params': [p for p in model.model_sag.parameters() if p.requires_grad], 'lr': 1e-6},
        {'params': [p for p in model.model_front.parameters() if p.requires_grad], 'lr': 1e-6},
    ], weight_decay=1e-3)
    weights = torch.Tensor(weights, device)
    criterion = CrossEntropyLoss(label_smoothing=0.1, weight=weights)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)
    scheduler2 = ReduceLROnPlateau(optimizer, patience=10, min_lr=1e-7)

    patience = 20
    counter  = 0
    best_loss = float("inf")
    best_acc = 0.0
    best_metric = 0.0

    log_file = open(r"classification/training.log", "w+")

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
            torch.save(model.state_dict(), SAVED_MODELS_PATH+"/best_multi.pth")
        else: counter += 1


        scheduler.step()
        scheduler2.step(f1)
        print(f"Epoch: {epoch + 1}/{n_epoch} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"f1 weighted: {f1:.4f} | recall: {recall:.4f} | precision: {precision:.4f}")
    log_file.close()
    sys.stdout = sys.__stdout__
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

def cross_validate(model: nn.Module, X: np.ndarray, y: np.ndarray) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {
        'fold_accuracies': [],
        'fold_models': [],
        'fold_histories': [],
        'all_predictions': [],
        'all_true_labels': []
    }

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model, acc_hist = train(100, model, X_train, y_train, X_val, y_val, batch_size=50, shuffle=True)

        X_val_tensor = torch.from_numpy(X_val).float().to(device)
        y_val_tensor = torch.from_numpy(y_val).long().to(device)

        model.eval()
        with torch.no_grad():
            val_output = model(X_val_tensor)
            val_preds = torch.argmax(val_output, dim=1)
            val_acc   = (y_val_tensor == val_preds).float().mean().item()

        results['fold_accuracies'].append(val_acc)
        results['fold_models'].append(model)
        results['fold_histories'].append(acc_hist)
        results['all_predictions'].extend(val_preds.cpu().numpy())
        results['all_true_labels'].extend(y_val)

    results['mean_accuracy'] = np.mean(results['fold_accuracies'])
    results['std_accuracy'] = np.std(results['fold_accuracies'])
    overall_acc = accuracy_score(results['all_true_labels'], results['all_predictions'])
    results['overall_accuracy'] = overall_acc

    return results
