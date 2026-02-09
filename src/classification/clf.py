from typing import Tuple

import torchvision.models as models
import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau

from dim_reduction.utils import SAVED_MODELS_PATH
from rl_env import ReinforceEnvironment, train_on_session, generate_session

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

class MultiCLF(nn.Module):
    def __init__(self):
        super().__init__()

        self.model_ax = models.resnet18(weights="DEFAULT")
        self.model_front = models.resnet18(weights="DEFAULT")
        self.model_sag = models.resnet18(weights="DEFAULT")

        for model in [self.model_ax, self.model_sag, self.model_front]:
            for name, param in model.named_parameters():
                if 'layer4' not in name and 'fc' not in name:
                    param.requires_grad = False

        head_in = self.model_ax.fc.in_features

        self.model_ax.fc = nn.Identity()
        self.model_front.fc = nn.Identity()
        self.model_sag.fc = nn.Identity()

        self.clf_head = nn.Sequential(
            nn.BatchNorm1d(head_in),
            nn.Dropout(0.55),
            nn.Linear(head_in, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )

        self.softmax = nn.Softmax()

    def forward(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        ax, front, sag = x

        ax_logits    = self.model_ax(ax)
        front_logits = self.model_front(front)
        sag_logits   = self.model_sag(sag)

        logits = (ax_logits + front_logits + sag_logits) / 3

        return self.clf_head(logits)

    def predict(self, x: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> np.ndarray:
        ax, front, sag = x

        ax_logits = self.model_ax(ax)
        front_logits = self.model_front(front)
        sag_logits = self.model_sag(sag)

        logits = (ax_logits + front_logits + sag_logits) / 3

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
          shuffle:    bool) -> [nn.Module, np.ndarray]:
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
                val_loader  : DataLoader) -> [nn.Module, np.ndarray]:
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
                val_loader  : DataLoader) -> [nn.Module, np.ndarray]:
    model = model.to(device)
    optimizer = AdamW([
        {'params': model.clf_head.parameters(), 'lr': 1e-4},
        {'params': [p for p in model.model_ax.parameters() if p.requires_grad], 'lr': 1e-6},
        {'params': [p for p in model.model_sag.parameters() if p.requires_grad], 'lr': 1e-6},
        {'params': [p for p in model.model_front.parameters() if p.requires_grad], 'lr': 1e-6},
    ], weight_decay=1e-3)
    criterion = CrossEntropyLoss(label_smoothing=0.1)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, eta_min=1e-6)

    patience = 20
    counter  = 0
    best_metric = 0.0

    for epoch in range(n_epoch):
        if counter > patience:
            print(f"Early stopping at epoch: {epoch+1}/{n_epoch} with best val acc: {best_metric}")
            break
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:

            ax, front, sag = images
            ax = ax.to(device)
            front = front.to(device)
            sag = sag.to(device)
            images = (ax, front, sag)
            labels = labels.to(device)

            loss, preds = train_step(images, labels, model, optimizer, criterion)

            train_loss += loss
            correct += (preds == labels).sum().item()
            total += len(labels)
        train_loss /= total
        train_acc   = correct / total

        val_loss, val_acc = validate(model, criterion, val_loader)

        if val_acc > best_metric:
            counter = 0
            best_metric = val_acc
            torch.save(model.state_dict(), SAVED_MODELS_PATH+"/multi_clf.pth")
        else: counter += 1

        scheduler.step()
        print(f"Epoch: {epoch + 1}/{n_epoch} | Val loss: {val_loss:.4f} | Val acc: {val_acc:.4f} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")

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
            correct += (preds == labels).sum().item()
            val_loss += criterion(output, labels).item()
            total += len(labels)
        val_loss /= total
        val_acc = correct / total

    return val_loss, val_acc

def train_rl(n_epoch: int,
             agent: nn.Module,
             X_val: np.ndarray,
             y_val: np.ndarray,
             env: ReinforceEnvironment) -> [nn.Module, np.ndarray]:

    agent = agent.to(device)
    rewards_hist = []
    optimizer = AdamW(agent.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(n_epoch):
        rewards = [train_on_session(agent, optimizer, *generate_session(env, agent, 100, epoch), entropy_coef=1e-3) for _ in range(10)]

        rewards_hist.extend(rewards)

        with torch.no_grad():
            X_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
            output = agent(X_tensor).detach().cpu().numpy()
            y_pred = np.argmax(output, axis=1)
            acc = accuracy_score(y_val, y_pred)

            print(f"epoch {epoch}/{n_epoch}, acc: {acc:.3f}, mean reward: {np.mean(rewards):.3f}")

    return agent, rewards_hist


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
