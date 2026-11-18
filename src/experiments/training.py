import numpy as np
import torch
from torch.utils.data import DataLoader

from src.evaluation.protocol import oof_metrics
from src.models.losses import FocalLoss


def fit(train, valid, params, cfg, epochs, lr, label_count, training_log=None):
    from src.models import architectures as models

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device)
    model_params = {**params, 'device': cfg.device}
    model = getattr(models, cfg.experiment.model)(**model_params)

    backbone_ids = {
        id(p)
        for name in ('model_ax', 'model_front', 'model_sag')
        if hasattr(model, name)
        for p in getattr(model, name).parameters()
    }

    optimizer = torch.optim.AdamW(
        [
            {
                'params': [
                    p for p in model.parameters()
                    if p.requires_grad and id(p) not in backbone_ids
                ],
            },
            {
                'params': [
                    p for p in model.parameters()
                    if p.requires_grad and id(p) in backbone_ids
                ],
                'lr': 1e-6,
            },
        ],
        lr=lr,
    )

    counts = np.bincount(train.labels, minlength=label_count)
    if (counts == 0).any():
        raise ValueError('Training partition is missing a class')

    criterion = FocalLoss(
        cfg.loss.gamma,
        torch.tensor(1 / counts, dtype=torch.float32, device=device),
        cfg.loss.smoothing,
    )

    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(
        train,
        batch_size=cfg.cv.batch_size,
        shuffle=True,
        generator=generator,
    )

    best_score, best_epoch = -1., 1

    for epoch in range(1, epochs + 1):
        model.train()

        for inputs, targets in loader:
            optimizer.zero_grad()
            output = model(*[x.to(device) for x in inputs])
            loss = criterion(output, targets.to(device))

            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')

            loss.backward()

            if training_log is not None:
                squared_norm = sum(
                    parameter.grad.detach().square().sum().item()
                    for parameter in model.clf_head.parameters()
                    if parameter.grad is not None
                )
                training_log.append({
                    'epoch': epoch,
                    'loss': loss.item(),
                    'head_gradient_norm': squared_norm ** 0.5,
                })

            optimizer.step()

        if valid is not None:
            predicted, _ = predict(model, valid, cfg)
            score = oof_metrics(
                valid.labels, predicted, list(range(label_count)),
            )['macro_f1']
            if score > best_score:
                best_score, best_epoch = score, epoch

    return model, best_score, best_epoch


def predict(model, dataset, cfg):
    model.eval()
    probabilities = []

    with torch.no_grad():
        for inputs, _ in DataLoader(dataset, batch_size=cfg.cv.batch_size, shuffle=False):
            probabilities.extend(
                torch.softmax(
                    model(*[x.to(cfg.device) for x in inputs]),
                    dim=1,
                ).cpu().tolist()
            )

    probabilities = np.asarray(probabilities)

    return probabilities.argmax(axis=1), probabilities
