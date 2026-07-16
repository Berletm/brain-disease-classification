import torch

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma: int, weights: torch.Tensor, smoothing: float = 0.0):
        super().__init__()
        
        self.smoothing = smoothing
        self.weights = weights
        
        self.gamma = gamma
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        p = torch.nn.functional.softmax(inputs, dim=1)
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - p_t) ** self.gamma
        ce = torch.nn.functional.cross_entropy(inputs, targets, self.weights, reduction="none", label_smoothing=self.smoothing)
        loss = ce * focal_weight
        return loss.mean()