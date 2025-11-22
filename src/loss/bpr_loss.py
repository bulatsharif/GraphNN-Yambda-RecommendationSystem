import torch
import torch.nn as nn

class BPRLoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super(BPRLoss, self).__init__()
        self.reduction = reduction
        self.log_sigmoid = nn.LogSigmoid()

    def forward(self, positive_scores: torch.Tensor, negative_scores: torch.Tensor):
        diff = positive_scores - negative_scores

        loss = -self.log_sigmoid(diff)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
