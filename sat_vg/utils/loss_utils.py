import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x, target):
        n_classes = x.size(-1)
        with torch.no_grad():
            target = target.unsqueeze(1)
            one_hot = torch.zeros_like(x)
            one_hot.scatter_(1, target, 1)
            one_hot = one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        log_probs = F.log_softmax(x, dim=-1)
        loss = -(one_hot * log_probs).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class TransVGLoss(nn.Module):
    def __init__(self, label_smoothing=0.1, focal_loss=False):
        super(TransVGLoss, self).__init__()
        self.label_smoothing = label_smoothing
        self.focal_loss = focal_loss
        
        # For bounding box regression
        self.bbox_loss = nn.L1Loss()
        self.giou_loss = nn.SmoothL1Loss()

    def forward(self, outputs, targets):
        """
        Compute loss between model outputs and targets.
        
        Args:
            outputs: Tensor of shape [B, 4] containing predicted bounding box coordinates
            targets: Tensor of shape [B, 4] containing target bounding box coordinates
        
        Returns:
            Dictionary containing loss components
        """
        # Compute box regression loss
        loss_bbox = self.bbox_loss(outputs, targets)
        
        # Compute GIoU loss
        loss_giou = self.giou_loss(outputs, targets)
        
        # Total loss
        loss = loss_bbox + loss_giou
        
        return {
            'loss': loss,
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou
        } 