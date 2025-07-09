# active_boundary_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ABL(nn.Module):
    """
    Implementation based on https://github.com/wangchi95/active-boundary-loss/blob/main/abl.py
    Standalone, no external dependencies.
    """
    def __init__(self, nclass=9, theta0=3, theta=5, ignore_label=0):
        super().__init__()
        self.nclass = nclass
        self.theta0 = theta0
        self.theta = theta
        self.ignore_label = ignore_label

    def forward(self, logits, labels):
        '''
        logits: (B, C, H, W)
        labels: (B, H, W)
        '''
        # Softmax to probability
        pred = torch.softmax(logits, dim=1)
        pred = torch.argmax(pred, dim=1) # (B, H, W)

        # One-hot
        labels_one_hot = F.one_hot(labels, num_classes=self.nclass).permute(0,3,1,2).float()
        pred_one_hot   = F.one_hot(pred, num_classes=self.nclass).permute(0,3,1,2).float()

        # Compute boundaries
        gt_b = self._mask_to_boundary(labels_one_hot, self.theta0)
        pred_b = self._mask_to_boundary(pred_one_hot, self.theta)

        # Ignore index
        mask = (labels != self.ignore_label).float()

        # Dice on boundary
        intersect = (pred_b * gt_b * mask.unsqueeze(1)).sum()
        denom = ((pred_b + gt_b) * mask.unsqueeze(1)).sum() + 1e-7
        loss = 1 - 2 * intersect / denom
        return loss

    @staticmethod
    def _mask_to_boundary(mask, dilation_ratio=0):
        # mask: (B, C, H, W), binary
        if dilation_ratio == 0:
            return mask - F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
        else:
            # Dilation: slightly fatter boundaries
            dilated = F.max_pool2d(mask, kernel_size=2 * dilation_ratio + 1, stride=1, padding=dilation_ratio)
            boundary = dilated - F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
            return (boundary > 0).float()
