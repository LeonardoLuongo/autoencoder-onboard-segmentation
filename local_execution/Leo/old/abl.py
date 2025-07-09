# abl.py (versione corretta e differenziabile)

import torch
import torch.nn as nn
import torch.nn.functional as F

class ABL(nn.Module):
    """
    Versione differenziabile della Active Boundary Loss.
    Opera sulle probabilità della softmax invece che sulle predizioni "hard" di argmax.
    È adatta per il training di una rete neurale.
    
    Implementazione basata sull'idea di https://github.com/wangchi95/active-boundary-loss/blob/main/abl.py
    ma resa differenziabile.
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
        # Softmax per ottenere le probabilità "soft"
        soft_pred = torch.softmax(logits, dim=1)

        # One-hot per le etichette (ground truth)
        labels_one_hot = F.one_hot(labels, num_classes=self.nclass).permute(0, 3, 1, 2).float()

        # Calcola i bordi del ground truth (sulla maschera hard)
        gt_b = self._mask_to_boundary(labels_one_hot, self.theta0)

        # Calcola i bordi della predizione (sulla mappa di probabilità soft)
        # Questa operazione è ora completamente differenziabile
        pred_b = self._mask_to_boundary(soft_pred, self.theta)

        # Maschera per ignorare un'etichetta specifica
        mask = (labels != self.ignore_label).unsqueeze(1).float()

        # Dice Loss sui bordi
        intersect = (pred_b * gt_b * mask).sum()
        denom = (pred_b.sum() + gt_b.sum()) * mask.sum() + 1e-7 # Uso la somma separata per stabilità
        # Originale: denom = ((pred_b + gt_b) * mask).sum() + 1e-7
        
        loss = 1 - 2 * intersect / denom
        return loss

    @staticmethod
    def _mask_to_boundary(mask, dilation_ratio=0):
        """
        Estrae i bordi da una maschera o da una mappa di probabilità.
        L'operazione 'mask - max_pool(mask)' crea un gradiente sui bordi.
        """
        if dilation_ratio == 0:
            # max_pool con kernel 3x3 è un'operazione di erosione
            eroded = F.max_pool2d(mask, kernel_size=3, stride=1, padding=1)
            boundary = mask - eroded
        else:
            # Dilatazione opzionale per bordi più spessi
            dilated = F.max_pool2d(mask, kernel_size=2 * dilation_ratio + 1, stride=1, padding=dilation_ratio)
            eroded = F.max_pool2d(dilated, kernel_size=3, stride=1, padding=1)
            boundary = dilated - eroded
        
        return boundary.float()