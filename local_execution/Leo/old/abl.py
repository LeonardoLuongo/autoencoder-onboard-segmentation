import torch
import torch.nn as nn
import torch.nn.functional as F

class ABL(nn.Module):
    """
    Active Boundary Loss (versione differenziabile e corretta).
    
    Questa loss calcola la Dice Loss tra i bordi della predizione "soft" (probabilità)
    e i bordi del ground truth "hard" (etichette).
    
    Correzioni principali:
    1. Calcolo del bordo corretto come (dilated - mask) per ottenere valori positivi.
    2. Correzione del bug nel denominatore della Dice Loss.
    """
    def __init__(self, nclass=9, ignore_label=0, dilation_k=3):
        super().__init__()
        self.nclass = nclass
        self.ignore_label = ignore_label
        # Un solo parametro per la dimensione del bordo (kernel size)
        self.dilation_k = dilation_k 
        self.padding = (dilation_k - 1) // 2

    def forward(self, logits, labels):
        '''
        logits: Tensore di output del modello (B, C, H, W)
        labels: Ground truth (B, H, W)
        '''
        # 1. Prepara i tensori di input
        soft_pred = torch.softmax(logits, dim=1)
        labels_one_hot = F.one_hot(labels, num_classes=self.nclass).permute(0, 3, 1, 2).float()
        
        # Maschera per ignorare un'etichetta specifica
        valid_pixel_mask = (labels != self.ignore_label).unsqueeze(1).float()

        # 2. Estrai i bordi (ora con la formula corretta)
        gt_b = self._get_boundary(labels_one_hot)
        pred_b = self._get_boundary(soft_pred)

        # 3. Calcola la Dice Loss sui bordi (con denominatore corretto)
        # Applica la maschera dei pixel validi a tutte le componenti
        gt_b_masked = gt_b * valid_pixel_mask
        pred_b_masked = pred_b * valid_pixel_mask
        
        intersect = (pred_b_masked * gt_b_masked).sum()
        denom = pred_b_masked.sum() + gt_b_masked.sum() + 1e-7
        
        # La loss è 1 meno il coefficiente di Dice
        loss = 1.0 - (2.0 * intersect) / denom
        
        return loss

    def _get_boundary(self, mask):
        """
        Estrae un bordo esterno positivo da una maschera (hard o soft).
        Usa la formula: dilated - mask
        """
        dilated = F.max_pool2d(
            mask, 
            kernel_size=self.dilation_k, 
            stride=1, 
            padding=self.padding
        )
        boundary = dilated - mask
        return boundary