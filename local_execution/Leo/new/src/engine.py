# src/engine.py
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Gestione import AMP per retrocompatibilità
try:
    from torch.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    class autocast:
        def __init__(self, enabled=False, device_type='cpu', **kwargs): self.enabled = enabled
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    class GradScaler:
        def __init__(self, enabled=False): self.enabled = enabled
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass

def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer, criterion: nn.Module, scaler: GradScaler, device: torch.device, use_amp: bool) -> float:
    """Esegue un'epoca di training."""
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device, dtype=torch.long)

        with autocast(enabled=use_amp, device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    return running_loss / len(loader)

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device, config: Dict[str, Any]) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Esegue la validazione e calcola le metriche."""
    model.eval()
    total_loss = 0.0
    num_classes = config['data']['num_classes']
    ignore_index = config['data']['ignore_index']
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione", leave=False):
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)
            
            with autocast(enabled=True, device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks)

            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            # Calcolo efficiente della matrice di confusione
            masks_flat, preds_flat = masks.flatten(), preds.flatten()
            indices = masks_flat * num_classes + preds_flat
            conf_matrix += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)

    # Calcolo mIoU dalla matrice di confusione
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    
    iou_per_class = (tp / (tp + fp + fn + 1e-6))
    
    # Esclude ignore_index dal calcolo del mIoU
    valid_classes_mask = torch.ones(num_classes, dtype=torch.bool)
    if ignore_index is not None:
        valid_classes_mask[ignore_index] = False
        
    mean_iou = iou_per_class[valid_classes_mask].mean().cpu().item()
    
    return total_loss / len(loader), mean_iou, iou_per_class.cpu().numpy(), conf_matrix.cpu().numpy()