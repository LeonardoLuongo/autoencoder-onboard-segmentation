# -*- coding: utf-8 -*-
"""
Script per l'addestramento di un modello di segmentazione semantica U-Net
con un encoder ResNet34 pre-addestrato. Utilizza una loss combinata per
ottimizzare le performance su dataset con classi sbilanciate.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import glob
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torchvision import transforms

import numpy as np
from PIL import Image
from tqdm import tqdm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import sys

# =============================================================================
# 2. CONFIGURAZIONE
# =============================================================================
def get_config() -> Dict[str, Any]:
    """Restituisce un dizionario con tutte le configurazioni dello script."""
    
    IS_COLAB = 'google.colab' in sys.modules

    if IS_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = "/content/drive/MyDrive/ML_Project/"
    else:
        base_path = "./"

    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        
        # Percorsi
        "DATASET_PATH": os.path.join(base_path, "dataset/training/train"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_model_advanced.pth"),

        # Iperparametri di addestramento
        "NUM_EPOCHS": 20,
        "BATCH_SIZE": 8,
        "LEARNING_RATE": 1e-4,
        "NUM_CLASSES": 9,
        "IGNORE_INDEX": 0,

        # Parametri del dataset
        "OUTPUT_SIZE": (256, 512), # Altezza, Larghezza
        "DATASET_MEAN": [0.4278, 0.4464, 0.4231],
        "DATASET_STD": [0.2543, 0.2535, 0.2567],
        "VALIDATION_SPLIT": 0.2,
        "RANDOM_SEED": 42,

        # Parametri del modello
        "ENCODER": "resnet34",
        "ENCODER_WEIGHTS": "imagenet",
    }
    return config

# =============================================================================
# 3. CLASSE DATASET
# =============================================================================
class SegmentationDataset(Dataset):
    """
    Dataset personalizzato per la segmentazione semantica.
    Carica immagini e le relative maschere, applicando trasformazioni.
    """
    def __init__(self, sample_dirs: List[str], output_size: Tuple[int, int], 
                 mean: List[float], std: List[float], is_train: bool = True):
        """
        Args:
            sample_dirs (List[str]): Lista dei percorsi alle cartelle dei campioni.
            output_size (Tuple[int, int]): Dimensione di output (H, W) per immagini e maschere.
            mean (List[float]): Media per la normalizzazione del dataset.
            std (List[float]): Deviazione standard per la normalizzazione.
            is_train (bool): Se True, applica data augmentation.
        """
        self.sample_dirs = sample_dirs
        self.output_size = output_size
        self.is_train = is_train
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        img_path = os.path.join(sample_dir, 'rgb.jpg')
        mask_path = os.path.join(sample_dir, 'labels.png')
        
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Trasformazioni base
        image = TF.resize(image, self.output_size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.output_size, interpolation=TF.InterpolationMode.NEAREST)
        
        # Data Augmentation (solo per il training set)
        if self.is_train:
            if torch.rand(1) < 0.5:  # Flip orizzontale
                image = TF.hflip(image)
                mask = TF.hflip(mask)

        # Conversione a tensore e normalizzazione
        image = TF.to_tensor(image)
        image = self.normalize(image)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        return image, mask

# =============================================================================
# 4. FUNZIONI HELPER
# =============================================================================
def create_splits(data_dir: str, val_percent: float, seed: int) -> Tuple[List[str], List[str]]:
    """Trova tutti i campioni e li divide in set di training e validazione."""
    all_sample_dirs = sorted([
        d for d in glob.glob(os.path.join(data_dir, "*")) 
        if os.path.isdir(d) and not os.path.basename(d).startswith('__')
    ])
    
    n_samples = len(all_sample_dirs)
    n_val = int(n_samples * val_percent)
    n_train = n_samples - n_val
    
    generator = torch.Generator().manual_seed(seed)
    train_dirs, val_dirs = random_split(all_sample_dirs, [n_train, n_val], generator=generator)
    
    print(f"Trovati {n_samples} campioni. Split: {len(train_dirs)} train, {len(val_dirs)} val.")
    return list(train_dirs), list(val_dirs)

def combined_loss_fn(pred: torch.Tensor, target: torch.Tensor, 
                     ce_loss: nn.Module, dice_loss: nn.Module) -> torch.Tensor:
    """
    Calcola una loss combinata. Dà più peso alla Dice Loss per aiutare a
    gestire lo sbilanciamento delle classi.
    """
    return 0.4 * ce_loss(pred, target) + 0.6 * dice_loss(pred, target)

def compute_iou_per_class(mask_true: np.ndarray, mask_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Calcola l'IoU per ogni classe (escludendo la classe 0, background)."""
    iou_scores = np.zeros(num_classes - 1)
    for label in range(1, num_classes):
        intersection = np.sum((mask_true == label) & (mask_pred == label))
        union = np.sum((mask_true == label) | (mask_pred == label))
        iou_scores[label - 1] = intersection / union if union > 0 else np.nan
    return iou_scores

# =============================================================================
# 5. FUNZIONI DI TRAINING E VALIDAZIONE
# =============================================================================
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                    criterion_ce: nn.Module, criterion_dice: nn.Module, device: torch.device) -> float:
    """Esegue un'epoca di training e restituisce la loss media."""
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(loader, desc="Training")
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = combined_loss_fn(outputs, masks, criterion_ce, criterion_dice)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    return running_loss / len(loader)

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int) -> float:
    """Valuta il modello sul set di validazione e restituisce il mIoU."""
    model.eval()
    all_ious = []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione"):
            images = images.to(device)
            masks_np = masks.numpy()
            
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            
            for i in range(images.size(0)):
                iou_scores = compute_iou_per_class(masks_np[i], preds[i], num_classes)
                all_ious.append(iou_scores)
    
    mean_iou = np.nanmean(all_ious)
    return float(mean_iou)

# =============================================================================
# 6. FUNZIONE PRINCIPALE
# =============================================================================
def main():
    """Funzione principale che orchestra il processo di addestramento."""
    
    # --- Setup ---
    config = get_config()
    device = config["DEVICE"]
    print(f"Device in uso: {device}")
    print(f"Current working directory: {os.getcwd()}")
    
    # --- Data Loading ---
    train_dirs, val_dirs = create_splits(
        config["DATASET_PATH"], config["VALIDATION_SPLIT"], config["RANDOM_SEED"]
    )
    
    train_dataset = SegmentationDataset(
        train_dirs, config["OUTPUT_SIZE"], config["DATASET_MEAN"], 
        config["DATASET_STD"], is_train=True
    )
    val_dataset = SegmentationDataset(
        val_dirs, config["OUTPUT_SIZE"], config["DATASET_MEAN"], 
        config["DATASET_STD"], is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, 
        num_workers=max(1, os.cpu_count()//2), pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, 
        num_workers=max(1, os.cpu_count()//2), pin_memory=True
    )

    # --- Modello, Loss e Ottimizzatore ---
    model = smp.Unet(
        encoder_name=config["ENCODER"],
        encoder_weights=config["ENCODER_WEIGHTS"],
        in_channels=3,
        classes=config["NUM_CLASSES"],
    ).to(device)

    criterion_ce = nn.CrossEntropyLoss(ignore_index=config["IGNORE_INDEX"])
    criterion_dice = DiceLoss(mode='multiclass')
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])

    # --- Ciclo di Training e Validazione ---
    best_iou = -1.0
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_ce, criterion_dice, device)
        val_iou = evaluate(model, val_loader, device, config["NUM_CLASSES"])
        
        print(f"Epoch {epoch+1} -> Training Loss: {train_loss:.4f} | Validation mIoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f">>> Nuovo modello migliore salvato in '{config['MODEL_SAVE_PATH']}' con mIoU: {best_iou:.4f}")

    print("\nAddestramento completato.")
    print(f"Miglior mIoU di validazione ottenuto: {best_iou:.4f}")

# =============================================================================
# 7. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()