# SCRIPT DI TRAINING PER IL MODELLO SPECIALISTA DEL TERRENO

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import random
import glob
from typing import Tuple, List, Dict, Any
import cv2
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

import numpy as np
from PIL import Image
from tqdm import tqdm
import imagehash
import networkx as nx
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import TverskyLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =============================================================================
# 2. RIPRODUCIBILITÀ E CONFIGURAZIONE
# =============================================================================
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_config() -> Dict[str, Any]:
    base_path = "./"
    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "RANDOM_SEED": 42, "NUM_WORKERS": 2, "USE_AMP": True,
        
        # --- Percorsi Dati (gli stessi di prima) ---
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_specialist_model.pth"),

        # --- Modello Specialista ---
        "ENCODER": "efficientnet-b2", # Più leggero, adatto al compito
        "ENCODER_WEIGHTS": "imagenet",
        "OUTPUT_SIZE": (384, 768),
        "BATCH_SIZE": 8, # Possiamo aumentare il batch size con un modello più leggero
        "VALIDATION_SPLIT": 0.2,

        # --- Classi di Interesse per lo Specialista ---
        "TARGET_CLASSES": {2: 0, 3: 1, 6: 2, 7: 3}, # Mappa da ID originali a ID del sub-problema
        "NUM_SPECIALIST_CLASSES": 4,
        "IGNORE_INDEX_SPECIALIST": -1, # Usiamo -1 per ignorare tutto il resto

        # --- Training ---
        "NUM_EPOCHS": 25,
        "LEARNING_RATE": 1e-3, # LR più alto, compito più semplice
        "WEIGHT_DECAY": 1e-5,
    }
    config["CLASS_NAMES_SPECIALIST"] = ["Spec_Rough", "Spec_Smooth", "Spec_LowVeg", "Spec_Puddle"]
    return config

# =============================================================================
# 3. DATASET SPECIALIZZATO (`SubproblemDataset`)
# =============================================================================
class SubproblemDataset(Dataset):
    def __init__(self, sample_dirs: List[str], transform: A.Compose, config: Dict[str, Any]):
        self.transform = transform
        self.config = config
        self.target_class_keys = list(config["TARGET_CLASSES"].keys())
        self.class_map = np.full(256, config["IGNORE_INDEX_SPECIALIST"], dtype=np.int64)
        for original_id, new_id in config["TARGET_CLASSES"].items():
            self.class_map[original_id] = new_id
        
        # Filtra i percorsi per includere solo quelli con le classi di interesse
        self.sample_dirs = self._filter_dirs(sample_dirs)

    def _filter_dirs(self, all_dirs):
        tqdm.write("Filtraggio del dataset per le classi di interesse...")
        filtered_dirs = []
        for path in tqdm(all_dirs, desc="Filtraggio"):
            mask = np.array(Image.open(os.path.join(path, 'labels.png')))
            if np.any(np.isin(mask, self.target_class_keys)):
                filtered_dirs.append(path)
        tqdm.write(f"Dataset filtrato: {len(filtered_dirs)} / {len(all_dirs)} campioni contengono le classi target.")
        return filtered_dirs

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        image = np.array(Image.open(os.path.join(sample_dir, 'rgb.jpg')).convert("RGB"))
        mask_original = np.array(Image.open(os.path.join(sample_dir, 'labels.png')))
        
        # Rimappatura efficiente con NumPy
        mask_remapped = self.class_map[mask_original]
        
        augmented = self.transform(image=image, mask=mask_remapped)
        return augmented['image'], augmented['mask']

def get_transforms(config: Dict[str, Any], is_train: bool) -> A.Compose:
    if is_train:
        return A.Compose([
            A.Resize(height=config["OUTPUT_SIZE"][0], width=config["OUTPUT_SIZE"][1], interpolation=cv2.INTER_NEAREST),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5),
            A.ColorJitter(p=0.5),
            A.CoarseDropout(p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else: # Validazione
        return A.Compose([
            A.Resize(height=config["OUTPUT_SIZE"][0], width=config["OUTPUT_SIZE"][1], interpolation=cv2.INTER_NEAREST),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

# =============================================================================
# 4. FUNZIONI DI TRAINING E VALIDAZIONE (Adattate)
# =============================================================================
def get_model(config: Dict[str, Any]) -> nn.Module:
    model = smp.Unet(
        encoder_name=config["ENCODER"],
        encoder_weights=config["ENCODER_WEIGHTS"],
        in_channels=3,
        classes=config["NUM_SPECIALIST_CLASSES"] # Output a 4 classi
    )
    tqdm.write(f"Creazione del MODELLO SPECIALISTA: U-Net con encoder '{config['ENCODER']}'")
    return model

def evaluate_specialist(model, loader, criterion, device, config):
    model.eval()
    total_loss = 0.0
    # Usiamo il calcolo dell'IoU standard, ma solo sulle 4 classi
    all_preds_flat, all_labels_flat = [], []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione Specialista", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            
            with autocast(enabled=config["USE_AMP"], device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            valid = masks != config["IGNORE_INDEX_SPECIALIST"]
            all_preds_flat.append(preds[valid].cpu())
            all_labels_flat.append(masks[valid].cpu())
    
    avg_loss = total_loss / len(loader)
    
    all_preds = torch.cat(all_preds_flat).numpy()
    all_labels = torch.cat(all_labels_flat).numpy()
    
    # Calcolo IoU per le 4 classi
    iou_per_class = np.zeros(config["NUM_SPECIALIST_CLASSES"])
    for cls in range(config["NUM_SPECIALIST_CLASSES"]):
        intersection = np.logical_and(all_preds == cls, all_labels == cls).sum()
        union = np.logical_or(all_preds == cls, all_labels == cls).sum()
        iou_per_class[cls] = intersection / union if union > 0 else np.nan
        
    mean_iou = np.nanmean(iou_per_class)
    return avg_loss, mean_iou, iou_per_class

# Funzione di training (rimane generica e funziona bene)
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train(); running_loss = 0.0; progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device, dtype=torch.long)
        with autocast(enabled=use_amp, device_type=device.type):
            outputs = model(images); loss = criterion(outputs, masks)
        optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        running_loss += loss.item(); progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader)

# =============================================================================
# 5. FUNZIONE PRINCIPALE
# =============================================================================
def main():
    config = get_config(); seed_everything(config["RANDOM_SEED"]); device = config["DEVICE"]
    tqdm.write(f"Device: {device} | Strategia: Training Modello Specialista del Terreno")

    # Carica TUTTI i percorsi del dataset, il filtraggio avverrà nella classe Dataset
    paths = {"original": config["DATASET_PATH_TRAIN"], "augmented": config["DATASET_PATH_AUG"], "aggressive": config["DATASET_PATH_AUG_AGGR"]}
    all_dirs = [d for p in paths.values() if os.path.isdir(p) for d in glob.glob(os.path.join(p, "*")) if os.path.isdir(d)]
    all_dirs = sorted(list(set(all_dirs)))

    train_dirs, val_dirs = train_test_split(all_dirs, test_size=config["VALIDATION_SPLIT"], random_state=config["RANDOM_SEED"])

    train_transforms = get_transforms(config, is_train=True)
    val_transforms = get_transforms(config, is_train=False)
    
    train_dataset = SubproblemDataset(train_dirs, train_transforms, config)
    val_dataset = SubproblemDataset(val_dirs, val_transforms, config)
    
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"], pin_memory=True)

    model = get_model(config).to(device)
    criterion = TverskyLoss(mode='multiclass', from_logits=True, ignore_index=config["IGNORE_INDEX_SPECIALIST"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    scaler = GradScaler(enabled=config["USE_AMP"])

    best_iou = -1.0
    
    tqdm.write("\n--- Inizio Training Specialista ---")
    for epoch in range(config["NUM_EPOCHS"]):
        tqdm.write(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, config["USE_AMP"])
        val_loss, val_miou, val_iou_per_class = evaluate_specialist(model, val_loader, criterion, device, config)
        scheduler.step(val_miou)
        
        tqdm.write(f"Risultati Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        for i, iou in enumerate(val_iou_per_class):
            tqdm.write(f"    - IoU {config['CLASS_NAMES_SPECIALIST'][i]}: {iou:.4f}")

        if val_miou > best_iou:
            best_iou = val_miou
            torch.save({'model_state_dict': model.state_dict(), 'config': config}, config["MODEL_SAVE_PATH"])
            tqdm.write(f"\n>>> 🎉 Nuovo modello specialista migliore salvato! mIoU: {best_iou:.4f} <<<")

    tqdm.write("\n--- Training Specialista Completato ---")
    tqdm.write(f"Miglior mIoU ottenuto: {best_iou:.4f}")
    tqdm.write(f"Modello salvato in: {config['MODEL_SAVE_PATH']}")

# =============================================================================
# 6. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()