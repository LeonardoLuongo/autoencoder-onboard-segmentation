# SCRIPT DI TRAINING FINALE CON AUGMENTATION OFFLINE E ANALISI AVANZATA

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import sys
import glob
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from PIL import Image
from tqdm import tqdm
import imagehash
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 2. CONFIGURAZIONE
# =============================================================================
def get_config() -> Dict[str, Any]:
    """Configurazione per l'esperimento finale."""
    base_path = "./"
    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "final_model_efficientnet-b4_dice_focal.pth"),
        
        "ENCODER": "efficientnet-b4",
        "ENCODER_WEIGHTS": "imagenet",
        "OUTPUT_SIZE": (384, 768),
        "BATCH_SIZE": 4,

        "NUM_EPOCHS": 30,
        "LEARNING_RATE": 1e-4,
        "WEIGHT_DECAY": 1e-5,

        "NUM_CLASSES": 9,
        "IGNORE_INDEX": 0,
        "VALIDATION_SPLIT": 0.2,
        "RANDOM_SEED": 42,
        "NUM_WORKERS": 0,
        "USE_AMP": True,
        
        "RARE_CLASS_INDICES": [2, 6, 8], # Rough Trail, LowVeg, Obstacle
    }
    
    config["CLASS_NAMES"] = [
        "Ignored", "Sky", "Rough", "Smooth", "Grass", 
        "HighVeg", "LowVeg", "Debris", "Obstacle"
    ]
    return config

# =============================================================================
# 3. GESTIONE DATI (DA SCRIPT ORIGINALE)
# =============================================================================
def get_grouped_stratified_split(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    paths = {
        "original": config["DATASET_PATH_TRAIN"],
        "augmented": config["DATASET_PATH_AUG"],
        "aggressive": config["DATASET_PATH_AUG_AGGR"],
    }
    all_dirs = []
    print("Caricamento percorsi dei dataset...")
    for key, path in paths.items():
        dirs = [d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)]
        print(f"Trovati {len(dirs)} campioni in '{key}'.")
        all_dirs.extend(dirs)
    
    print(f"Dimensione totale del dataset di training: {len(all_dirs)} campioni.")
    all_dirs = sorted(list(set(all_dirs))) # Rimuovi duplicati e ordina

    hashes = {path: imagehash.phash(Image.open(os.path.join(path, 'rgb.jpg'))) for path in tqdm(all_dirs, desc="Calcolo Hash")}
    
    G = nx.Graph()
    G.add_nodes_from(hashes.keys())
    path_list = list(hashes.keys())
    for i in range(len(path_list)):
        for j in range(i + 1, len(path_list)):
            if hashes[path_list[i]] - hashes[path_list[j]] < 5:
                G.add_edge(path_list[i], path_list[j])
                
    image_groups = list(nx.connected_components(G))
    print(f"Trovati {len(image_groups)} gruppi di immagini uniche.")
    
    group_labels = []
    for group in tqdm(image_groups, desc="Etichettatura gruppi per lo split"):
        has_rare = any(
            rare_id in np.unique(Image.open(os.path.join(p, 'labels.png')).convert("L"))
            for p in group for rare_id in config["RARE_CLASS_INDICES"]
        )
        group_labels.append(1 if has_rare else 0)
        
    train_groups, val_groups = train_test_split(
        image_groups, test_size=config["VALIDATION_SPLIT"],
        random_state=config["RANDOM_SEED"], stratify=group_labels
    )
    
    train_dirs = [p for group in train_groups for p in group]
    val_dirs = [p for group in val_groups for p in group]
    
    print(f"Split stratificato creato: {len(train_dirs)} train, {len(val_dirs)} validation.")
    return train_dirs, val_dirs

def get_transforms(output_size: Tuple[int, int]) -> A.Compose:
    """Restituisce la pipeline di solo preprocessing (no augmentation online)."""
    return A.Compose([
        A.Resize(height=output_size[0], width=output_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class SegmentationDataset(Dataset):
    """Dataset che carica le maschere pre-etichettate (non RGB)."""
    def __init__(self, sample_dirs: List[str], transform: A.Compose):
        self.sample_dirs = sample_dirs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        img_path = os.path.join(sample_dir, 'rgb.jpg')
        mask_path = os.path.join(sample_dir, 'labels.png')
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L")) # Carica come scala di grigi (indici)
        
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']

# =============================================================================
# 4. FUNZIONI HELPER E MODELLO
# =============================================================================
def get_model(config: Dict[str, Any]) -> nn.Module:
    print(f"Creazione del modello: U-Net con encoder '{config['ENCODER']}'")
    model = smp.Unet(
        encoder_name=config["ENCODER"],
        encoder_weights=config["ENCODER_WEIGHTS"],
        in_channels=3,
        classes=config["NUM_CLASSES"],
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri addestrabili del modello: {total_params:,}")
    return model

class CombinedDiceFocalLoss(nn.Module):
    """Combina Dice Loss e Focal Loss."""
    def __init__(self, dice_weight=0.5, focal_weight=0.5, focal_gamma=2.0):
        super().__init__()
        # DiceLoss necessita dell'argomento esplicito
        self.dice = DiceLoss(mode='multiclass', from_logits=True)
        # FocalLoss assume già i logits, quindi l'argomento non serve e non esiste
        self.focal = FocalLoss(mode='multiclass', gamma=focal_gamma) # <-- MODIFICA QUI
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight

    def forward(self, pred, target):
        dice_loss_val = self.dice(pred, target)
        focal_loss_val = self.focal(pred, target)
        return self.dice_weight * dice_loss_val + self.focal_weight * focal_loss_val
    

def compute_iou(preds, labels, num_classes, ignore_index):
    iou_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        if cls == ignore_index: continue
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        iou_per_class[cls] = intersection / union if union > 0 else np.nan
    mean_iou = np.nanmean(iou_per_class)
    return mean_iou, iou_per_class

def plot_confusion_matrix(all_labels, all_preds, class_names, ignore_index):
    """Calcola e visualizza la matrice di confusione normalizzata."""
    print("Calcolo della Matrice di Confusione...")
    labels_to_include = [i for i, name in enumerate(class_names) if i != ignore_index]
    names_to_include = [name for i, name in enumerate(class_names) if i != ignore_index]

    cm = confusion_matrix(
        all_labels.flatten(), 
        all_preds.flatten(), 
        labels=labels_to_include,
        normalize='true'
    )
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=names_to_include, yticklabels=names_to_include
    )
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Matrice di Confusione (Normalizzata per Riga)', fontsize=14)
    plt.show()

# =============================================================================
# 5. CICLI DI TRAINING E VALIDAZIONE
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device, dtype=torch.long)
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

def evaluate(model, loader, criterion, device, config) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            with autocast(enabled=True, device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(masks.cpu().numpy())
            
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    avg_loss = total_loss / len(loader)
    mean_iou, iou_per_class = compute_iou(all_preds, all_labels, config["NUM_CLASSES"], config["IGNORE_INDEX"])
    return avg_loss, mean_iou, iou_per_class, all_labels, all_preds

# =============================================================================
# 6. FUNZIONE PRINCIPALE
# =============================================================================
def main():
    config = get_config()
    device = config["DEVICE"]
    use_amp = config["USE_AMP"] and device.type == 'cuda'
    print(f"Device: {device} | Mixed Precision: {use_amp}")
    print(f"Esperimento Finale: {config['ENCODER']}, dati offline, Dice+Focal Loss")

    train_dirs, val_dirs = get_grouped_stratified_split(config)
    
    transforms = get_transforms(config["OUTPUT_SIZE"])
    train_dataset = SegmentationDataset(train_dirs, transforms)
    val_dataset = SegmentationDataset(val_dirs, transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True,
        num_workers=config["NUM_WORKERS"], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False,
        num_workers=config["NUM_WORKERS"], pin_memory=True
    )

    model = get_model(config).to(device)
    criterion = CombinedDiceFocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, verbose=True, factor=0.2)
    scaler = GradScaler(enabled=use_amp)

    best_iou = -1.0
    
    print("\n--- Inizio Training ---")
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_miou, val_iou_per_class, val_labels, val_preds = evaluate(model, val_loader, criterion, device, config)
        
        scheduler.step(val_miou)
        
        print(f"Risultati Epoch {epoch+1}:")
        print(f"  - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        print("  - IoU per Classe (Validation):")
        for i, iou in enumerate(val_iou_per_class):
            if i != config["IGNORE_INDEX"]:
                print(f"    - Classe {i} ({config['CLASS_NAMES'][i]}): {iou:.4f}")
        
        if val_miou > best_iou:
            best_iou = val_miou
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<")
            plot_confusion_matrix(val_labels, val_preds, config["CLASS_NAMES"], config["IGNORE_INDEX"])

    print("\n--- Training Completato ---")
    print(f"Miglior mIoU di validazione ottenuto: {best_iou:.4f}")

# =============================================================================
# 7. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()