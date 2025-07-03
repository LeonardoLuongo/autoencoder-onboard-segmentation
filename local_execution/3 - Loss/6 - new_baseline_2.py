# SCRIPT DI TRAINING PER L'ESPERIMENTO "ATTACCO FINALE"

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import glob
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import TverskyLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =============================================================================
# 2. CONFIGURAZIONE
# =============================================================================
def get_config() -> Dict[str, Any]:
    """
    Restituisce un dizionario con la configurazione per l'esperimento "Attacco Finale".
    """
    config = {
        # --- Parametri di base ---
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "DATASET_PATH": "./dataset/training/train",
        "MODEL_SAVE_PATH": "./efficientnet-b4_tversky_model.pth", # Nuovo nome per il modello
        "RANDOM_SEED": 42,
        "NUM_WORKERS": 0,
        "USE_AMP": True,

        # --- Pilastro 1: Aumentare la Capacità del Modello ---
        "ENCODER": "efficientnet-b4",
        "ENCODER_WEIGHTS": "imagenet",

        # --- Pilastro 2: Sfruttare la Memoria (Risoluzione e Batch Size) ---
        "OUTPUT_SIZE": (384, 768),
        "BATCH_SIZE": 4, # Iniziamo con 4, come suggerito

        # --- Parametri di Training ---
        "NUM_EPOCHS": 25, # Aumentato a 25 per dare tempo al modello più grande di convergere
        "LEARNING_RATE": 1e-4,
        "WEIGHT_DECAY": 1e-4, # Aumentato per regolarizzare il modello più grande

        # --- Parametri del Dataset ---
        "NUM_CLASSES": 9,
        "IGNORE_INDEX": 0,
        "VALIDATION_SPLIT": 0.2,
    }
    return config

# =============================================================================
# 3. MAPPA COLORI E FUNZIONE DI CONVERSIONE (INVARIATA)
# =============================================================================
COLOR_TO_CLASS_ID = {
    (255, 255, 255): 0, (1, 88, 255): 1, (255, 0, 128): 2, (178, 176, 153): 3,
    (128, 255, 0): 4, (40, 80, 0): 5, (0, 160, 0): 6, (156, 76, 30): 7, (255, 0, 0): 8
}

def rgb_to_class_id_mask(mask_rgb: np.ndarray, color_map: Dict[Tuple[int, int, int], int], ignore_index: int) -> np.ndarray:
    class_mask = np.full(mask_rgb.shape[:2], fill_value=ignore_index, dtype=np.uint8)
    for color, class_id in color_map.items():
        matches = np.all(mask_rgb == color, axis=-1)
        class_mask[matches] = class_id
    return class_mask

# =============================================================================
# 4. DATASET E AUGMENTATION (INVARIATI)
# =============================================================================
def get_transforms(is_train: bool, output_size: Tuple[int, int]) -> A.Compose:
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(height=output_size[0], width=output_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(height=output_size[0], width=output_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

class SegmentationDataset(Dataset):
    def __init__(self, sample_dirs: List[str], color_map: Dict, transform: A.Compose, ignore_index: int):
        self.sample_dirs = sample_dirs
        self.color_map = color_map
        self.transform = transform
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        img_path = os.path.join(sample_dir, 'rgb.jpg')
        mask_path = os.path.join(sample_dir, 'labels.png')
        image_rgb = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        mask_class_ids = rgb_to_class_id_mask(mask_rgb, self.color_map, self.ignore_index)
        augmented = self.transform(image=image_rgb, mask=mask_class_ids)
        return augmented['image'], augmented['mask']

# =============================================================================
# 5. MODELLO E FUNZIONI HELPER (INVARIATI)
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

def compute_iou(preds, labels, num_classes, ignore_index):
    iou_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds, target_inds = (preds == cls), (labels == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        iou_per_class[cls] = intersection / union if union > 0 else np.nan
    mean_iou = np.nanmean(iou_per_class)
    return mean_iou, iou_per_class

# =============================================================================
# 6. FUNZIONI DI TRAINING E VALIDAZIONE (INVARIATE)
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

def evaluate(model, loader, criterion, device, num_classes, ignore_index):
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
    mean_iou, iou_per_class = compute_iou(all_preds, all_labels, num_classes, ignore_index)
    return avg_loss, mean_iou, iou_per_class

# =============================================================================
# 7. FUNZIONE PRINCIPALE (AGGIORNATA)
# =============================================================================
def main():
    config = get_config()
    device = config["DEVICE"]
    use_amp = config["USE_AMP"] and device.type == 'cuda'
    print(f"Device: {device} | Mixed Precision: {use_amp}")
    print(f"Esperimento: Attacco Finale con {config['ENCODER']} e TverskyLoss")

    all_sample_dirs = glob.glob(os.path.join(config["DATASET_PATH"], "*"))
    all_sample_dirs = sorted([d for d in all_sample_dirs if os.path.isdir(d)])
    
    train_dirs, val_dirs = train_test_split(
        all_sample_dirs, test_size=config["VALIDATION_SPLIT"], random_state=config["RANDOM_SEED"]
    )
    print(f"Dataset: {len(train_dirs)} train, {len(val_dirs)} val | Risoluzione: {config['OUTPUT_SIZE']}")

    train_transforms = get_transforms(is_train=True, output_size=config["OUTPUT_SIZE"])
    val_transforms = get_transforms(is_train=False, output_size=config["OUTPUT_SIZE"])

    train_dataset = SegmentationDataset(train_dirs, COLOR_TO_CLASS_ID, train_transforms, config["IGNORE_INDEX"])
    val_dataset = SegmentationDataset(val_dirs, COLOR_TO_CLASS_ID, val_transforms, config["IGNORE_INDEX"])

    train_loader = DataLoader(
        train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True,
        num_workers=config["NUM_WORKERS"], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False,
        num_workers=config["NUM_WORKERS"], pin_memory=True
    )

    model = get_model(config).to(device)

    # --- Pilastro 3: Attaccare lo sbilanciamento con una Loss Ibrida/Avanzata ---
    print("Loss function: TverskyLoss (generalizzazione di Dice Loss)")
    criterion = TverskyLoss(mode='multiclass', from_logits=True, alpha=0.5, beta=0.5, gamma=1.0)
    
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    scaler = GradScaler(enabled=use_amp)

    best_miou = -1.0
    
    print("\n--- Inizio Training 'Attacco Finale' ---")
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_miou, val_iou_per_class = evaluate(model, val_loader, criterion, device, config["NUM_CLASSES"], config["IGNORE_INDEX"])
        
        print(f"Risultati Epoch {epoch+1}:")
        print(f"  - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        print("  - IoU per Classe (Validation):")
        class_names = ["Ignored", "Sky", "Rough", "Smooth", "Grass", "HighVeg", "LowVeg", "Debris", "Obstacle"]
        for i, iou in enumerate(val_iou_per_class):
            if i != config["IGNORE_INDEX"]:
                print(f"    - Classe {i} ({class_names[i]}): {iou:.4f}")
        
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_miou:.4f} <<<\n")

    print("\n--- Training Completato ---")
    print(f"Miglior mIoU di validazione ottenuto: {best_miou:.4f}")
    print(f"Modello salvato in: {config['MODEL_SAVE_PATH']}")

# =============================================================================
# 8. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()


'''
> & "C:/Program Files/Python312/python.exe" "d:/MLPoject-Git/autoencoder-onboard-segmentation/local_execution/3 - Loss/6 - new_baseline_2.py"
Device: cuda | Mixed Precision: True
Esperimento: Attacco Finale con efficientnet-b4 e TverskyLoss
Dataset: 744 train, 187 val | Risoluzione: (384, 768)
Creazione del modello: U-Net con encoder 'efficientnet-b4'
Parametri addestrabili del modello: 20,226,849
Loss function: TverskyLoss (generalizzazione di Dice Loss)

--- Inizio Training 'Attacco Finale' ---

--- Epoch 1/25 ---
Risultati Epoch 1:
  - Train Loss: 0.6260 | Val Loss: 0.4980 | Val mIoU: 0.2818
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.8194
    - Classe 2 (Rough): 0.0000
    - Classe 3 (Smooth): 0.5201
    - Classe 4 (Grass): 0.4137
    - Classe 5 (HighVeg): 0.7162
    - Classe 6 (LowVeg): 0.0032
    - Classe 7 (Debris): 0.0630
    - Classe 8 (Obstacle): 0.0008

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2818 <<<


--- Epoch 2/25 ---
Risultati Epoch 2:                                                                                                                                                                
  - Train Loss: 0.4473 | Val Loss: 0.3771 | Val mIoU: 0.3575
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.8748
    - Classe 2 (Rough): 0.0000
    - Classe 3 (Smooth): 0.5776
    - Classe 4 (Grass): 0.5267
    - Classe 5 (HighVeg): 0.8153
    - Classe 6 (LowVeg): 0.0056
    - Classe 7 (Debris): 0.4176
    - Classe 8 (Obstacle): 0.0000

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3575 <<<


--- Epoch 3/25 ---
Risultati Epoch 3:                                                                                                                                                                
  - Train Loss: 0.3782 | Val Loss: 0.3280 | Val mIoU: 0.3958
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.8853
    - Classe 2 (Rough): 0.0000
    - Classe 3 (Smooth): 0.5402
    - Classe 4 (Grass): 0.5843
    - Classe 5 (HighVeg): 0.8227
    - Classe 6 (LowVeg): 0.2005
    - Classe 7 (Debris): 0.5289
    - Classe 8 (Obstacle): 0.0000

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3958 <<<


--- Epoch 4/25 ---
Risultati Epoch 4:                                                                                                                                                                
  - Train Loss: 0.3249 | Val Loss: 0.3030 | Val mIoU: 0.4068
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.8921
    - Classe 2 (Rough): 0.0000
    - Classe 3 (Smooth): 0.6558
    - Classe 4 (Grass): 0.5591
    - Classe 5 (HighVeg): 0.7778
    - Classe 6 (LowVeg): 0.2113
    - Classe 7 (Debris): 0.5651
    - Classe 8 (Obstacle): 0.0000

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4068 <<<


--- Epoch 5/25 ---
Risultati Epoch 5:                                                                                                                                                                
  - Train Loss: 0.2945 | Val Loss: 0.2876 | Val mIoU: 0.4210
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.8874
    - Classe 2 (Rough): 0.0000
    - Classe 3 (Smooth): 0.6690
    - Classe 4 (Grass): 0.6032
    - Classe 5 (HighVeg): 0.8268
    - Classe 6 (LowVeg): 0.2453
    - Classe 7 (Debris): 0.5574
    - Classe 8 (Obstacle): 0.0000

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4210 <<<


--- Epoch 6/25 ---
Risultati Epoch 6:                                                                                                                                                                
  - Train Loss: 0.2778 | Val Loss: 0.2893 | Val mIoU: 0.4120
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.8968
    - Classe 2 (Rough): 0.0000
    - Classe 3 (Smooth): 0.5298
    - Classe 4 (Grass): 0.6235
    - Classe 5 (HighVeg): 0.8313
    - Classe 6 (LowVeg): 0.2695
    - Classe 7 (Debris): 0.5572
    - Classe 8 (Obstacle): 0.0000

--- Epoch 7/25 ---
Risultati Epoch 7:                                                                                                                                                                
  - Train Loss: 0.2667 | Val Loss: 0.2885 | Val mIoU: 0.4116
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.8914
    - Classe 2 (Rough): 0.0000
    - Classe 3 (Smooth): 0.6231
    - Classe 4 (Grass): 0.6457
    - Classe 5 (HighVeg): 0.8160
    - Classe 6 (LowVeg): 0.2777
    - Classe 7 (Debris): 0.4505
    - Classe 8 (Obstacle): 0.0000

--- Epoch 8/25 ---
Risultati Epoch 8:                                                                                                                                                                
  - Train Loss: 0.2643 | Val Loss: 0.2741 | Val mIoU: 0.4262
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.8992
    - Classe 2 (Rough): 0.0000
    - Classe 3 (Smooth): 0.5968
    - Classe 4 (Grass): 0.6361
    - Classe 5 (HighVeg): 0.8380
    - Classe 6 (LowVeg): 0.2781
    - Classe 7 (Debris): 0.5860
    - Classe 8 (Obstacle): 0.0018

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4262 <<<


--- Epoch 9/25 ---
Risultati Epoch 9:                                                                                                                                                                
  - Train Loss: 0.2506 | Val Loss: 0.2790 | Val mIoU: 0.4306
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.8984
    - Classe 2 (Rough): 0.0000
    - Classe 3 (Smooth): 0.6381
    - Classe 4 (Grass): 0.5860
    - Classe 5 (HighVeg): 0.8246
    - Classe 6 (LowVeg): 0.2339
    - Classe 7 (Debris): 0.5502
    - Classe 8 (Obstacle): 0.1445

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4306 <<<


--- Epoch 10/25 ---
Risultati Epoch 10:                                                                                                                                                               
  - Train Loss: 0.2291 | Val Loss: 0.2648 | Val mIoU: 0.4477
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.9017
    - Classe 2 (Rough): 0.0000
    - Classe 3 (Smooth): 0.6307
    - Classe 4 (Grass): 0.6066
    - Classe 5 (HighVeg): 0.8294
    - Classe 6 (LowVeg): 0.2508
    - Classe 7 (Debris): 0.5619
    - Classe 8 (Obstacle): 0.2479

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4477 <<<



'''