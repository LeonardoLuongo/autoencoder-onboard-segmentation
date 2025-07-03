# SCRIPT DI TRAINING PER LA BASELINE DEFINITIVA

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
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# =============================================================================
# 2. CONFIGURAZIONE
# =============================================================================
config = {
    "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "DATASET_PATH": "./dataset/training/train",
    "MODEL_SAVE_PATH": "./baseline_model.pth",
    "NUM_EPOCHS": 20,
    "BATCH_SIZE": 4,
    "LEARNING_RATE": 1e-4,
    "NUM_CLASSES": 9,
    "IGNORE_INDEX": 0,
    "OUTPUT_SIZE": (256, 512),
    "VALIDATION_SPLIT": 0.2,
    "RANDOM_SEED": 42,
    "NUM_WORKERS": 0, # Impostato a 0 per massima compatibilità, può essere aumentato
    "ENCODER": "resnet34",
    "ENCODER_WEIGHTS": "imagenet",
    "USE_AMP": True, # Abilita la Mixed Precision per un training più veloce
}

# =============================================================================
# 3. MAPPA COLORI E FUNZIONE DI CONVERSIONE
# =============================================================================

# La "Stele di Rosetta": mappa i valori RGB della maschera agli ID di classe interi.
COLOR_TO_CLASS_ID = {
    (255, 255, 255): 0, # Bianco -> background/veicolo (da ignorare)
    (1, 88, 255): 1,   # Blu -> sky
    (255, 0, 128): 2,   # Magenta -> rough trail
    (178, 176, 153): 3, # Grigio -> smooth trail
    (128, 255, 0): 4,   # Verde Lime -> traversable grass
    (40, 80, 0): 5,     # Verde Scuro -> high vegetation
    (0, 160, 0): 6,     # Verde -> low vegetation
    (156, 76, 30): 7,   # Marrone -> "puddle"/terreno con detriti
    (255, 0, 0): 8      # Rosso -> obstacle
}

def rgb_to_class_id_mask(mask_rgb: np.ndarray, color_map: Dict[Tuple[int, int, int], int]) -> np.ndarray:
    """
    Converte una maschera RGB (H, W, 3) in una maschera di ID di classe (H, W).
    """
    # Crea una maschera vuota (inizializzata con un valore di default come l'ignore_index)
    class_mask = np.full(mask_rgb.shape[:2], fill_value=config["IGNORE_INDEX"], dtype=np.uint8)
    
    # Itera su ogni colore e ID di classe nella mappa
    for color, class_id in color_map.items():
        # Crea una maschera booleana per i pixel che corrispondono al colore corrente
        matches = np.all(mask_rgb == color, axis=-1)
        # Assegna l'ID della classe a quei pixel
        class_mask[matches] = class_id
        
    return class_mask

# =============================================================================
# 4. DATASET E AUGMENTATION (ONLINE)
# =============================================================================

def get_transforms(is_train: bool, output_size: Tuple[int, int]) -> A.Compose:
    """
    Restituisce la pipeline di trasformazioni di Albumentations.
    - Training: include augmentation leggera (flip) e preprocessing.
    - Validazione: solo preprocessing.
    """
    # Normalizzazione standard per encoder pre-addestrati su ImageNet
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if is_train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Resize(height=output_size[0], width=output_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else: # Per validazione e test
        return A.Compose([
            A.Resize(height=output_size[0], width=output_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

class SegmentationDataset(Dataset):
    """
    Dataset che carica immagini e maschere, e converte le maschere da RGB a ID di classe
    prima di applicare le trasformazioni.
    """
    def __init__(self, sample_dirs: List[str], color_map: Dict, transform: A.Compose):
        self.sample_dirs = sample_dirs
        self.color_map = color_map
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        img_path = os.path.join(sample_dir, 'rgb.jpg')
        mask_path = os.path.join(sample_dir, 'labels.png') # Il nome del file è ancora labels.png
        
        # Carica l'immagine e la maschera come array NumPy
        image_rgb = np.array(Image.open(img_path).convert("RGB"))
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        
        # --- PASSO CRUCIALE: conversione da RGB a ID di classe ---
        mask_class_ids = rgb_to_class_id_mask(mask_rgb, self.color_map)
        
        # Applica le trasformazioni (augmentation + preprocessing)
        augmented = self.transform(image=image_rgb, mask=mask_class_ids)
        
        return augmented['image'], augmented['mask']

# =============================================================================
# 5. MODELLO E FUNZIONI HELPER
# =============================================================================

def get_model(config: Dict[str, Any]) -> nn.Module:
    """Crea e restituisce il modello di segmentazione."""
    print(f"Creazione del modello: U-Net con encoder '{config['ENCODER']}'")
    model = smp.Unet(
        encoder_name=config["ENCODER"],
        encoder_weights=config["ENCODER_WEIGHTS"],
        in_channels=3,
        classes=config["NUM_CLASSES"],
    )
    return model

def compute_iou(preds, labels, num_classes, ignore_index):
    """Calcola l'IoU per classe e il mIoU medio."""
    iou_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        
        if union == 0:
            iou_per_class[cls] = np.nan # Classe non presente nel ground truth
        else:
            iou_per_class[cls] = intersection / union
            
    # mIoU è la media degli IoU validi (ignora le classi non presenti)
    mean_iou = np.nanmean(iou_per_class)
    return mean_iou, iou_per_class

# =============================================================================
# 6. FUNZIONI DI TRAINING E VALIDAZIONE
# =============================================================================

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
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

def evaluate(model, loader, criterion, device, num_classes, ignore_index):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione", leave=False):
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)
            
            with autocast(enabled=True, device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.append(preds.cpu().numpy())
            all_labels.append(masks.cpu().numpy())
    
    # Concatena i risultati di tutti i batch
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    
    avg_loss = total_loss / len(loader)
    mean_iou, iou_per_class = compute_iou(all_preds, all_labels, num_classes, ignore_index)
    
    return avg_loss, mean_iou, iou_per_class

# =============================================================================
# 7. FUNZIONE PRINCIPALE
# =============================================================================
def main():
    """Orchestra il processo di training della baseline."""
    
    # 1. Setup
    device = config["DEVICE"]
    use_amp = config["USE_AMP"] and device.type == 'cuda'
    print(f"Device: {device} | Mixed Precision: {use_amp}")

    # 2. Prepara i dati
    all_sample_dirs = glob.glob(os.path.join(config["DATASET_PATH"], "*"))
    all_sample_dirs = [d for d in all_sample_dirs if os.path.isdir(d)]
    
    train_dirs, val_dirs = train_test_split(
        all_sample_dirs, 
        test_size=config["VALIDATION_SPLIT"], 
        random_state=config["RANDOM_SEED"]
    )
    print(f"Dataset diviso in {len(train_dirs)} campioni di training e {len(val_dirs)} di validazione.")

    # 3. Crea transform, dataset e dataloader
    train_transforms = get_transforms(is_train=True, output_size=config["OUTPUT_SIZE"])
    val_transforms = get_transforms(is_train=False, output_size=config["OUTPUT_SIZE"])

    train_dataset = SegmentationDataset(train_dirs, COLOR_TO_CLASS_ID, train_transforms)
    val_dataset = SegmentationDataset(val_dirs, COLOR_TO_CLASS_ID, val_transforms)

    train_loader = DataLoader(
        train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, 
        num_workers=config["NUM_WORKERS"], pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False,
        num_workers=config["NUM_WORKERS"], pin_memory=True
    )

    # 4. Prepara Modello, Loss, Ottimizzatore
    model = get_model(config).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=config["IGNORE_INDEX"]) # Loss semplice
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    scaler = GradScaler(enabled=use_amp)

    # 5. Ciclo di Training e Validazione
    best_miou = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
    
    print("\n--- Inizio Training Baseline ---")
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_miou, val_iou_per_class = evaluate(model, val_loader, criterion, device, config["NUM_CLASSES"], config["IGNORE_INDEX"])
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        
        print(f"Risultati Epoch {epoch+1}:")
        print(f"  - Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation mIoU: {val_miou:.4f}")
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