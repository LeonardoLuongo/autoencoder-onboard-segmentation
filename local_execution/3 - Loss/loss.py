# -*- coding: utf-8 -*-
"""
Script per l'addestramento di un modello di segmentazione semantica (U-Net)
con le seguenti caratteristiche:
- Data augmentation tramite Albumentations.
- Loss pesata (Weighted Cross-Entropy) per gestire lo sbilanciamento delle classi.
- Split stratificato del dataset basato sulla presenza di classi rare.
- Ottimizzato per alte performance tramite Automatic Mixed Precision (AMP).
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import sys
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

from segmentation_models_pytorch.losses import DiceLoss

# =============================================================================
# 2. CONFIGURAZIONE
# =============================================================================
def get_config() -> Dict[str, Any]:
    """Restituisce un dizionario con tutte le configurazioni dello script."""
    IS_COLAB = 'google.colab' in sys.modules
    base_path = "./"
    if IS_COLAB:
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = "/content/drive/MyDrive/ML_Project/"
    else:
        # Assicurati che il percorso relativo sia corretto
        base_path = "./" 

    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        
        # Percorsi
        "DATASET_PATH": os.path.join(base_path, "dataset/training/train"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_model_albumentations.pth"),

        # Iperparametri di addestramento
        "NUM_EPOCHS": 15,
        "BATCH_SIZE": 8,
        "LEARNING_RATE": 1e-4,

        # Parametri del dataset, split e augmentation
        "OUTPUT_SIZE": (544, 1024), # (Altezza, Larghezza) -> Cruciale per la velocità
        "NUM_CLASSES": 9,
        "VALIDATION_SPLIT": 0.2,
        "RANDOM_SEED": 42,
        "CLASSI_RARE_INDICI": [4, 5, 6, 8],

        # Parametri di performance
        "USE_AMP": True, # Abilita Automatic Mixed Precision
        "NUM_WORKERS_TRAIN": 2, # Aumenta se la CPU lo permette, 0 su Windows se dà problemi
        "NUM_WORKERS_VAL": 1,

        # Parametri del modello
        "ENCODER": "resnet34",
        "ENCODER_WEIGHTS": "imagenet",
    }
    return config

# =============================================================================
# 3. DATA AUGMENTATION
# =============================================================================
def get_train_transforms(output_size: Tuple[int, int]) -> A.Compose:
    """Restituisce la pipeline di augmentation per il training set."""
    return A.Compose([
        A.Resize(height=output_size[0], width=output_size[1]),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

def get_val_transforms(output_size: Tuple[int, int]) -> A.Compose:
    """Restituisce la pipeline di trasformazioni per il validation set."""
    return A.Compose([
        A.Resize(height=output_size[0], width=output_size[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

# =============================================================================
# 4. CLASSE DATASET
# =============================================================================
class SegmentationDataset(Dataset):
    """Dataset per la segmentazione che utilizza Albumentations."""
    def __init__(self, sample_dirs: List[str], transform: A.Compose):
        self.sample_dirs = sample_dirs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        img_path = os.path.join(sample_dir, 'rgb.jpg')
        mask_path = os.path.join(sample_dir, 'labels.png')

        # Carica come array NumPy, richiesto da Albumentations
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = augmented['mask']
        
        # ToTensorV2 si occupa della conversione, ma assicuriamoci dei tipi
        return image.float(), mask.long()

# =============================================================================
# 5. FUNZIONI HELPER
# =============================================================================
def analyze_and_split_dataset(
    config: Dict[str, Any]
) -> Tuple[List[str], List[str], np.ndarray]:
    """Analizza il dataset, calcola la distribuzione dei pixel e esegue lo split stratificato."""
    dataset_path = config["DATASET_PATH"]
    print("Avvio analisi del dataset e preparazione dello split...")
    
    all_dirs = sorted([
        os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
        if os.path.isdir(os.path.join(dataset_path, d)) and d.isdigit()
    ])
    if not all_dirs:
        raise FileNotFoundError(f"Nessun campione valido trovato in '{dataset_path}'")
    
    print(f"Trovati {len(all_dirs)} campioni.")
    pixel_counts = np.zeros(config["NUM_CLASSES"], dtype=np.int64)
    stratify_metadata, valid_sample_dirs = [], []

    for sample_dir in tqdm(all_dirs, desc="Analisi campioni"):
        mask_path = os.path.join(sample_dir, 'labels.png')
        if not os.path.exists(mask_path):
            continue
        try:
            mask_np = np.array(Image.open(mask_path), dtype=np.uint8)
            pixel_counts += np.bincount(mask_np.flatten(), minlength=config["NUM_CLASSES"])
            
            classes_present = np.unique(mask_np)
            stratify_vector = np.zeros(config["NUM_CLASSES"], dtype=np.int8)
            stratify_vector[classes_present] = 1
            stratify_metadata.append(stratify_vector)
            valid_sample_dirs.append(sample_dir)
        except Exception as e:
            print(f"Errore nell'elaborazione di {sample_dir}: {e}. Campione saltato.")

    # Strategia di stratificazione semplificata
    stratify_labels = [
        1 if any(vec[i] == 1 for i in config["CLASSI_RARE_INDICI"]) else 0 
        for vec in stratify_metadata
    ]
    print(f"Stratificazione basata su {sum(stratify_labels)} campioni con classi rare.")

    try:
        train_dirs, val_dirs = train_test_split(
            valid_sample_dirs, test_size=config["VALIDATION_SPLIT"],
            random_state=config["RANDOM_SEED"], stratify=stratify_labels
        )
        print("Split stratificato eseguito con successo.")
    except ValueError:
        print("ERRORE: Split stratificato fallito. Eseguo uno split casuale.")
        train_dirs, val_dirs = train_test_split(
            valid_sample_dirs, test_size=config["VALIDATION_SPLIT"],
            random_state=config["RANDOM_SEED"], shuffle=True
        )

    print(f"Split creato: {len(train_dirs)} train, {len(val_dirs)} validation.")
    return train_dirs, val_dirs, pixel_counts

def calculate_loss_weights(pixel_counts: np.ndarray, device: torch.device) -> torch.Tensor:
    """Calcola i pesi per la CrossEntropyLoss basandosi sulla frequenza inversa dei pixel."""
    total_pixels = pixel_counts.sum()
    class_frequencies = pixel_counts / total_pixels
    weights = 1.0 / (class_frequencies + 1e-6)
    weights = weights / weights.mean()
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    
    print("Pesi calcolati per la loss (normalizzati):")
    for i, w in enumerate(weights_tensor):
        print(f"  - Classe {i}: {w.item():.2f}")
    return weights_tensor

def calculate_iou_per_class(preds_np: np.ndarray, labels_np: np.ndarray, num_classes: int) -> np.ndarray:
    """Calcola l'IoU per ogni classe su array NumPy."""
    iou_scores = np.zeros(num_classes)
    for cls in range(num_classes):
        pred_inds = (preds_np == cls)
        target_inds = (labels_np == cls)
        
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        
        if union == 0:
            iou_scores[cls] = 1.0
        else:
            iou_scores[cls] = intersection / union
            
    return iou_scores

# =============================================================================
# 6. FUNZIONI DI TRAINING E VALIDAZIONE
# =============================================================================
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                    criterion_ce: nn.Module, criterion_dice:nn.Module, scaler: GradScaler, device: torch.device, use_amp: bool) -> float:
    """Esegue un'epoca di training con AMP e restituisce la loss media."""
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(loader, desc="Training")
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(enabled=use_amp):
            outputs = model(images)
            loss = 0.5 * criterion_ce(outputs, masks) + 0.5 * criterion_dice(outputs, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    return running_loss / len(loader)

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
             device: torch.device, num_classes: int) -> Tuple[float, float, np.ndarray]:
    """Valuta il modello in modo ottimizzato, restituendo loss, mIoU e IoU per classe."""
    model.eval()
    total_loss = 0.0
    total_iou = np.zeros(num_classes)
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validation")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            
            # Calcola IoU per ogni elemento nel batch, spostando solo i dati necessari
            for i in range(images.size(0)):
                total_iou += calculate_iou_per_class(
                    preds[i].cpu().numpy(), 
                    masks[i].cpu().numpy(), 
                    num_classes
                )

    avg_loss = total_loss / len(loader)
    iou_per_class = total_iou / len(loader.dataset)
    mean_iou = iou_per_class.mean()
    
    return avg_loss, mean_iou, iou_per_class

# =============================================================================
# 7. FUNZIONE PRINCIPALE
# =============================================================================
def main():
    """Funzione principale che orchestra il processo di addestramento."""
    
    config = get_config()
    device = config["DEVICE"]
    use_amp = config["USE_AMP"] and device.type == 'cuda'
    print(f"Device in uso: {device} | Automatic Mixed Precision: {'Abilitata' if use_amp else 'Disabilitata'}")

    train_dirs, val_dirs, pixel_counts = analyze_and_split_dataset(config)
    
    train_transforms = get_train_transforms(config["OUTPUT_SIZE"])
    val_transforms = get_val_transforms(config["OUTPUT_SIZE"])

    train_dataset = SegmentationDataset(train_dirs, train_transforms)
    val_dataset = SegmentationDataset(val_dirs, val_transforms)
    
    train_loader = DataLoader(
        train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True,
        num_workers=config["NUM_WORKERS_TRAIN"], pin_memory=True,
        persistent_workers=(config["NUM_WORKERS_TRAIN"] > 0)
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False,
        num_workers=config["NUM_WORKERS_VAL"], pin_memory=True
    )

    model = smp.Unet(
        encoder_name=config["ENCODER"], encoder_weights=config["ENCODER_WEIGHTS"],
        in_channels=3, classes=config["NUM_CLASSES"]
    ).to(device)
    
    
    print(f"Modello smp.Unet con encoder '{config['ENCODER']}' istanziato.")
    print(f"Parametri addestrabili: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    class_weights = calculate_loss_weights(pixel_counts, device)
    # Mantieni la tua criterion CE pesata
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights)
    # Crea una DiceLoss
    criterion_dice = DiceLoss(mode='multiclass')
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      # Riduci quando la metrica smette di AUMENTARE
        factor=0.1,      # Riduci LR del 90% (1 * 0.1)
        patience=3,      # Aspetta 3 epoche senza miglioramenti prima di ridurre
        verbose=True     # Stampa un messaggio quando riduce il LR
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_val_miou = -1.0
    print("\n--- Inizio Addestramento Ottimizzato ---")
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion_ce, criterion_dice, scaler, device, use_amp)
        val_loss, val_miou, val_iou_per_class = evaluate(model, val_loader, criterion_ce, device, criterion_dice, config["NUM_CLASSES"])
        scheduler.step(val_miou)  # Riduci LR se mIoU non migliora
        
        print(f"Risultati Epoch {epoch+1}:")
        print(f"  - Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation mIoU: {val_miou:.4f}")
        
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"\n>>> Nuovo modello migliore salvato in '{config['MODEL_SAVE_PATH']}' con mIoU: {best_val_miou:.4f} <<<\n")

    print(f"\n--- Addestramento Completato ---")
    print(f"Miglior mIoU di validazione: {best_val_miou:.4f}")

# =============================================================================
# 8. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()

