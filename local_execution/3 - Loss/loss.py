# -*- coding: utf-8 -*-
"""
Script per l'addestramento di un modello di segmentazione semantica (U-Net)
utilizzando un approccio con loss pesata (Weighted Cross-Entropy) per gestire
lo sbilanciamento delle classi.

OTTIMIZZAZIONI INCLUSE:
1. Ridimensionamento delle immagini per evitare colli di bottiglia I/O e di calcolo.
2. Utilizzo di Automatic Mixed Precision (AMP) per un training più veloce su GPU moderne.
3. Calcolo delle metriche di validazione ottimizzato.
4. Predisposizione per torch.compile (PyTorch 2.x).
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
# --- MODIFICA ---: Import aggiuntivi per trasformazioni e AMP
from torchvision import transforms
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp

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

    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        
        "DATASET_PATH": os.path.join(base_path, "dataset/training/train"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_model_optimized.pth"),

        "NUM_EPOCHS": 10,
        "BATCH_SIZE": 8,
        "LEARNING_RATE": 1e-4,

        # --- MODIFICA ---: Aggiunti parametri per il ridimensionamento e AMP
        "OUTPUT_SIZE": (512, 1024), # (Altezza, Larghezza) -> Cruciale per la velocità!
        "USE_AMP": True, # Abilita Automatic Mixed Precision

        "NUM_CLASSES": 9,
        "VALIDATION_SPLIT": 0.2,
        "RANDOM_SEED": 42,
        "NUM_WORKERS_TRAIN": 2, 
        "NUM_WORKERS_VAL": 1,

        "ENCODER": "resnet34",
        "ENCODER_WEIGHTS": "imagenet",
        
        "CLASSI_RARE_INDICI": [4, 5, 6, 8],
    }
    return config

# =============================================================================
# 3. CLASSE DATASET (MODIFICATA)
# =============================================================================
class SegmentationDataset(Dataset):
    """
    Dataset ottimizzato: carica, ridimensiona e converte immagini e maschere.
    """
    def __init__(self, sample_dirs: List[str], output_size: Tuple[int, int]):
        """
        Args:
            sample_dirs (List[str]): Lista dei percorsi alle cartelle dei campioni.
            output_size (Tuple[int, int]): Dimensione di output (H, W) per immagini e maschere.
        """
        self.sample_dirs = sample_dirs
        self.output_size = output_size
        # --- MODIFICA ---: Usiamo TF.to_tensor che normalizza a [0, 1] e permuta le dimensioni
        # La normalizzazione con media/std può essere aggiunta qui se necessario
        self.image_transform = transforms.ToTensor()

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]

        img_path = os.path.join(sample_dir, 'rgb.jpg')
        mask_path = os.path.join(sample_dir, 'labels.png')

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # --- MODIFICA ---: Ridimensionamento, la parte più importante per la performance
        image = TF.resize(image, self.output_size, interpolation=TF.InterpolationMode.BILINEAR)
        # Per le maschere, usare NEAREST per non creare valori di classe intermedi
        mask = TF.resize(mask, self.output_size, interpolation=TF.InterpolationMode.NEAREST)

        # Converte in tensori
        image_tensor = self.image_transform(image)
        mask_tensor = torch.from_numpy(np.array(mask, dtype=np.int64))
        
        return image_tensor, mask_tensor

# =============================================================================
# 4. FUNZIONI HELPER (con piccole modifiche)
# =============================================================================
# La funzione analyze_and_split_dataset rimane quasi invariata

def analyze_and_split_dataset(
    dataset_path: str, 
    val_size: float, 
    random_state: int,
    rare_class_indices: List[int]
) -> Tuple[List[str], List[str], np.ndarray]:
    """Analizza il dataset, calcola la distribuzione dei pixel e esegue uno split stratificato."""
    # (Codice identico a prima, lo ometto per brevità ma va incluso qui)
    print("Avvio analisi del dataset e preparazione dello split...")
    all_dirs = sorted([
        os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
        if os.path.isdir(os.path.join(dataset_path, d)) and d.isdigit()
    ])
    if not all_dirs: raise FileNotFoundError(f"Nessun campione valido trovato in '{dataset_path}'")
    print(f"Trovati {len(all_dirs)} campioni.")
    num_classes = get_config()["NUM_CLASSES"]
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    stratify_metadata, valid_sample_dirs = [], []
    for sample_dir in tqdm(all_dirs, desc="Analisi campioni"):
        mask_path = os.path.join(sample_dir, 'labels.png')
        if not os.path.exists(mask_path): continue
        try:
            mask_np = np.array(Image.open(mask_path), dtype=np.uint8)
            pixel_counts += np.bincount(mask_np.flatten(), minlength=num_classes)
            classes_present = np.unique(mask_np)
            stratify_vector = np.zeros(num_classes, dtype=np.int8)
            stratify_vector[classes_present] = 1
            stratify_metadata.append(stratify_vector)
            valid_sample_dirs.append(sample_dir)
        except Exception as e:
            print(f"Errore in {sample_dir}: {e}. Campione saltato.")
    stratify_labels = [1 if any(vec[i] == 1 for i in rare_class_indices) else 0 for vec in stratify_metadata]
    print(f"Stratificazione basata su {sum(stratify_labels)} campioni con classi rare.")
    try:
        train_dirs, val_dirs = train_test_split(valid_sample_dirs, test_size=val_size, random_state=random_state, stratify=stratify_labels)
        print("Split stratificato eseguito.")
    except ValueError:
        print("ERRORE: Split stratificato fallito. Eseguo uno split casuale.")
        train_dirs, val_dirs = train_test_split(valid_sample_dirs, test_size=val_size, random_state=random_state, shuffle=True)
    print(f"Split creato: {len(train_dirs)} train, {len(val_dirs)} validation.")
    return train_dirs, val_dirs, pixel_counts

def calculate_loss_weights(pixel_counts: np.ndarray, device: torch.device) -> torch.Tensor:
    """Calcola i pesi per la CrossEntropyLoss."""
    # (Codice identico a prima)
    total_pixels = pixel_counts.sum()
    class_frequencies = pixel_counts / total_pixels
    weights = 1.0 / (class_frequencies + 1e-6)
    weights = weights / weights.mean()
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(device)
    print("Pesi calcolati per la loss (normalizzati):")
    for i, w in enumerate(weights_tensor): print(f"  - Classe {i}: {w.item():.2f}")
    return weights_tensor

# --- MODIFICA ---: La funzione ora riceve predizioni (indici) invece di logits
def calculate_iou_per_class(preds_np: np.ndarray, labels_np: np.ndarray, num_classes: int) -> np.ndarray:
    """Calcola l'IoU per ogni classe su array NumPy."""
    iou_scores = np.zeros(num_classes)
    for cls in range(num_classes):
        pred_inds = (preds_np == cls)
        target_inds = (labels_np == cls)
        
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        
        if union == 0:
            iou_scores[cls] = 1.0  # La classe non è presente nel target né nella predizione
        else:
            iou_scores[cls] = intersection / union
            
    return iou_scores

# =============================================================================
# 5. FUNZIONI DI TRAINING E VALIDAZIONE (MODIFICATE)
# =============================================================================
def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: optim.Optimizer,
                    criterion: nn.Module, scaler: GradScaler, device: torch.device, use_amp: bool) -> float:
    """Esegue un'epoca di training con AMP e restituisce la loss media."""
    model.train()
    running_loss = 0.0
    
    progress_bar = tqdm(loader, desc="Training")
    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # --- MODIFICA ---: Usa autocast per mixed precision
        with autocast(enabled=use_amp):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        # --- MODIFICA ---: Scaler per il backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * images.size(0)
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    return running_loss / len(loader.dataset)

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
             device: torch.device, num_classes: int) -> Tuple[float, float, np.ndarray]:
    """Valuta il modello in modo ottimizzato."""
    model.eval()
    running_loss = 0.0
    total_iou = np.zeros(num_classes)
    num_samples = 0
    
    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Validation")
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            running_loss += loss.item() * images.size(0)
            
            # --- MODIFICA ---: Argmax sulla GPU, poi sposta solo gli indici su CPU
            preds = torch.argmax(outputs, dim=1)
            
            # Calcola IoU per ogni elemento nel batch
            for i in range(images.size(0)):
                total_iou += calculate_iou_per_class(
                    preds[i].cpu().numpy(), 
                    masks[i].cpu().numpy(), 
                    num_classes
                )
            num_samples += images.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    # Media per campione, non per batch
    iou_per_class = total_iou / num_samples
    mean_iou = iou_per_class.mean()
    
    return epoch_loss, mean_iou, iou_per_class

# =============================================================================
# 6. FUNZIONE PRINCIPALE (MODIFICATA)
# =============================================================================
def main():
    """Funzione principale che orchestra il processo di addestramento."""
    config = get_config()
    device = config["DEVICE"]
    use_amp = config["USE_AMP"] and device.type == 'cuda'
    print(f"Device in uso: {device} | Automatic Mixed Precision: {'Abilitata' if use_amp else 'Disabilitata'}")

    train_dirs, val_dirs, pixel_counts = analyze_and_split_dataset(
        config["DATASET_PATH"], config["VALIDATION_SPLIT"],
        config["RANDOM_SEED"], config["CLASSI_RARE_INDICI"]
    )
    
    # --- MODIFICA ---: Passa la dimensione di output al Dataset
    train_dataset = SegmentationDataset(sample_dirs=train_dirs, output_size=config["OUTPUT_SIZE"])
    val_dataset = SegmentationDataset(sample_dirs=val_dirs, output_size=config["OUTPUT_SIZE"])
    
    train_loader = DataLoader(
        train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True,
        num_workers=config["NUM_WORKERS_TRAIN"], pin_memory=True, persistent_workers=True if config["NUM_WORKERS_TRAIN"] > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False,
        num_workers=config["NUM_WORKERS_VAL"], pin_memory=True
    )

    model = smp.Unet(
        encoder_name=config["ENCODER"], encoder_weights=config["ENCODER_WEIGHTS"],
        in_channels=3, classes=config["NUM_CLASSES"]
    ).to(device)
    
    # --- MODIFICA (Opzionale) ---: Per PyTorch 2.x, decommentare per un ulteriore speed-up
    # try:
    #     model = torch.compile(model)
    #     print("Modello compilato con torch.compile() per performance migliori.")
    # except Exception:
    #     print("torch.compile() non disponibile. Si procede senza.")

    class_weights = calculate_loss_weights(pixel_counts, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    
    # --- MODIFICA ---: Inizializza il GradScaler per AMP
    scaler = GradScaler(enabled=use_amp)

    best_val_miou = -1.0
    print("\n--- Inizio Addestramento Ottimizzato ---")
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_miou, val_iou_per_class = evaluate(model, val_loader, criterion, device, config["NUM_CLASSES"])
        
        print(f"Risultati Epoch {epoch+1}:")
        print(f"  - Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation mIoU: {val_miou:.4f}")
        
        if val_miou > best_val_miou:
            best_val_miou = val_miou
            # Se il modello è compilato, salva lo stato del modello originale
            model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
            torch.save(model_to_save.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"\n>>> Nuovo modello migliore salvato in '{config['MODEL_SAVE_PATH']}' con mIoU: {best_val_miou:.4f} <<<\n")

    print(f"\n--- Addestramento Completato ---")
    print(f"Miglior mIoU di validazione: {best_val_miou:.4f}. Modello salvato in: {config['MODEL_SAVE_PATH']}")

# =============================================================================
# 7. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()