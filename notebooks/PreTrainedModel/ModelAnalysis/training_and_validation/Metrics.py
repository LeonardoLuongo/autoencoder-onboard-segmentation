# -*- coding: utf-8 -*-
"""
Script completo per l'addestramento e la valutazione di un modello di
segmentazione U-Net. Include:
- Training e validazione con salvataggio del modello migliore.
- Tracciamento e plotting delle curve di apprendimento (Loss vs mIoU).
- Visualizzazione qualitativa delle predizioni del modello.
- Confronto quantitativo con altre run del modello.
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.transforms.functional as TF
from torchvision import transforms

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
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
        "MODEL_CHECKPOINT": os.path.join(base_path, "PreTrainedModel.pth"), # Checkpoint da cui partire

        # Iperparametri di addestramento
        "NUM_EPOCHS": 20,
        "BATCH_SIZE": 4,
        "LEARNING_RATE": 1e-5,
        "NUM_CLASSES": 9,
        "IGNORE_INDEX": 0,

        # Parametri del dataset e visualizzazione
        "OUTPUT_SIZE": (512, 1024), # Altezza, Larghezza
        "DATASET_MEAN": np.array([0.4278, 0.4464, 0.4231]),
        "DATASET_STD": np.array([0.2543, 0.2535, 0.2567]),
        "CLASS_ID_TO_COLOR": {
            0: (0, 0, 0),       # Background
            1: (70, 70, 70),    # Buildings
            2: (190, 153, 153), # Fences
            3: (250, 170, 160), # Other
            4: (220, 20, 60),   # Pedestrians
            5: (153, 153, 153), # Poles
            6: (157, 234, 50),  # RoadLines
            7: (128, 64, 128),  # Roads
            8: (244, 35, 232),  # Sidewalks
        },

        # Parametri del modello
        "ENCODER": "resnet34",
        "ENCODER_WEIGHTS": "imagenet",
    }
    return config

# =============================================================================
# 3. DATA HANDLING
# =============================================================================
class SegmentationDataset(Dataset):
    """Dataset personalizzato per la segmentazione semantica."""
    def __init__(self, sample_dirs: List[str], output_size: Tuple[int, int], 
                 mean: np.ndarray, std: np.ndarray, is_train: bool = True):
        self.sample_dirs = sample_dirs
        self.output_size = output_size
        self.is_train = is_train
        self.normalize = transforms.Normalize(mean=mean.tolist(), std=std.tolist())

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        image = Image.open(os.path.join(sample_dir, 'rgb.jpg')).convert("RGB")
        mask = Image.open(os.path.join(sample_dir, 'labels.png'))
        
        image = TF.resize(image, self.output_size, TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.output_size, TF.InterpolationMode.NEAREST)
        
        if self.is_train and torch.rand(1) < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        image = self.normalize(TF.to_tensor(image))
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask

def create_splits(data_dir: str, val_percent: float, seed: int) -> Tuple[List[str], List[str]]:
    """Divide i percorsi dei campioni in set di training e validazione."""
    all_sample_dirs = sorted([d for d in glob.glob(os.path.join(data_dir, "*")) if os.path.isdir(d)])
    n_val = int(len(all_sample_dirs) * val_percent)
    n_train = len(all_sample_dirs) - n_val
    train_dirs, val_dirs = random_split(all_sample_dirs, [n_train, n_val], torch.Generator().manual_seed(seed))
    print(f"Split creato: {len(train_dirs)} train, {len(val_dirs)} val.")
    return list(train_dirs), list(val_dirs)

# =============================================================================
# 4. VISUALIZATION & PLOTTING
# =============================================================================
def mask_to_rgb(mask: torch.Tensor, color_map: Dict[int, Tuple[int, int, int]]) -> np.ndarray:
    """Converte una maschera di class-id in un'immagine RGB."""
    rgb_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for class_id, color in color_map.items():
        rgb_mask[mask == class_id] = color
    return rgb_mask

def denormalize(tensor: torch.Tensor, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """De-normalizza un tensore immagine per la visualizzazione."""
    img_np = tensor.cpu().permute(1, 2, 0).numpy()
    img_np = (img_np * std) + mean
    return np.clip(img_np, 0, 1)

def visualize_predictions(model: nn.Module, loader: DataLoader, config: Dict, num_samples: int = 5):
    """Mostra un confronto tra immagine originale, ground truth e predizione."""
    model.eval()
    images, masks = next(iter(loader))
    images = images.to(config["DEVICE"])
    
    with torch.no_grad():
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1).cpu()

    print("\n--- Visualizzazione delle Predizioni ---")
    for i in range(min(num_samples, len(images))):
        plt.figure(figsize=(20, 6))
        
        plt.subplot(1, 3, 1)
        plt.imshow(denormalize(images[i], config["DATASET_MEAN"], config["DATASET_STD"]))
        plt.title("Input Image")
        plt.axis('off')
        
        plt.subplot(1, 3, 2)
        plt.imshow(mask_to_rgb(masks[i], config["CLASS_ID_TO_COLOR"]))
        plt.title("Ground Truth Mask")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(mask_to_rgb(preds[i], config["CLASS_ID_TO_COLOR"]))
        plt.title("Model Prediction")
        plt.axis('off')
        
        plt.show()

def plot_learning_curves(history: Dict[str, List[float]]):
    """Plotta le curve di training loss e validation mIoU."""
    print("\n--- Curve di Apprendimento ---")
    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.set_xlabel('Epoche')
    ax1.set_ylabel('Training Loss', color='tab:blue')
    ax1.plot(history['train_loss'], color='tab:blue', marker='o', label='Training Loss')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Validation mIoU', color='tab:red')
    ax2.plot(history['val_iou'], color='tab:red', marker='x', label='Validation mIoU')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.tight_layout()
    plt.title('Curve di Apprendimento del Modello')
    plt.grid(True, linestyle='--')
    plt.show()

def plot_model_comparison(results: Dict[str, float]):
    """Crea un bar chart per confrontare le performance di diversi modelli."""
    print("\n--- Confronto Performance Modelli ---")
    plt.figure(figsize=(10, 6))
    bars = plt.bar(results.keys(), results.values())
    plt.ylabel('Miglior Validation mIoU')
    plt.title('Confronto Performance Modelli')
    plt.ylim(0, max(results.values()) * 1.2)
    plt.xticks(rotation=15, ha="right")
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, f'{yval:.4f}', va='bottom', ha='center')
        
    plt.show()

# =============================================================================
# 5. CORE TRAINING & EVALUATION LOGIC
# =============================================================================
def combined_loss_fn(pred, target, ce, dice):
    return 0.4 * ce(pred, target) + 0.6 * dice(pred, target)

def compute_iou(mask_true, mask_pred, num_classes):
    iou_scores = np.zeros(num_classes - 1)
    for label in range(1, num_classes):
        intersection = np.sum((mask_true == label) & (mask_pred == label))
        union = np.sum((mask_true == label) | (mask_pred == label))
        iou_scores[label - 1] = intersection / union if union > 0 else np.nan
    return iou_scores

def evaluate(model, loader, device, num_classes):
    model.eval()
    all_ious = []
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione", leave=False):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            for i in range(len(preds)):
                all_ious.append(compute_iou(masks[i].numpy(), preds[i], num_classes))
    return float(np.nanmean(all_ious))

def training_loop(model, config, train_loader, val_loader, optimizer, scheduler, ce_loss, dice_loss):
    """Esegue il ciclo completo di training e validazione."""
    history = {'train_loss': [], 'val_iou': []}
    best_iou = -1.0
    
    for epoch in range(config["NUM_EPOCHS"]):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['NUM_EPOCHS']} [Training]")
        
        for images, masks in progress_bar:
            images, masks = images.to(config["DEVICE"]), masks.to(config["DEVICE"], dtype=torch.long)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = combined_loss_fn(outputs, masks, ce_loss, dice_loss)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
        epoch_loss = running_loss / len(train_loader)
        val_iou = evaluate(model, val_loader, config["DEVICE"], config["NUM_CLASSES"])
        scheduler.step(val_iou)
        
        history['train_loss'].append(epoch_loss)
        history['val_iou'].append(val_iou)
        
        print(f"Epoch {epoch+1} -> Loss: {epoch_loss:.4f} | Val mIoU: {val_iou:.4f} | LR: {optimizer.param_groups[0]['lr']:.1e}")

        if val_iou > best_iou:
            best_iou = val_iou
            
    return history, best_iou

# =============================================================================
# 6. MAIN FUNCTION
# =============================================================================
def main():
    config = get_config()
    device = config["DEVICE"]
    print(f"Device in uso: {device}")

    # --- Data ---
    train_dirs, val_dirs = create_splits(config["DATASET_PATH"], config["VALIDATION_SPLIT"], config["RANDOM_SEED"])
    train_dataset = SegmentationDataset(train_dirs, config["OUTPUT_SIZE"], config["DATASET_MEAN"], config["DATASET_STD"], True)
    val_dataset = SegmentationDataset(val_dirs, config["OUTPUT_SIZE"], config["DATASET_MEAN"], config["DATASET_STD"], False)
    train_loader = DataLoader(train_dataset, config["BATCH_SIZE"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, config["BATCH_SIZE"], num_workers=2, pin_memory=True)

    # --- Model & Components ---
    model = smp.Unet(
        encoder_name=config["ENCODER"],
        encoder_weights=config["ENCODER_WEIGHTS"],
        classes=config["NUM_CLASSES"]
    ).to(device)
    
    if os.path.exists(config["MODEL_CHECKPOINT"]):
        print(f"Caricamento pesi da: {config['MODEL_CHECKPOINT']}")
        model.load_state_dict(torch.load(config["MODEL_CHECKPOINT"], map_location=device))
    else:
        print("Nessun checkpoint trovato, si parte da pesi pre-addestrati (ImageNet).")
        
    ce_loss = nn.CrossEntropyLoss(ignore_index=config["IGNORE_INDEX"])
    dice_loss = DiceLoss(mode='multiclass')
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=True)

    # --- Training ---
    history, best_iou = training_loop(model, config, train_loader, val_loader, optimizer, scheduler, ce_loss, dice_loss)
    print(f"\nAddestramento completato. Miglior mIoU di validazione: {best_iou:.4f}")

    # --- Post-Training Analysis ---
    # Carica il modello migliore per l'analisi finale
    print(f"Caricamento del modello migliore da '{config['MODEL_SAVE_PATH']}' per l'analisi.")
    model.load_state_dict(torch.load(config["MODEL_SAVE_PATH"], map_location=device))
    
    plot_learning_curves(history)
    visualize_predictions(model, val_loader, config)
    
    # Dati per il confronto, aggiornati con il risultato corrente
    comparison_results = {
        'UNet Micro (Low Res)': 0.4167,
        'UNet Micro (High Res)': 0.3285,
        'UNet+ResNet34 (Questa Run)': best_iou,
    }
    plot_model_comparison(comparison_results)

# =============================================================================
# 7. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()