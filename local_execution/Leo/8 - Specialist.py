# SCRIPT DI TRAINING PER IL MODELLO SPECIALISTA - EDIZIONE CON LOSS PESATA IBRIDA (CORRETTO)

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import random
import glob
from typing import Tuple, List, Dict, Any
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    class autocast:
        def __init__(self, enabled=False, **kwargs): self.enabled=enabled
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    class GradScaler:
        def __init__(self, enabled=False): self.enabled=enabled
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 2. RIPRODUCIBILITÀ E CONFIGURAZIONE
# =============================================================================
def seed_everything(seed: int):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def get_config() -> Dict[str, Any]:
    base_path = "./"
    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "RANDOM_SEED": 42, "NUM_WORKERS": 2, "USE_AMP": True,
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_specialist_model_weighted_loss.pth"),
        "ENCODER": "efficientnet-b2", "ENCODER_WEIGHTS": "imagenet",
        "OUTPUT_SIZE": (384, 768), "BATCH_SIZE": 8, "VALIDATION_SPLIT": 0.2,
        "TARGET_CLASSES": {2: 0, 3: 1, 6: 2, 7: 3},
        "NUM_SPECIALIST_CLASSES": 4, "IGNORE_INDEX_SPECIALIST": -1,
        "NUM_EPOCHS": 25, "LEARNING_RATE": 1e-3, "WEIGHT_DECAY": 1e-5,
    }
    config["CLASS_NAMES_SPECIALIST"] = ["Spec_Class2", "Spec_Class3", "Spec_Class6", "Spec_Class7"]
    return config

# =============================================================================
# 3. DATASET SPECIALIZZATO E CALCOLO PESI
# =============================================================================
class SubproblemDataset(Dataset):
    # ... (invariato) ...
    def __init__(self, sample_dirs: List[str], transform: A.Compose, config: Dict[str, Any], filter_data=True):
        self.transform, self.config = transform, config
        self.target_class_keys = list(config["TARGET_CLASSES"].keys())
        self.class_map = np.full(256, config["IGNORE_INDEX_SPECIALIST"], dtype=np.int64)
        for original_id, new_id in config["TARGET_CLASSES"].items(): self.class_map[original_id] = new_id
        self.sample_dirs = self._filter_dirs(sample_dirs) if filter_data else sample_dirs
    def _filter_dirs(self, all_dirs):
        tqdm.write("Filtraggio del dataset per le classi di interesse...")
        filtered_dirs = [p for p in tqdm(all_dirs, desc="Filtraggio") if np.any(np.isin(np.array(Image.open(os.path.join(p, 'labels.png'))), self.target_class_keys))]
        tqdm.write(f"Dataset filtrato: {len(filtered_dirs)} / {len(all_dirs)} campioni contengono le classi target.")
        return filtered_dirs
    def __len__(self) -> int: return len(self.sample_dirs)
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        image = np.array(Image.open(os.path.join(sample_dir, 'rgb.jpg')).convert("RGB"))
        mask_original = np.array(Image.open(os.path.join(sample_dir, 'labels.png')))
        mask_remapped = self.class_map[mask_original]
        augmented = self.transform(image=image, mask=mask_remapped)
        return augmented['image'], augmented['mask']

def get_transforms(config: Dict[str, Any], is_train: bool) -> A.Compose:
    # ... (invariato) ...
    mean_std = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    if is_train: return A.Compose([A.Resize(height=config["OUTPUT_SIZE"][0], width=config["OUTPUT_SIZE"][1], interpolation=cv2.INTER_NEAREST), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.ShiftScaleRotate(p=0.5), A.ColorJitter(p=0.5), A.CoarseDropout(p=0.3), A.Normalize(mean=mean_std[0], std=mean_std[1]), ToTensorV2()])
    else: return A.Compose([A.Resize(height=config["OUTPUT_SIZE"][0], width=config["OUTPUT_SIZE"][1], interpolation=cv2.INTER_NEAREST), A.Normalize(mean=mean_std[0], std=mean_std[1]), ToTensorV2()])

def calculate_specialist_weights(loader: DataLoader, config: Dict[str, Any]) -> torch.Tensor:
    # ... (invariato) ...
    tqdm.write("Calcolo dei pesi per le classi specialistiche...")
    pixel_counts = torch.zeros(config["NUM_SPECIALIST_CLASSES"], dtype=torch.float64)
    for _, masks in tqdm(loader, desc="Analisi distribuzione pixel"):
        masks = masks.long(); valid_pixels = masks != config["IGNORE_INDEX_SPECIALIST"]
        counts = torch.bincount(masks[valid_pixels], minlength=config["NUM_SPECIALIST_CLASSES"]); pixel_counts += counts.double()
    total_pixels = pixel_counts.sum(); class_frequencies = pixel_counts / total_pixels
    weights = 1.0 / (class_frequencies + 1e-6); weights = weights / weights.mean()
    weights_tensor = weights.float().to(config["DEVICE"])
    tqdm.write("\nPesi calcolati per la loss dello specialista:"); 
    for i, w in enumerate(weights_tensor): tqdm.write(f"  - {config['CLASS_NAMES_SPECIALIST'][i]}: {w.item():.2f}")
    return weights_tensor

# =============================================================================
# 4. MODELLO E LOSS IBRIDA
# =============================================================================
class CombinedSpecialistLoss(nn.Module):
    # ... (invariato) ...
    def __init__(self, class_weights: torch.Tensor, config: Dict[str, Any]):
        super().__init__(); self.ignore_index = config["IGNORE_INDEX_SPECIALIST"]
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=self.ignore_index)
        self.dice = DiceLoss(mode='multiclass', from_logits=True, ignore_index=self.ignore_index)
        self.focal = FocalLoss(mode='multiclass', gamma=2, ignore_index=self.ignore_index)
    def forward(self, pred, target): return 0.3 * self.ce(pred, target) + 0.4 * self.dice(pred, target) + 0.3 * self.focal(pred, target)

def get_model(config: Dict[str, Any]) -> nn.Module:
    # ... (invariato) ...
    model = smp.Unet(encoder_name=config["ENCODER"], encoder_weights=config["ENCODER_WEIGHTS"], in_channels=3, classes=config["NUM_SPECIALIST_CLASSES"])
    tqdm.write(f"Creazione del MODELLO SPECIALISTA: U-Net con encoder '{config['ENCODER']}'"); return model

# =============================================================================
# 5. FUNZIONI DI TRAINING/VALIDAZIONE (CON FIX)
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    # ... (invariato) ...
    model.train(); running_loss = 0.0; progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device, dtype=torch.long)
        with autocast(enabled=use_amp, device_type=device.type):
            outputs = model(images); loss = criterion(outputs, masks)
        optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        running_loss += loss.item(); progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader)

def evaluate_specialist(model, loader, criterion, device, config) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """FIX: Funzione modificata per calcolare e restituire la Matrice di Confusione."""
    model.eval(); total_loss = 0.0
    conf_matrix = torch.zeros(config["NUM_SPECIALIST_CLASSES"], config["NUM_SPECIALIST_CLASSES"], dtype=torch.int64, device=device)
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione Specialista", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            with autocast(enabled=config["USE_AMP"], device_type=device.type):
                outputs = model(images); loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            valid = masks != config["IGNORE_INDEX_SPECIALIST"]
            # Calcolo efficiente della matrice di confusione del batch
            indices = masks[valid] * config["NUM_SPECIALIST_CLASSES"] + preds[valid]
            counts = torch.bincount(indices, minlength=config["NUM_SPECIALIST_CLASSES"]**2)
            conf_matrix += counts.reshape(config["NUM_SPECIALIST_CLASSES"], config["NUM_SPECIALIST_CLASSES"])

    # Calcolo IoU dalla matrice di confusione aggregata (più efficiente)
    tp = torch.diag(conf_matrix)
    fp = conf_matrix.sum(axis=0) - tp
    fn = conf_matrix.sum(axis=1) - tp
    union = tp + fp + fn
    
    iou_per_class = (tp / (union + 1e-6)).cpu().numpy()
    mean_iou = np.nanmean(iou_per_class)
    
    return total_loss / len(loader), mean_iou, iou_per_class, conf_matrix.cpu().numpy()

def plot_confusion_matrix(conf_matrix_np, class_names, save_path="confusion_matrix_specialist.png"):
    """FIX: Funzione di plotting pura che accetta una matrice pre-calcolata."""
    tqdm.write("Visualizzazione e salvataggio Matrice di Confusione...")
    row_sums = conf_matrix_np.sum(axis=1)[:, np.newaxis]
    norm_conf_matrix = np.divide(conf_matrix_np, row_sums, out=np.zeros_like(conf_matrix_np, dtype=float), where=row_sums!=0)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(norm_conf_matrix, annot=True, fmt='.2%', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12); ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Matrice di Confusione Specialista (Normalizzata per Riga)', fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    tqdm.write(f"Matrice di confusione salvata in '{save_path}'")

# =============================================================================
# 6. FUNZIONE PRINCIPALE (CON FIX)
# =============================================================================
def main():
    config = get_config(); seed_everything(config["RANDOM_SEED"]); device = config["DEVICE"]
    tqdm.write(f"Device: {device} | Strategia: Training Specialista con Loss Pesata Ibrida")

    paths = {"original": config["DATASET_PATH_TRAIN"], "augmented": config["DATASET_PATH_AUG"], "aggressive": config["DATASET_PATH_AUG_AGGR"]}
    all_dirs = [d for p in paths.values() if os.path.isdir(p) for d in glob.glob(os.path.join(p, "*")) if os.path.isdir(d)]; all_dirs = sorted(list(set(all_dirs)))
    train_dirs, val_dirs = train_test_split(all_dirs, test_size=config["VALIDATION_SPLIT"], random_state=config["RANDOM_SEED"])

    train_transforms = get_transforms(config, is_train=True)
    val_transforms = get_transforms(config, is_train=False)
    
    full_train_dataset_for_weights = SubproblemDataset(train_dirs, val_transforms, config, filter_data=False)
    weights_loader = DataLoader(full_train_dataset_for_weights, batch_size=config["BATCH_SIZE"]*2, shuffle=False, num_workers=config["NUM_WORKERS"])
    class_weights = calculate_specialist_weights(weights_loader, config)

    train_dataset = SubproblemDataset(train_dirs, train_transforms, config, filter_data=True); val_dataset = SubproblemDataset(val_dirs, val_transforms, config, filter_data=True)
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"], pin_memory=True)

    model = get_model(config).to(device)
    criterion = CombinedSpecialistLoss(class_weights=class_weights, config=config)
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, verbose=True)
    scaler = GradScaler(enabled=config["USE_AMP"])

    best_iou = -1.0; best_conf_matrix = None;
    tqdm.write("\n--- Inizio Training Specialista (Loss Pesata) ---")
    for epoch in range(config["NUM_EPOCHS"]):
        tqdm.write(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, config["USE_AMP"])
        
        # FIX: Unpacking corretto dei valori restituiti
        val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate_specialist(model, val_loader, criterion, device, config)
        scheduler.step(val_miou)
        
        tqdm.write(f"Risultati Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        for i, iou in enumerate(val_iou_per_class):
            tqdm.write(f"    - IoU {config['CLASS_NAMES_SPECIALIST'][i]}: {iou:.4f}")
            
        if val_miou > best_iou:
            best_iou = val_miou
            # FIX: Salvataggio della matrice di confusione corretta
            best_conf_matrix = val_conf_matrix
            torch.save({'model_state_dict': model.state_dict(), 'config': config}, config["MODEL_SAVE_PATH"])
            tqdm.write(f"\n>>> 🎉 Nuovo modello specialista migliore salvato! mIoU: {best_iou:.4f} <<<")

    tqdm.write(f"\n--- Training Specialista Completato. Miglior mIoU: {best_iou:.4f} ---")
    
    # FIX: Chiamata corretta a plot_confusion_matrix con i dati giusti
    if best_conf_matrix is not None:
        plot_confusion_matrix(best_conf_matrix, config["CLASS_NAMES_SPECIALIST"])

# =============================================================================
# 7. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()