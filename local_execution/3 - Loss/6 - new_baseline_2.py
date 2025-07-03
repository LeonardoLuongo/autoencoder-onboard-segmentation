# SCRIPT DI TRAINING FINALE - EDIZIONE PROFESSIONALE

# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import random
import glob
from typing import Tuple, List, Dict, Any
import cv2 # Importato per l'interpolazione NEAREST

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Import AMP retro-compatibile e robusto
try:
    from torch.cuda.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    # Placeholder per ambienti senza CUDA/AMP
    class autocast:
        def __init__(self, enabled=False): self.enabled = enabled
        def __enter__(self): pass
        def __exit__(self, exc_type, exc_val, exc_tb): pass
    class GradScaler:
        def __init__(self, enabled=False): self.enabled = enabled
        def scale(self, loss): return loss
        def step(self, optimizer): optimizer.step()
        def update(self): pass

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
# 2. RIPRODUCIBILITÀ
# =============================================================================
def seed_everything(seed: int):
    """Fissa la seed per garantire la riproducibilità degli esperimenti."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =============================================================================
# 3. CONFIGURAZIONE
# =============================================================================
def get_config() -> Dict[str, Any]:
    base_path = "./"
    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_model.pth"), # Nome generico per il modello migliore
        
        # --- Configurazione conservativa per stabilità ---
        "ENCODER": "efficientnet-b2",
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
        "NUM_WORKERS_TRAIN": 2,
        "NUM_WORKERS_VAL": 1,
        "USE_AMP": True,
        
        "RARE_CLASS_INDICES": [2, 6, 8], # Rough Trail, LowVeg, Obstacle
    }
    
    config["CLASS_NAMES"] = [
        "Ignored", "Sky", "Rough", "Smooth", "Grass", 
        "HighVeg", "LowVeg", "Puddle", "Obstacle" # FIX: Rinominata classe 7
    ]
    return config

# =============================================================================
# 4. GESTIONE DATI
# =============================================================================
def get_grouped_stratified_split(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    # ... (invariata, ma robusta) ...
    paths = {
        "original": config["DATASET_PATH_TRAIN"],
        "augmented": config["DATASET_PATH_AUG"],
        "aggressive": config["DATASET_PATH_AUG_AGGR"],
    }
    all_dirs = []
    tqdm.write("Caricamento percorsi dei dataset...")
    for key, path in paths.items():
        if not os.path.isdir(path):
            tqdm.write(f"Attenzione: il percorso '{path}' non esiste, verrà saltato.")
            continue
        dirs = [d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)]
        tqdm.write(f"Trovati {len(dirs)} campioni in '{key}'.")
        all_dirs.extend(dirs)
    if not all_dirs: raise FileNotFoundError("Nessuna directory di dati trovata.")
    all_dirs = sorted(list(set(all_dirs)))

    hashes = {path: imagehash.phash(Image.open(os.path.join(path, 'rgb.jpg'))) for path in tqdm(all_dirs, desc="Calcolo Hash")}
    G = nx.Graph(list(nx.connected_components(nx.from_edgelist(
        (u, v) for i, u in enumerate(list(hashes.keys())) for j, v in enumerate(list(hashes.keys())) if i < j and hashes[u] - hashes[v] < 5
    ))))
    image_groups = list(nx.connected_components(G))
    tqdm.write(f"Trovati {len(image_groups)} gruppi di immagini uniche.")
    
    group_labels = []
    for group in tqdm(image_groups, desc="Etichettatura gruppi per lo split"):
        has_rare = any(
            rare_id in np.unique(np.array(Image.open(os.path.join(p, 'labels.png'))))
            for p in group for rare_id in config["RARE_CLASS_INDICES"]
        )
        group_labels.append(1 if has_rare else 0)
        
    train_groups, val_groups = train_test_split(
        image_groups, test_size=config["VALIDATION_SPLIT"],
        random_state=config["RANDOM_SEED"], stratify=group_labels
    )
    train_dirs = [p for group in train_groups for p in group]
    val_dirs = [p for group in val_groups for p in group]
    tqdm.write(f"Split stratificato creato: {len(train_dirs)} train, {len(val_dirs)} validation.")
    return train_dirs, val_dirs

def get_transforms(output_size: Tuple[int, int]) -> A.Compose:
    return A.Compose([
        # FIX: Aggiunta interpolazione NEAREST per le maschere, per evitare di creare ID non interi.
        A.Resize(height=output_size[0], width=output_size[1], interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

class SegmentationDataset(Dataset):
    def __init__(self, sample_dirs: List[str], transform: A.Compose, config: Dict[str, Any]):
        self.sample_dirs, self.transform, self.config = sample_dirs, transform, config

    def __len__(self) -> int: return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        image = np.array(Image.open(os.path.join(sample_dir, 'rgb.jpg')).convert("RGB"))
        mask = np.array(Image.open(os.path.join(sample_dir, 'labels.png')))
        
        # FIX: Asserzioni di robustezza sulla maschera
        assert mask.dtype == np.uint8, f"Tipo maschera non corretto: {mask.dtype}"
        assert mask.max() < self.config["NUM_CLASSES"], f"ID classe non valido in {os.path.join(sample_dir, 'labels.png')}"
            
        return self.transform(image=image, mask=mask).values()

# =============================================================================
# 5. MODELLO, LOSS E METRICHE
# =============================================================================
def get_model(config: Dict[str, Any]) -> nn.Module:
    model = smp.Unet(
        encoder_name=config["ENCODER"], encoder_weights=config["ENCODER_WEIGHTS"],
        in_channels=3, classes=config["NUM_CLASSES"]
    )
    tqdm.write(f"Creazione del modello: U-Net con encoder '{config['ENCODER']}'")
    tqdm.write(f"Parametri addestrabili: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model

class CombinedDiceFocalLoss(nn.Module):
    def __init__(self, dice_weight=0.7, focal_weight=0.3, focal_gamma=2.0):
        super().__init__()
        # FIX: Ponderazione modificata per dare più peso a Dice/IoU
        self.dice = DiceLoss(mode='multiclass', from_logits=True)
        self.focal = FocalLoss(mode='multiclass', gamma=focal_gamma)
        self.dice_weight, self.focal_weight = dice_weight, focal_weight

    def forward(self, pred, target):
        return self.dice_weight * self.dice(pred, target) + self.focal_weight * self.focal(pred, target)

def compute_iou(preds, labels, num_classes, ignore_index):
    iou_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        if cls == ignore_index: continue
        pred_inds, target_inds = (preds == cls), (labels == cls)
        intersection, union = np.logical_and(pred_inds, target_inds).sum(), np.logical_or(pred_inds, target_inds).sum()
        iou_per_class[cls] = intersection / union if union > 0 else np.nan
    return np.nanmean(iou_per_class), iou_per_class

def plot_confusion_matrix(conf_matrix_np, class_names, ignore_index):
    tqdm.write("Visualizzazione Matrice di Confusione...")
    plot_matrix, plot_names = conf_matrix_np.copy(), class_names.copy()
    if ignore_index is not None:
        plot_matrix = np.delete(np.delete(plot_matrix, ignore_index, axis=0), ignore_index, axis=1)
        plot_names = [name for i, name in enumerate(class_names) if i != ignore_index]

    row_sums = plot_matrix.sum(axis=1)[:, np.newaxis]
    norm_conf_matrix = np.divide(plot_matrix, row_sums, out=np.zeros_like(plot_matrix, dtype=float), where=row_sums!=0)

    plt.figure(figsize=(12, 10))
    sns.heatmap(norm_conf_matrix, annot=True, fmt='.2%', cmap='Blues', xticklabels=plot_names, yticklabels=plot_names)
    plt.xlabel('Predicted Label', fontsize=12); plt.ylabel('True Label', fontsize=12)
    plt.title('Matrice di Confusione (Normalizzata per Riga)', fontsize=14); plt.show()

# =============================================================================
# 6. CICLI DI TRAINING E VALIDAZIONE
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

def evaluate(model, loader, criterion, device, config) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, all_preds_flat, all_labels_flat = 0.0, [], []
    # FIX: Calcolo della matrice di confusione memoria-efficiente
    conf_matrix = torch.zeros(config["NUM_CLASSES"], config["NUM_CLASSES"], dtype=torch.int64, device=device)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            with autocast(enabled=True, device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            masks_flat, preds_flat = masks.flatten(), preds.flatten()
            valid = masks_flat != config["IGNORE_INDEX"]
            indices = masks_flat[valid] * config["NUM_CLASSES"] + preds_flat[valid]
            counts = torch.bincount(indices, minlength=config["NUM_CLASSES"]**2)
            conf_matrix += counts.reshape(config["NUM_CLASSES"], config["NUM_CLASSES"])
            
            all_preds_flat.append(preds_flat[valid].cpu())
            all_labels_flat.append(masks_flat[valid].cpu())

    avg_loss = total_loss / len(loader)
    all_preds_flat, all_labels_flat = torch.cat(all_preds_flat).numpy(), torch.cat(all_labels_flat).numpy()
    mean_iou, iou_per_class = compute_iou(all_preds_flat, all_labels_flat, config["NUM_CLASSES"], config["IGNORE_INDEX"])
    
    return avg_loss, mean_iou, iou_per_class, conf_matrix.cpu().numpy()

# =============================================================================
# 7. FUNZIONE PRINCIPALE
# =============================================================================
def main():
    config = get_config()
    seed_everything(config["RANDOM_SEED"])
    device = config["DEVICE"]
    use_amp = config["USE_AMP"] and AMP_AVAILABLE and device.type == 'cuda'

    tqdm.write(f"Device: {device} | Mixed Precision: {use_amp}")
    tqdm.write(f"Production Run: {config['ENCODER']} con pipeline dati professionale")

    train_dirs, val_dirs = get_grouped_stratified_split(config)
    transforms = get_transforms(config["OUTPUT_SIZE"])
    train_dataset = SegmentationDataset(train_dirs, transforms, config)
    val_dataset = SegmentationDataset(val_dirs, transforms, config)
    
    pin_mem = device.type == 'cuda' # FIX: usa pin_memory solo con GPU
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS_TRAIN"], pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS_VAL"], pin_memory=pin_mem)

    model = get_model(config).to(device)
    criterion = CombinedDiceFocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'max', patience=3, factor=0.2, cooldown=2, min_lr=1e-6, verbose=True # FIX: Scheduler migliorato
    )
    scaler = GradScaler(enabled=use_amp)

    best_iou = -1.0
    best_conf_matrix = None
    
    tqdm.write("\n--- Inizio Training ---")
    for epoch in range(config["NUM_EPOCHS"]):
        tqdm.write(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate(model, val_loader, criterion, device, config)
        scheduler.step(val_miou)
        
        tqdm.write(f"Risultati Epoch {epoch+1}:")
        tqdm.write(f"  - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        tqdm.write("  - IoU per Classe (Validation):")
        for i, iou in enumerate(val_iou_per_class):
            if i != config["IGNORE_INDEX"]:
                tqdm.write(f"    - Classe {i} ({config['CLASS_NAMES'][i]}): {iou:.4f}")
        
        if val_miou > best_iou:
            best_iou = val_miou
            best_conf_matrix = val_conf_matrix
            # FIX: Salvataggio robusto con configurazione inclusa
            save_payload = {'model_state_dict': model.state_dict(), 'config': config}
            torch.save(save_payload, config["MODEL_SAVE_PATH"])
            tqdm.write(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<")

    tqdm.write("\n--- Training Completato ---")
    tqdm.write(f"Miglior mIoU di validazione ottenuto: {best_iou:.4f}")

    if best_conf_matrix is not None:
        plot_confusion_matrix(best_conf_matrix, config["CLASS_NAMES"], config["IGNORE_INDEX"])
    
    tqdm.write("\nAnalisi finale completata. Per eseguire le predizioni su nuovi dati, usare uno script separato "
               "caricando i pesi da 'best_model.pth' e implementando la funzione predict(X).")

# =============================================================================
# 8. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()