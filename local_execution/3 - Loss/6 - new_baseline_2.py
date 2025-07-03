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
    from torch.amp import GradScaler, autocast
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

    # 1. Calcola gli hash per ogni immagine
    hashes = {path: imagehash.phash(Image.open(os.path.join(path, 'rgb.jpg'))) 
              for path in tqdm(all_dirs, desc="Calcolo Hash")}
    
    # 2. Crea un grafo vuoto e aggiungi tutti i percorsi come nodi
    G = nx.Graph()
    G.add_nodes_from(hashes.keys())
    
    # 3. Aggiungi gli archi (collegamenti) tra le immagini simili
    #    Questo è il modo corretto e leggibile per farlo.
    path_list = list(hashes.keys())
    for i in range(len(path_list)):
        for j in range(i + 1, len(path_list)):
            if hashes[path_list[i]] - hashes[path_list[j]] < 5:
                G.add_edge(path_list[i], path_list[j])
                
    # 4. Ora, su questo grafo corretto, trova i gruppi di immagini connesse
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
            
        return tuple(self.transform(image=image, mask=mask).values())

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
        optimizer, 'max', patience=3, factor=0.2, cooldown=2, min_lr=1e-6
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

'''

Etichettatura gruppi per lo split: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 2475/2475 [00:24<00:00, 99.67it/s] 
















Split stratificato creato: 2779 train, 673 validation.
Creazione del modello: U-Net con encoder 'efficientnet-b2'
Parametri addestrabili: 10,047,451

--- Inizio Training ---

--- Epoch 1/30 ---
Risultati Epoch 1:                                                                                                                                                                
  - Train Loss: 0.7209 | Val Loss: 0.5485 | Val mIoU: 0.1905
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.0025
    - Classe 2 (Rough): 0.0956
    - Classe 3 (Smooth): 0.0004
    - Classe 4 (Grass): 0.0000
    - Classe 5 (HighVeg): 0.0201
    - Classe 6 (LowVeg): 0.2330
    - Classe 7 (Puddle): 0.6336
    - Classe 8 (Obstacle): 0.7296

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1905 <<<

--- Epoch 2/30 ---
  - Train Loss: 0.5885 | Val Loss: 0.5098 | Val mIoU: 0.3175
  - Train Loss: 0.5885 | Val Loss: 0.5098 | Val mIoU: 0.3175
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.4298
    - Classe 2 (Rough): 0.3439
    - Classe 1 (Sky): 0.4298
    - Classe 2 (Rough): 0.3439
    - Classe 3 (Smooth): 0.0002
    - Classe 4 (Grass): 0.0000
    - Classe 5 (HighVeg): 0.2928
    - Classe 6 (LowVeg): 0.2118
    - Classe 7 (Puddle): 0.7491
    - Classe 8 (Obstacle): 0.8301

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3175 <<<

--- Epoch 3/30 ---
Training:  47%|██████████████████████████████████████████████████████                                                             | 327/695 [01:16<01:10,  5.22it/s, loss=0.5497]
Training:  47%|██████████████████████████████████████████████████████                                                             | 327/690.44Training:  47%|██████████████████████████████████████████████████████▎                                                            | 32Training:  47%|██████████████████████████████████████████████████████                                                             | 327/69=0.44Training:  47%|██████████████████████████████████████████████████████▎                                                            | 3 losTraining:  47%|██████████████████████████████████████████████████████▎                                                            | 32, loss=0Training:  47%|██████████████████████████████████████████████████████▍                                                            it/s, loTraining:  47%|██████████████████████████████████████████████████████▍                                                            10it/s, loss=Training:  47%|██████████████████████████████████████████████████████▌                                                       ,  5.02it/s, loss=Training:  47%|██████████████████████████████████████████████████████▌                                                  01:12,  5.02it/s, loss=Training:  48%|██████████████████████████████████████████████████████▊                                             ,  5.02it/s, loss=Training:  47%|██████████████████████████████████████████████████████▌                                                  <01:12,  5.02it/s, loss=Training:  48%|██████████████████████████████████████████████████████▊                                            [01:17<01:11,  5.06it/s, loss=Training:  48%|██████████████████████████████████████████████████████▊                                       331/695 [01:17<01:11,  5.06it/s, loss=Training:  48%|██████████████████████████████████████████████████████▉                                     | 332/695 [01:17<01:10,  5.14it/s, loss=Training:  48%|██████████████████████████████████████████████████████▉                                     | 332/695 [01:17<01:10,  5.14it/s, loss=Training:  48%|███████████████████████████████████████████████████████                 | 332/695 [01:17<01:10,  5.14it/s, loss=Training:  48%|██████████████████████████████████████████████████████▉                                      | 332/695 [01:17<01:10,  5.14it/s, loss=Training:  48%|███████████████████████████████████████████████████████                                      | 333/695 [01:17<01:09,  5.18it/s, loss=Training:  48%|███████████████████████████████████████████████████████                                         | 333/695 [01:17<01:09,  5.18it/s, loss=Training:  48%|████████████████████████████████████████████                                                    | 334/695 [01:17<01:09,  5.23it/s, loss=Training:  48%|████████████████████████████████████▎                                                           | 334/695 [01:17<01:09,  5.23it/s, loss=Training:  48%|██████████████████                                                  | 334/695 [01:17<01:09,  5.23it/s, loss=Training:  48%|███████████████████████████████████████▎                                                           | 334/695 [01:17<01:09,  5.23it/s, loss=Training:  48%|███████████████████████████████████████▍                                                           | 335/695 [01:17<01:12, Training:  48%|███████████████████████████████████████▍                                                           | 335/695 [01:18<01:12,  4.98it/s, lossTraining:  48%|██████████████████████▍                                                           | 335/695 [01:17<01:12, Training:  48%|█████████████████████████████████████████▍                                                           | 335/695 [01:18<01:12,  4.98it/s, lossTraining:  48%|█████████████████████████████████████████▌                                                           | 336/6Training:  48%|█████████████████████████████████████████▌                                                           | 336/695 [01:18<01:11,  5.03it/s, loTraining:  48%|█████████████████████████████████████████▊                                                           | 337Training:  48%|█████████████████████████████████████████▊                                                           | 337/695 [01:18<01:12,  4.91it/s, Training:  49%|█████████████████████████████████████████████▊                                                           | 337Training:  48%|██████████████████████████████████████████████████████████▊                                                           | 337Training:  48%|██████████████████████████████████████████████████████████▊                                                           | 337Training:  48%|██████████████████████████████████████████████████████████▊                                                           | 337Training:  48%|███████████████████████████████████████████████████▊                                                           | 337/695 [01:18<01:12,  4.91it/s, Training:  ███████████████████████████████████████████████████████▉                                                           | 338/695 [01:18<01Trai:  49%|███████████████████████████████████████████████████████▉                                                           | 338/695 [01:18█████████████████████████████████████████████████▉                                                           | 338/695 [01:18<01Training: %|███████████████████████████████████████████████████████▉                                                           | 338/695 [01:18<01:1 4.84it/s,Training:  49%|████████████████████████████████████████████████████████                                                          Training:  49%|████████████████████████████████████████████████████████                                                           | 339/6 4.84it/s,Training:  49%|████████████████████████████████████████████████████████                                                           | Training:  49%|████████████████████████████████████████████████████████                                                           | 339/695 [01:18<01:14,  4.78it/Training:  49%|████████████████████████████████████████████████████████▎                                                          Training:  49%|████████████████████████████████████████████████████████▎                                              695 [01:18<01:14,  4.78it/Training:  49%|████████████████████████████████████████████████████████▎                                                          Training:  49%|████████████████████████████████████████████████████████▎                                                          | 340/695 [01:19<01:13,  4.82iTraining:  49%|████████████████████████████████████████████████████████▍                                                        Training:  49%|████████████████████████████████████████████████████████▍                                                          | 341/695 [01:19<01:12,  4.8Training:  49%|████████████████████████████████████████████████████████▌                                      Training:  49%|████████████████████████████████████████████████████████▌                                                          | 342/695 [01:19<01:12,  Training:  49%|████████████████████████████████████████████████████████▊                                   Training:  49%|████████████████████████████████████████████████████████▊                                                          | 343/695 [01:19<01:12Training:  49%|████████████████████████████████████████████████████████▉                                                          | 344/695 [01:19<01:12Training:  49%|████████████████████████████████████████████████████████▉                                                          | 344/695 [01:19<01:12Training:  50%|█████████████████████████Training:  50%|█████████████████████████████████████████████████████████                                                          | 345/695 [01:2Training:  50%|█████████████████████████████████████████████████████████▎        Training:  50%|█████████████████████████████████████████████████████████▎                                                         | 346/695 [Training:  50%|█████████████████████████████████████████████████████████▍                                    Training:  50%|█████████████████████████████████████████████████████████▍                                                         | 347/695Training:  50%|█████████████████████████████████████████████████████████▌                                                  Training:  50%|█████████████████████████████████████████████████████████▌                                                         | 348/69Training:  50%|█████████████████████████████████████████████████████████▋                                                         | 349/69Training:  50%|█████████████████████████████████████████████████████████▋                                                         | 349/69Training:  50%|█████████████████████████████████████████████████████████▉                                                         | 350/69Training:  50%|█████████████████████████████████████████████████████████▉                                                         | 350/69Training:  51%|██████████████████████████████████████████████████████████                                                         | 351/695 [Training:  51%|██████████████████████████████████████████████████████████▋                             Risultati Epoch 3:
  - Train Loss: 0.5532 | Val Loss: 0.4983 | Val mIoU: 0.3497
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.4445
    - Classe 2 (Rough): 0.3528
    - Classe 3 (Smooth): 0.0275
    - Classe 4 (Grass): 0.0000
    - Classe 5 (HighVeg): 0.4501
    - Classe 6 (LowVeg): 0.2598
    - Classe 7 (Puddle): 0.7661
    - Classe 8 (Obstacle): 0.8467

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3497 <<<

--- Epoch 4/30 ---
Risultati Epoch 4:                                                                                Risultati Epoch 4:
  - Train Loss: 0.5255 | Val Loss: 0.4715 | Val mIoU: 0.3524
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.4270
    - Classe 2 (Rough): 0.3896
    - Classe 3 (Smooth): 0.0198
    - Classe 4 (Grass): 0.0001
    - Classe 5 (HighVeg): 0.5155
    - Classe 6 (LowVeg): 0.2818
    - Classe 7 (Puddle): 0.6848
    - Classe 8 (Obstacle): 0.8527

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3524 <<<

--- Epoch 5/30 ---
Training:  94%|███████████████████████████████████████████████████████████████████████████████████Training:  94%|███████████████████████████████████████████████████████████████████████████████████Training:  94%|███████████████████████████████████████████████████████████████████████████████████Training:  94%|███████████████████████████████████████████████████████████████████████████████████Training:  94%|███████████████████████████████████████████████████████████████████████████████████Training:  94%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  95%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  96%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  97%|███████████████████████████████████████████████████████████████████████████████████Training:  98%|███████████████████████████████████████████████████████████████████████████████████Training:  98%|███████████████████████████████████████████████████████████████████████████████████Training:  98%|███████████████████████████████████████████████████████████████████████████████████Training:  98%|███████████████████████████████████████████████████████████████████████████████████Training:  98%|███████████████████████████████████████████████████████████████████████████████████Training:  98%|███████████████████████████████████████████████████████████████████████████████████Training:  98%|███████████████████████████████████████████████████████████████████████████████████Training:  98%|███████████████████████████████████████████████████████████████████████████████████Training:  98%|███████████████Training:  99%|███████████████████████████████████████████████████████████████████████████████████Training:  99%|███████████████████████████████████████████████████████████████████████████████████Training:  99%|███████████████████████████████████████████████████████████████████████████████████Training:  99%|███████████████████████████████████████████████████████████████████████████████████Training:  99%|███████████████████████████████████████████████████████████████████████████████████Training:  99%|███████████████████████████████████████████████████████████████████████████████████Training:  99%|███████████████████████████████████████████████████████████████████████████████████Training:  99%|███████████████████████████████████████████████████████████████████████████████████Training: 100%|███████████████████████████████████████████████████████████████████████████████████Training: 100%|███████████████████████████████████████████████████████████████████████████████████Training: 100%|███████████████████████████████████████████████████████████████████████████████████Training: 100%|███████████████████████████████████████████████████████████████████████████████████Training: 100%|███████████████████████████████████████████████████████████████████████████████████Training: 100%|███████████████████████████████████████████████████████████████████████████████████Training: 100%|███████████████████████████████████████████████████████████████████████████████████                                                                                                  Validazione:   8%|███▌            Risultati Epoch 5:
  - Train Loss: 0.5063 | Val Loss: 0.4624 | Val mIoU: 0.3765
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.4587
    - Classe 2 (Rough): 0.3996
    - Classe 3 (Smooth): 0.0180
    - Classe 4 (Grass): 0.0070
    - Classe 5 (HighVeg): 0.6048
    - Classe 6 (LowVeg): 0.2803
    - Classe 7 (Puddle): 0.7580
    - Classe 8 (Obstacle): 0.8624

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3765 <<<

--- Epoch 6/30 ---
Risultati Epoch 6:                                                                                                                                                                
  - Train Loss: 0.4840 | Val Loss: 0.4442 | Val mIoU: 0.4477
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.4602
    - Classe 2 (Rough): 0.4828
    - Classe 3 (Smooth): 0.1219
    - Classe 4 (Grass): 0.4078
    - Classe 5 (HighVeg): 0.5897
    - Classe 6 (LowVeg): 0.3343
    - Classe 7 (Puddle): 0.7667
    - Classe 8 (Obstacle): 0.8664

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4477 <<<

--- Epoch 7/30 ---
Risultati Epoch 7:                                                                                                                                                                
  - Train Loss: 0.4633 | Val Loss: 0.4402 | Val mIoU: 0.4716
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.4449
    - Classe 2 (Rough): 0.5370
    - Classe 3 (Smooth): 0.3275
    - Classe 4 (Grass): 0.4317
    - Classe 5 (HighVeg): 0.5632
    - Classe 6 (LowVeg): 0.3132
    - Classe 7 (Puddle): 0.7515
    - Classe 8 (Obstacle): 0.8758

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4716 <<<

--- Epoch 8/30 ---
Risultati Epoch 8:                                                                                                                                                                
  - Train Loss: 0.4357 | Val Loss: 0.4437 | Val mIoU: 0.5037
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.3866
    - Classe 2 (Rough): 0.5410
    - Classe 3 (Smooth): 0.3888
    - Classe 4 (Grass): 0.5666
    - Classe 5 (HighVeg): 0.6378
    - Classe 6 (LowVeg): 0.3449
    - Classe 7 (Puddle): 0.7902
    - Classe 8 (Obstacle): 0.8773

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5037 <<<

--- Epoch 9/30 ---
Risultati Epoch 9:                                                                                                                                                                
  - Train Loss: 0.4285 | Val Loss: 0.4227 | Val mIoU: 0.5110
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.4406
    - Classe 2 (Rough): 0.5207
    - Classe 3 (Smooth): 0.3751
    - Classe 4 (Grass): 0.6408
    - Classe 5 (HighVeg): 0.6136
    - Classe 6 (LowVeg): 0.3394
    - Classe 7 (Puddle): 0.7931
    - Classe 8 (Obstacle): 0.8755

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5110 <<<

--- Epoch 10/30 ---
Risultati Epoch 10:                                                                                                                                                               
  - Train Loss: 0.4083 | Val Loss: 0.4159 | Val mIoU: 0.5247
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.4261
    - Classe 2 (Rough): 0.5410
    - Classe 3 (Smooth): 0.4593
    - Classe 4 (Grass): 0.5908
    - Classe 5 (HighVeg): 0.6825
    - Classe 6 (LowVeg): 0.3488
    - Classe 7 (Puddle): 0.7953
    - Classe 8 (Obstacle): 0.8782

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5247 <<<

--- Epoch 11/30 ---
Risultati Epoch 11:                                                                                                                                                               
  - Train Loss: 0.4000 | Val Loss: 0.4128 | Val mIoU: 0.5125
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.3954
    - Classe 2 (Rough): 0.5508
    - Classe 3 (Smooth): 0.4795
    - Classe 4 (Grass): 0.5730
    - Classe 5 (HighVeg): 0.6227
    - Classe 6 (LowVeg): 0.3152
    - Classe 7 (Puddle): 0.7915
    - Classe 8 (Obstacle): 0.8846

--- Epoch 12/30 ---
Risultati Epoch 12:                                                                                                                                                               
  - Train Loss: 0.3856 | Val Loss: 0.4127 | Val mIoU: 0.5141
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.2144
    - Classe 2 (Rough): 0.5791
    - Classe 3 (Smooth): 0.4801
    - Classe 4 (Grass): 0.6917
    - Classe 5 (HighVeg): 0.6925
    - Classe 6 (LowVeg): 0.3045
    - Classe 7 (Puddle): 0.7851
    - Classe 8 (Obstacle): 0.8794

--- Epoch 13/30 ---
Risultati Epoch 13:                                                                                                                                                               
  - Train Loss: 0.3774 | Val Loss: 0.4185 | Val mIoU: 0.5200
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.4911
    - Classe 2 (Rough): 0.5325
    - Classe 3 (Smooth): 0.4874
    - Classe 4 (Grass): 0.5700
    - Classe 5 (HighVeg): 0.6143
    - Classe 6 (LowVeg): 0.3201
    - Classe 7 (Puddle): 0.7876
    - Classe 8 (Obstacle): 0.8769

--- Epoch 14/30 ---
Risultati Epoch 14:                                                                                                                                                               
  - Train Loss: 0.3711 | Val Loss: 0.4061 | Val mIoU: 0.5448
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.4841
    - Classe 2 (Rough): 0.5870
    - Classe 3 (Smooth): 0.5009
    - Classe 4 (Grass): 0.6941
    - Classe 5 (HighVeg): 0.6344
    - Classe 6 (LowVeg): 0.3187
    - Classe 7 (Puddle): 0.7980
    - Classe 8 (Obstacle): 0.8857

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5448 <<<

--- Epoch 15/30 ---
Risultati Epoch 15:                                                                                                                                                               
  - Train Loss: 0.3654 | Val Loss: 0.4168 | Val mIoU: 0.5438
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.5103
    - Classe 2 (Rough): 0.5560
    - Classe 3 (Smooth): 0.4335
    - Classe 4 (Grass): 0.7078
    - Classe 5 (HighVeg): 0.6973
    - Classe 6 (LowVeg): 0.3186
    - Classe 7 (Puddle): 0.7870
    - Classe 8 (Obstacle): 0.8834

--- Epoch 16/30 ---
Risultati Epoch 16:                                                                                                                                                               
  - Train Loss: 0.3505 | Val Loss: 0.4275 | Val mIoU: 0.5619
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.5605
    - Classe 2 (Rough): 0.6100
    - Classe 3 (Smooth): 0.4961
    - Classe 4 (Grass): 0.7336
    - Classe 5 (HighVeg): 0.6851
    - Classe 6 (LowVeg): 0.2860
    - Classe 7 (Puddle): 0.8036
    - Classe 8 (Obstacle): 0.8826

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5619 <<<

--- Epoch 17/30 ---
Risultati Epoch 17:                                                                                                                                                               
  - Train Loss: 0.3519 | Val Loss: 0.3963 | Val mIoU: 0.5311
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.2590
    - Classe 2 (Rough): 0.5576
    - Classe 3 (Smooth): 0.5505
    - Classe 4 (Grass): 0.7550
    - Classe 5 (HighVeg): 0.6852
    - Classe 6 (LowVeg): 0.3062
    - Classe 7 (Puddle): 0.7799
    - Classe 8 (Obstacle): 0.8860

--- Epoch 18/30 ---
Risultati Epoch 18:                                                                                                                                                               
  - Train Loss: 0.3396 | Val Loss: 0.4247 | Val mIoU: 0.5155
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.3018
    - Classe 2 (Rough): 0.5732
    - Classe 3 (Smooth): 0.4949
    - Classe 4 (Grass): 0.7290
    - Classe 5 (HighVeg): 0.5628
    - Classe 6 (LowVeg): 0.3025
    - Classe 7 (Puddle): 0.7888
    - Classe 8 (Obstacle): 0.8867

--- Epoch 19/30 ---
Risultati Epoch 19:                                                                                                                                                               
  - Train Loss: 0.3467 | Val Loss: 0.4024 | Val mIoU: 0.5522
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.4194
    - Classe 2 (Rough): 0.5939
    - Classe 3 (Smooth): 0.5640
    - Classe 4 (Grass): 0.7667
    - Classe 5 (HighVeg): 0.6877
    - Classe 6 (LowVeg): 0.2723
    - Classe 7 (Puddle): 0.7777
    - Classe 8 (Obstacle): 0.8881

--- Epoch 20/30 ---
Risultati Epoch 20:                                                                                                                                                               
  - Train Loss: 0.3314 | Val Loss: 0.4129 | Val mIoU: 0.5467
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.5530
    - Classe 2 (Rough): 0.5818
    - Classe 3 (Smooth): 0.5551
    - Classe 4 (Grass): 0.7332
    - Classe 5 (HighVeg): 0.6288
    - Classe 6 (LowVeg): 0.1942
    - Classe 7 (Puddle): 0.7893
    - Classe 8 (Obstacle): 0.8847

--- Epoch 21/30 ---
Risultati Epoch 21:                                                                                                                                                               
  - Train Loss: 0.3114 | Val Loss: 0.3845 | Val mIoU: 0.5781
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.5505
    - Classe 2 (Rough): 0.6019
    - Classe 3 (Smooth): 0.5908
    - Classe 4 (Grass): 0.7812
    - Classe 5 (HighVeg): 0.7256
    - Classe 6 (LowVeg): 0.2817
    - Classe 7 (Puddle): 0.7873
    - Classe 8 (Obstacle): 0.8839

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5781 <<<

--- Epoch 22/30 ---
Risultati Epoch 22:                                                                                                                                                               
  - Train Loss: 0.3003 | Val Loss: 0.3792 | Val mIoU: 0.5838
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.5946
    - Classe 2 (Rough): 0.6090
    - Classe 3 (Smooth): 0.5923
    - Classe 4 (Grass): 0.7815
    - Classe 5 (HighVeg): 0.7058
    - Classe 6 (LowVeg): 0.2851
    - Classe 7 (Puddle): 0.7942
    - Classe 8 (Obstacle): 0.8915

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5838 <<<

--- Epoch 23/30 ---
Validazione:   0%|                                                                                                                                       | 0/169 [00:00<?, ?it/s]C:\Users\leona\AppData\Roaming\Python\Python312\site-packages\albumentations\check_version.py:147: UserWarning: Error fetching version info <urlopen error _ssl.c:983: The handshake operation timed out>
  data = fetch_version_info()
Risultati Epoch 23:
  - Train Loss: 0.2959 | Val Loss: 0.3770 | Val mIoU: 0.5828
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.5616
    - Classe 2 (Rough): 0.6132
    - Classe 3 (Smooth): 0.5985
    - Classe 4 (Grass): 0.7890
    - Classe 5 (HighVeg): 0.6970
    - Classe 6 (LowVeg): 0.2986
    - Classe 7 (Puddle): 0.7952
    - Classe 8 (Obstacle): 0.8923

--- Epoch 24/30 ---
Risultati Epoch 24:                                                                                                                                                               
  - Train Loss: 0.2949 | Val Loss: 0.3796 | Val mIoU: 0.5877
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.5612
    - Classe 2 (Rough): 0.6114
    - Classe 3 (Smooth): 0.6026
    - Classe 4 (Grass): 0.7823
    - Classe 5 (HighVeg): 0.7475
    - Classe 6 (LowVeg): 0.3044
    - Classe 7 (Puddle): 0.7906
    - Classe 8 (Obstacle): 0.8893

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5877 <<<

--- Epoch 25/30 ---
Risultati Epoch 25:                                                                                                                                                               
  - Train Loss: 0.2885 | Val Loss: 0.3887 | Val mIoU: 0.5846
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.5626
    - Classe 2 (Rough): 0.6161
    - Classe 3 (Smooth): 0.5997
    - Classe 4 (Grass): 0.7895
    - Classe 5 (HighVeg): 0.7155
    - Classe 6 (LowVeg): 0.2970
    - Classe 7 (Puddle): 0.7913
    - Classe 8 (Obstacle): 0.8894

--- Epoch 26/30 ---
Risultati Epoch 26:                                                                                                                                                               
  - Train Loss: 0.2888 | Val Loss: 0.3778 | Val mIoU: 0.5862
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.5814
    - Classe 2 (Rough): 0.6128
    - Classe 3 (Smooth): 0.6047
    - Classe 4 (Grass): 0.7839
    - Classe 5 (HighVeg): 0.7030
    - Classe 6 (LowVeg): 0.3147
    - Classe 7 (Puddle): 0.7879
    - Classe 8 (Obstacle): 0.8876

--- Epoch 27/30 ---
Risultati Epoch 27:                                                                                                                                                               
  - Train Loss: 0.2855 | Val Loss: 0.3712 | Val mIoU: 0.5923
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.6254
    - Classe 2 (Rough): 0.6008
    - Classe 3 (Smooth): 0.5943
    - Classe 4 (Grass): 0.7853
    - Classe 5 (HighVeg): 0.7303
    - Classe 6 (LowVeg): 0.3106
    - Classe 7 (Puddle): 0.7944
    - Classe 8 (Obstacle): 0.8897

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5923 <<<

--- Epoch 28/30 ---
Risultati Epoch 28:                                                                                                                                                               
  - Train Loss: 0.2819 | Val Loss: 0.3794 | Val mIoU: 0.5858
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.6083
    - Classe 2 (Rough): 0.5849
    - Classe 3 (Smooth): 0.6062
    - Classe 4 (Grass): 0.7958
    - Classe 5 (HighVeg): 0.7099
    - Classe 6 (LowVeg): 0.2939
    - Classe 7 (Puddle): 0.7849
    - Classe 8 (Obstacle): 0.8884

--- Epoch 29/30 ---
Risultati Epoch 29:                                                                                                                                                               
  - Train Loss: 0.2820 | Val Loss: 0.3823 | Val mIoU: 0.5850
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.5706
    - Classe 2 (Rough): 0.5976
    - Classe 3 (Smooth): 0.6074
    - Classe 4 (Grass): 0.7899
    - Classe 5 (HighVeg): 0.7104
    - Classe 6 (LowVeg): 0.3142
    - Classe 7 (Puddle): 0.7862
    - Classe 8 (Obstacle): 0.8886

--- Epoch 30/30 ---
Risultati Epoch 30:                                                                                                                                                               
  - Train Loss: 0.2812 | Val Loss: 0.3745 | Val mIoU: 0.5975
  - IoU per Classe (Validation):
    - Classe 1 (Sky): 0.6305
    - Classe 2 (Rough): 0.6128
    - Classe 3 (Smooth): 0.6061
    - Classe 4 (Grass): 0.8000
    - Classe 5 (HighVeg): 0.7303
    - Classe 6 (LowVeg): 0.3151
    - Classe 7 (Puddle): 0.7918
    - Classe 8 (Obstacle): 0.8912

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5975 <<<

--- Training Completato ---
Miglior mIoU di validazione ottenuto: 0.5975
Visualizzazione Matrice di Confusione...
d:\MLPoject-Git\autoencoder-onboard-segmentation\local_execution\3 - Loss\6 - new_baseline_2.py:229: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown
  plt.title('Matrice di Confusione (Normalizzata per Riga)', fontsize=14); plt.show()

Analisi finale completata. Per eseguire le predizioni su nuovi dati, usare uno script separato caricando i pesi da 'best_model.pth' e implementando la funzione predict(X). 

'''