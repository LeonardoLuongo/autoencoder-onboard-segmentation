# SCRIPT DI TRAINING FINALE - EDIZIONE "ALL-IN" (2-STAGES, TVERSKY LOSS)

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

try:
    from torch.amp import GradScaler, autocast
    AMP_AVAILABLE = True
except ImportError:
    AMP_AVAILABLE = False
    class autocast:
        def __init__(self, enabled=False, **kwargs): self.enabled = enabled
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
from segmentation_models_pytorch.losses import TverskyLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# 2. RIPRODUCIBILITÀ
# =============================================================================
def seed_everything(seed: int):
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
        # --- Infrastruttura ---
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "RANDOM_SEED": 42,
        "NUM_WORKERS_TRAIN": 2,
        "NUM_WORKERS_VAL": 1,
        "USE_AMP": True,

        # --- Percorsi ---
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_model_b5_tversky_2stage.pth"),

        # --- Modello e Dati ---
        "ENCODER": "efficientnet-b5",
        "ENCODER_WEIGHTS": "imagenet",
        "OUTPUT_SIZE": (384, 768),
        "BATCH_SIZE": 2,  # Ridotto per encoder b5
        "NUM_CLASSES": 9,
        "IGNORE_INDEX": 0,
        "VALIDATION_SPLIT": 0.2,

        # --- Training a 2 Fasi ---
        "FREEZE_EPOCHS": 7,
        "UNFREEZE_EPOCHS": 23, # Totale 30 epoche
        "LR_HEAD": 1e-3,      # LR alto per il decoder
        "LR_FULL": 1e-5,      # LR basso per fine-tuning totale
        "WEIGHT_DECAY": 1e-5,
    }
    
    config["CLASS_NAMES"] = ["Ignored", "Sky", "Rough", "Smooth", "Grass", "HighVeg", "LowVeg", "Puddle", "Obstacle"]
    return config

# =============================================================================
# 4. GESTIONE DATI (Ottimizzata e Robusta)
# =============================================================================
def get_grouped_stratified_split(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    # ... (Codice invariato, già ottimizzato e corretto) ...
    paths = {"original": config["DATASET_PATH_TRAIN"], "augmented": config["DATASET_PATH_AUG"], "aggressive": config["DATASET_PATH_AUG_AGGR"]}
    all_dirs = []
    tqdm.write("Caricamento percorsi dei dataset..."); 
    for key, path in paths.items():
        if not os.path.isdir(path):
            tqdm.write(f"Attenzione: il percorso '{path}' non esiste, verrà saltato."); continue
        dirs = [d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)]; tqdm.write(f"Trovati {len(dirs)} campioni in '{key}'."); all_dirs.extend(dirs)
    if not all_dirs: raise FileNotFoundError("Nessuna directory di dati trovata.")
    all_dirs = sorted(list(set(all_dirs)))
    hashes = {path: imagehash.phash(Image.open(os.path.join(path, 'rgb.jpg'))) for path in tqdm(all_dirs, desc="Calcolo Hash")}
    tqdm.write("Creazione dei bucket per l'ottimizzazione del confronto...")
    buckets = defaultdict(list)
    for path, h in hashes.items(): buckets[int(str(h)[:2], 16)].append((path, h))
    G = nx.Graph(); G.add_nodes_from(hashes.keys())
    tqdm.write("Confronto delle immagini all'interno dei bucket (ottimizzato)...")
    for bucket_key, items_in_bucket in tqdm(buckets.items(), desc="Analisi Bucket"):
        if len(items_in_bucket) < 2: continue
        for i in range(len(items_in_bucket)):
            for j in range(i + 1, len(items_in_bucket)):
                path1, hash1 = items_in_bucket[i]; path2, hash2 = items_in_bucket[j]
                if hash1 - hash2 < 5: G.add_edge(path1, path2)
    image_groups = list(nx.connected_components(G)); tqdm.write(f"Trovati {len(image_groups)} gruppi di immagini uniche.")
    group_labels = []
    for group in tqdm(image_groups, desc="Etichettatura gruppi per lo split"):
        group_labels.append(1 if any(rare_id in np.unique(np.array(Image.open(os.path.join(p, 'labels.png')))) for p in group for rare_id in config.get("RARE_CLASS_INDICES", [])) else 0)
    train_groups, val_groups = train_test_split(image_groups, test_size=config["VALIDATION_SPLIT"], random_state=config["RANDOM_SEED"], stratify=group_labels)
    train_dirs, val_dirs = [p for group in train_groups for p in group], [p for group in val_groups for p in group]
    tqdm.write(f"Split stratificato creato: {len(train_dirs)} train, {len(val_dirs)} validation."); return train_dirs, val_dirs

def get_transforms(output_size: Tuple[int, int]) -> A.Compose:
    return A.Compose([
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
        assert mask.dtype == np.uint8, f"Tipo maschera non corretto: {mask.dtype}"
        assert mask.max() < self.config["NUM_CLASSES"], f"ID classe non valido in {os.path.join(sample_dir, 'labels.png')}"
        augmented = self.transform(image=image, mask=mask); return augmented['image'], augmented['mask']

# =============================================================================
# 5. MODELLO, LOSS E METRICHE
# =============================================================================
def get_model(config: Dict[str, Any]) -> nn.Module:
    # ... (invariato) ...
    model = smp.Unet(encoder_name=config["ENCODER"], encoder_weights=config["ENCODER_WEIGHTS"], in_channels=3, classes=config["NUM_CLASSES"])
    tqdm.write(f"Creazione del modello: U-Net con encoder '{config['ENCODER']}'"); tqdm.write(f"Parametri addestrabili: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}"); return model

def plot_confusion_matrix(conf_matrix_np, class_names, ignore_index, save_path="confusion_matrix_best.png"):
    tqdm.write("Visualizzazione e salvataggio Matrice di Confusione...")
    plot_matrix, plot_names = conf_matrix_np.copy(), class_names.copy()
    if ignore_index is not None:
        plot_matrix = np.delete(np.delete(plot_matrix, ignore_index, axis=0), ignore_index, axis=1)
        plot_names = [name for i, name in enumerate(class_names) if i != ignore_index]
    row_sums = plot_matrix.sum(axis=1)[:, np.newaxis]
    norm_conf_matrix = np.divide(plot_matrix, row_sums, out=np.zeros_like(plot_matrix, dtype=float), where=row_sums!=0)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(norm_conf_matrix, annot=True, fmt='.2%', cmap='Blues', xticklabels=plot_names, yticklabels=plot_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12); ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Matrice di Confusione (Normalizzata per Riga)', fontsize=14)
    # FIX: Salva su file invece di mostrare
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    tqdm.write(f"Matrice di confusione salvata in '{save_path}'")

# =============================================================================
# 6. CICLI DI TRAINING E VALIDAZIONE
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    # ... (invariato) ...
    model.train(); running_loss = 0.0; progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device, dtype=torch.long)
        with autocast(enabled=use_amp, device_type=device.type if use_amp else 'cpu'):
            outputs = model(images); loss = criterion(outputs, masks)
        optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        running_loss += loss.item(); progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device, config) -> Tuple[float, float, np.ndarray, np.ndarray]:
    # ... (invariato) ...
    model.eval(); total_loss = 0.0; conf_matrix = torch.zeros(config["NUM_CLASSES"], config["NUM_CLASSES"], dtype=torch.int64, device=device)
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            with autocast(enabled=True, device_type=device.type if device.type=='cuda' else 'cpu'):
                outputs = model(images); loss = criterion(outputs, masks)
            total_loss += loss.item(); preds = torch.argmax(outputs, dim=1)
            masks_flat, preds_flat = masks.flatten(), preds.flatten(); valid = masks_flat != config["IGNORE_INDEX"]
            indices = masks_flat[valid] * config["NUM_CLASSES"] + preds_flat[valid]; conf_matrix += torch.bincount(indices, minlength=config["NUM_CLASSES"]**2).reshape(config["NUM_CLASSES"], config["NUM_CLASSES"])
    tp = torch.diag(conf_matrix); fp = conf_matrix.sum(axis=0) - tp; fn = conf_matrix.sum(axis=1) - tp
    iou_per_class = (tp / (tp + fp + fn + 1e-6)).cpu().numpy(); mean_iou = np.nanmean(iou_per_class[1:])
    return total_loss / len(loader), mean_iou, iou_per_class, conf_matrix.cpu().numpy()

# =============================================================================
# 7. FUNZIONE PRINCIPALE (CON TRAINING A 2 FASI)
# =============================================================================
def main():
    config = get_config(); seed_everything(config["RANDOM_SEED"]); device = config["DEVICE"]
    use_amp = config["USE_AMP"] and AMP_AVAILABLE and device.type == 'cuda'
    tqdm.write(f"Device: {device} | Mixed Precision: {use_amp}"); tqdm.write(f"Strategia All-In: {config['ENCODER']} con TverskyLoss e training a 2 fasi")

    train_dirs, val_dirs = get_grouped_stratified_split(config)
    transforms = get_transforms(config["OUTPUT_SIZE"]); train_dataset = SegmentationDataset(train_dirs, transforms, config); val_dataset = SegmentationDataset(val_dirs, transforms, config)
    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS_TRAIN"], pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS_VAL"], pin_memory=pin_mem)

    model = get_model(config).to(device)
    scaler = GradScaler(enabled=use_amp)
    # STRATEGIA 1: Loss Tversky per penalizzare i Falsi Negativi
    criterion = TverskyLoss(mode='multiclass', alpha=0.7, beta=0.3, from_logits=True).to(device)

    best_iou = -1.0; best_conf_matrix = None; global_epoch = 0

    # --- FASE 1: FREEZE ENCODER, TRAIN DECODER ---
    tqdm.write("\n--- INIZIO FASE 1: TRAINING DEL DECODER (ENCODER CONGELATO) ---")
    for param in model.encoder.parameters(): param.requires_grad = False
    head_params = list(model.decoder.parameters()) + list(model.segmentation_head.parameters())
    optimizer = optim.Adam(head_params, lr=config["LR_HEAD"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    for epoch in range(config["FREEZE_EPOCHS"]):
        global_epoch += 1; tqdm.write(f"\n--- Fase 1 / Epoch {epoch+1}/{config['FREEZE_EPOCHS']} (Global: {global_epoch}) ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate(model, val_loader, criterion, device, config)
        scheduler.step(val_miou)
        tqdm.write(f"Risultati Epoch {global_epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        if val_miou > best_iou:
            best_iou, best_conf_matrix = val_miou, val_conf_matrix
            torch.save({'model_state_dict': model.state_dict(), 'config': config}, config["MODEL_SAVE_PATH"])
            tqdm.write(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<")

    # --- FASE 2: UNFREEZE ALL, FINE-TUNE ---
    tqdm.write("\n--- INIZIO FASE 2: FINE-TUNING DELL'INTERA RETE (ENCODER SCONGELATO) ---")
    for param in model.encoder.parameters(): param.requires_grad = True
    optimizer = optim.Adam(model.parameters(), lr=config["LR_FULL"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, cooldown=1, min_lr=1e-7)
    
    for epoch in range(config["UNFREEZE_EPOCHS"]):
        global_epoch += 1; tqdm.write(f"\n--- Fase 2 / Epoch {epoch+1}/{config['UNFREEZE_EPOCHS']} (Global: {global_epoch}) ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate(model, val_loader, criterion, device, config)
        scheduler.step(val_miou)
        tqdm.write(f"Risultati Epoch {global_epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        for i, iou in enumerate(val_iou_per_class):
            if i != config["IGNORE_INDEX"]: tqdm.write(f"    - Classe {i} ({config['CLASS_NAMES'][i]}): {iou:.4f}")
        if val_miou > best_iou:
            best_iou, best_conf_matrix = val_miou, val_conf_matrix
            torch.save({'model_state_dict': model.state_dict(), 'config': config}, config["MODEL_SAVE_PATH"])
            tqdm.write(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<")

    tqdm.write("\n--- Training Completato ---")
    tqdm.write(f"Miglior mIoU di validazione ottenuto: {best_iou:.4f}")
    if best_conf_matrix is not None: plot_confusion_matrix(best_conf_matrix, config["CLASS_NAMES"], config["IGNORE_INDEX"])

# =============================================================================
# 8. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()

'''
> & "C:/Program Files/Python312/python.exe" "d:/MLPoject-Git/autoencoder-onboard-segmentation/local_execution/3 - Loss/7 - All_in_1.py"
Device: cuda | Mixed Precision: True
Strategia All-In: efficientnet-b5 con TverskyLoss e training a 2 fasi
Caricamento percorsi dei dataset...
Trovati 931 campioni in 'original'.
Trovati 561 campioni in 'augmented'.
Trovati 1960 campioni in 'aggressive'.
Calcolo Hash: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3452/3452 [00:16<00:00, 215.25it/s]
Creazione dei bucket per l'ottimizzazione del confronto...
Confronto delle immagini all'interno dei bucket (ottimizzato)...
Analisi Bucket: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 86/86 [00:00<00:00, 179.00it/s] 
Trovati 2524 gruppi di immagini uniche.
Etichettatura gruppi per lo split: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2524/2524 [00:00<?, ?it/s] 
Split stratificato creato: 2799 train, 653 validation.
Creazione del modello: U-Net con encoder 'efficientnet-b5'
Parametri addestrabili: 31,217,161

--- INIZIO FASE 1: TRAINING DEL DECODER (ENCODER CONGELATO) ---

--- Fase 1 / Epoch 1/7 (Global: 1) ---
Risultati Epoch 1: Train Loss: 0.5170 | Val Loss: 0.4589 | Val mIoU: 0.2805                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2805 <<<

--- Fase 1 / Epoch 2/7 (Global: 2) ---
Risultati Epoch 2: Train Loss: 0.4821 | Val Loss: 0.4469 | Val mIoU: 0.3175                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3175 <<<

--- Fase 1 / Epoch 3/7 (Global: 3) ---
Risultati Epoch 3: Train Loss: 0.4712 | Val Loss: 0.4528 | Val mIoU: 0.3155                                                                                                       

--- Fase 1 / Epoch 4/7 (Global: 4) ---
Risultati Epoch 4: Train Loss: 0.4601 | Val Loss: 0.4374 | Val mIoU: 0.3329                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3329 <<<

--- Fase 1 / Epoch 5/7 (Global: 5) ---
Risultati Epoch 5: Train Loss: 0.4576 | Val Loss: 0.4380 | Val mIoU: 0.3468                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3468 <<<

--- Fase 1 / Epoch 6/7 (Global: 6) ---
Risultati Epoch 6: Train Loss: 0.4488 | Val Loss: 0.4256 | Val mIoU: 0.3420                                                                                                       

--- Fase 1 / Epoch 7/7 (Global: 7) ---
Risultati Epoch 7: Train Loss: 0.4429 | Val Loss: 0.4311 | Val mIoU: 0.3422                                                                                                       

--- INIZIO FASE 2: FINE-TUNING DELL'INTERA RETE (ENCODER SCONGELATO) ---

--- Fase 2 / Epoch 1/23 (Global: 8) ---
Risultati Epoch 8: Train Loss: 0.4270 | Val Loss: 0.4136 | Val mIoU: 0.3599                                                                                                       
    - Classe 1 (Sky): 0.1419
    - Classe 2 (Rough): 0.3276
    - Classe 3 (Smooth): 0.3588
    - Classe 4 (Grass): 0.1392
    - Classe 5 (HighVeg): 0.2378
    - Classe 6 (LowVeg): 0.3022
    - Classe 7 (Puddle): 0.5468
    - Classe 8 (Obstacle): 0.8250

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3599 <<<

--- Fase 2 / Epoch 2/23 (Global: 9) ---
Risultati Epoch 9: Train Loss: 0.4142 | Val Loss: 0.4080 | Val mIoU: 0.3518                                                                                                       
    - Classe 1 (Sky): 0.1206
    - Classe 2 (Rough): 0.3398
    - Classe 3 (Smooth): 0.3508
    - Classe 4 (Grass): 0.1372
    - Classe 5 (HighVeg): 0.2299
    - Classe 6 (LowVeg): 0.3013
    - Classe 7 (Puddle): 0.5375
    - Classe 8 (Obstacle): 0.7970

--- Fase 2 / Epoch 3/23 (Global: 10) ---
Risultati Epoch 10: Train Loss: 0.4082 | Val Loss: 0.4019 | Val mIoU: 0.3846                                                                                                      
    - Classe 1 (Sky): 0.1837
    - Classe 2 (Rough): 0.3317
    - Classe 3 (Smooth): 0.4042
    - Classe 4 (Grass): 0.1554
    - Classe 5 (HighVeg): 0.2589
    - Classe 6 (LowVeg): 0.3154
    - Classe 7 (Puddle): 0.6034
    - Classe 8 (Obstacle): 0.8242

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3846 <<<

--- Fase 2 / Epoch 4/23 (Global: 11) ---
Risultati Epoch 11: Train Loss: 0.4052 | Val Loss: 0.3967 | Val mIoU: 0.3866                                                                                                      
    - Classe 1 (Sky): 0.1552
    - Classe 2 (Rough): 0.3565
    - Classe 3 (Smooth): 0.3928
    - Classe 4 (Grass): 0.1577
    - Classe 5 (HighVeg): 0.2746
    - Classe 6 (LowVeg): 0.3171
    - Classe 7 (Puddle): 0.6153
    - Classe 8 (Obstacle): 0.8239

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3866 <<<

--- Fase 2 / Epoch 5/23 (Global: 12) ---
Risultati Epoch 12: Train Loss: 0.3993 | Val Loss: 0.3954 | Val mIoU: 0.4011                                                                                                      
    - Classe 1 (Sky): 0.1850
    - Classe 2 (Rough): 0.3692
    - Classe 3 (Smooth): 0.4189
    - Classe 4 (Grass): 0.1690
    - Classe 5 (HighVeg): 0.2874
    - Classe 6 (LowVeg): 0.3160
    - Classe 7 (Puddle): 0.6372
    - Classe 8 (Obstacle): 0.8263

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4011 <<<

--- Fase 2 / Epoch 6/23 (Global: 13) ---
Risultati Epoch 13: Train Loss: 0.3942 | Val Loss: 0.3893 | Val mIoU: 0.3972                                                                                                      
    - Classe 1 (Sky): 0.1574
    - Classe 2 (Rough): 0.3565
    - Classe 3 (Smooth): 0.4180
    - Classe 4 (Grass): 0.1650
    - Classe 5 (HighVeg): 0.2792
    - Classe 6 (LowVeg): 0.3216
    - Classe 7 (Puddle): 0.6631
    - Classe 8 (Obstacle): 0.8167

--- Fase 2 / Epoch 7/23 (Global: 14) ---
Risultati Epoch 14: Train Loss: 0.3853 | Val Loss: 0.3890 | Val mIoU: 0.4026                                                                                                      
    - Classe 1 (Sky): 0.1585
    - Classe 2 (Rough): 0.3672
    - Classe 3 (Smooth): 0.4245
    - Classe 4 (Grass): 0.1749
    - Classe 5 (HighVeg): 0.2909
    - Classe 6 (LowVeg): 0.3225
    - Classe 7 (Puddle): 0.6602
    - Classe 8 (Obstacle): 0.8217

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4026 <<<

--- Fase 2 / Epoch 8/23 (Global: 15) ---
Risultati Epoch 15: Train Loss: 0.3809 | Val Loss: 0.3904 | Val mIoU: 0.4030                                                                                                      
    - Classe 1 (Sky): 0.1427
    - Classe 2 (Rough): 0.3543
    - Classe 3 (Smooth): 0.4351
    - Classe 4 (Grass): 0.1657
    - Classe 5 (HighVeg): 0.2875
    - Classe 6 (LowVeg): 0.3314
    - Classe 7 (Puddle): 0.6873
    - Classe 8 (Obstacle): 0.8203

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4030 <<<

--- Fase 2 / Epoch 9/23 (Global: 16) ---
Risultati Epoch 16: Train Loss: 0.3749 | Val Loss: 0.3849 | Val mIoU: 0.4108                                                                                                      
    - Classe 1 (Sky): 0.1673
    - Classe 2 (Rough): 0.3725
    - Classe 3 (Smooth): 0.4370
    - Classe 4 (Grass): 0.1817
    - Classe 5 (HighVeg): 0.2995
    - Classe 6 (LowVeg): 0.3280
    - Classe 7 (Puddle): 0.6793
    - Classe 8 (Obstacle): 0.8213

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4108 <<<

--- Fase 2 / Epoch 10/23 (Global: 17) ---
Risultati Epoch 17: Train Loss: 0.3762 | Val Loss: 0.3813 | Val mIoU: 0.4150                                                                                                      
    - Classe 1 (Sky): 0.1793
    - Classe 2 (Rough): 0.3909
    - Classe 3 (Smooth): 0.4434
    - Classe 4 (Grass): 0.1850
    - Classe 5 (HighVeg): 0.3029
    - Classe 6 (LowVeg): 0.3237
    - Classe 7 (Puddle): 0.6792
    - Classe 8 (Obstacle): 0.8159

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4150 <<<

--- Fase 2 / Epoch 11/23 (Global: 18) ---
Risultati Epoch 18: Train Loss: 0.3734 | Val Loss: 0.3851 | Val mIoU: 0.4069                                                                                                      
    - Classe 1 (Sky): 0.1403
    - Classe 2 (Rough): 0.3844
    - Classe 3 (Smooth): 0.4387
    - Classe 4 (Grass): 0.1743
    - Classe 5 (HighVeg): 0.2822
    - Classe 6 (LowVeg): 0.3278
    - Classe 7 (Puddle): 0.6855
    - Classe 8 (Obstacle): 0.8220

--- Fase 2 / Epoch 12/23 (Global: 19) ---
Risultati Epoch 19: Train Loss: 0.3669 | Val Loss: 0.3788 | Val mIoU: 0.4252                                                                                                      
    - Classe 1 (Sky): 0.1956
    - Classe 2 (Rough): 0.3985
    - Classe 3 (Smooth): 0.4457
    - Classe 4 (Grass): 0.1999
    - Classe 5 (HighVeg): 0.3176
    - Classe 6 (LowVeg): 0.3222
    - Classe 7 (Puddle): 0.6979
    - Classe 8 (Obstacle): 0.8241

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4252 <<<

--- Fase 2 / Epoch 13/23 (Global: 20) ---
Risultati Epoch 20: Train Loss: 0.3617 | Val Loss: 0.3800 | Val mIoU: 0.4193                                                                                                      
    - Classe 1 (Sky): 0.1624
    - Classe 2 (Rough): 0.3914
    - Classe 3 (Smooth): 0.4442
    - Classe 4 (Grass): 0.1935
    - Classe 5 (HighVeg): 0.3039
    - Classe 6 (LowVeg): 0.3310
    - Classe 7 (Puddle): 0.6932
    - Classe 8 (Obstacle): 0.8350

--- Fase 2 / Epoch 14/23 (Global: 21) ---
Risultati Epoch 21: Train Loss: 0.3615 | Val Loss: 0.3778 | Val mIoU: 0.4166                                                                                                      
    - Classe 1 (Sky): 0.1709
    - Classe 2 (Rough): 0.3737
    - Classe 3 (Smooth): 0.4494
    - Classe 4 (Grass): 0.1917
    - Classe 5 (HighVeg): 0.2894
    - Classe 6 (LowVeg): 0.3396
    - Classe 7 (Puddle): 0.6913
    - Classe 8 (Obstacle): 0.8272

--- Fase 2 / Epoch 15/23 (Global: 22) ---
Risultati Epoch 22: Train Loss: 0.3572 | Val Loss: 0.3736 | Val mIoU: 0.4227                                                                                                      
    - Classe 1 (Sky): 0.1886
    - Classe 2 (Rough): 0.3839
    - Classe 3 (Smooth): 0.4534
    - Classe 4 (Grass): 0.1989
    - Classe 5 (HighVeg): 0.3079
    - Classe 6 (LowVeg): 0.3290
    - Classe 7 (Puddle): 0.6885
    - Classe 8 (Obstacle): 0.8315

--- Fase 2 / Epoch 16/23 (Global: 23) ---
Risultati Epoch 23: Train Loss: 0.3540 | Val Loss: 0.3747 | Val mIoU: 0.4234                                                                                                      
    - Classe 1 (Sky): 0.1724
    - Classe 2 (Rough): 0.3723
    - Classe 3 (Smooth): 0.4589
    - Classe 4 (Grass): 0.1987
    - Classe 5 (HighVeg): 0.3117
    - Classe 6 (LowVeg): 0.3354
    - Classe 7 (Puddle): 0.7096
    - Classe 8 (Obstacle): 0.8285

--- Fase 2 / Epoch 17/23 (Global: 24) ---
Risultati Epoch 24: Train Loss: 0.3520 | Val Loss: 0.3729 | Val mIoU: 0.4245                                                                                                      
    - Classe 1 (Sky): 0.1763
    - Classe 2 (Rough): 0.3862
    - Classe 3 (Smooth): 0.4553
    - Classe 4 (Grass): 0.2095
    - Classe 5 (HighVeg): 0.3185
    - Classe 6 (LowVeg): 0.3374
    - Classe 7 (Puddle): 0.6786
    - Classe 8 (Obstacle): 0.8340

--- Fase 2 / Epoch 18/23 (Global: 25) ---
Risultati Epoch 25: Train Loss: 0.3487 | Val Loss: 0.3739 | Val mIoU: 0.4209                                                                                                      
    - Classe 1 (Sky): 0.1715
    - Classe 2 (Rough): 0.3541
    - Classe 3 (Smooth): 0.4513
    - Classe 4 (Grass): 0.2010
    - Classe 5 (HighVeg): 0.3056
    - Classe 6 (LowVeg): 0.3424
    - Classe 7 (Puddle): 0.7058
    - Classe 8 (Obstacle): 0.8354

--- Fase 2 / Epoch 19/23 (Global: 26) ---
Risultati Epoch 26: Train Loss: 0.3481 | Val Loss: 0.3741 | Val mIoU: 0.4153                                                                                                      
    - Classe 1 (Sky): 0.1482
    - Classe 2 (Rough): 0.3765
    - Classe 3 (Smooth): 0.4513
    - Classe 4 (Grass): 0.2018
    - Classe 5 (HighVeg): 0.2919
    - Classe 6 (LowVeg): 0.3414
    - Classe 7 (Puddle): 0.6821
    - Classe 8 (Obstacle): 0.8293

--- Fase 2 / Epoch 20/23 (Global: 27) ---
Risultati Epoch 27: Train Loss: 0.3443 | Val Loss: 0.3706 | Val mIoU: 0.4246                                                                                                      
    - Classe 1 (Sky): 0.1966
    - Classe 2 (Rough): 0.3683
    - Classe 3 (Smooth): 0.4494
    - Classe 4 (Grass): 0.2173
    - Classe 5 (HighVeg): 0.3003
    - Classe 6 (LowVeg): 0.3440
    - Classe 7 (Puddle): 0.6985
    - Classe 8 (Obstacle): 0.8221

--- Fase 2 / Epoch 21/23 (Global: 28) ---
Risultati Epoch 28: Train Loss: 0.3439 | Val Loss: 0.3728 | Val mIoU: 0.4188                                                                                                      
    - Classe 1 (Sky): 0.1488
    - Classe 2 (Rough): 0.3795
    - Classe 3 (Smooth): 0.4540
    - Classe 4 (Grass): 0.1995
    - Classe 5 (HighVeg): 0.2963
    - Classe 6 (LowVeg): 0.3457
    - Classe 7 (Puddle): 0.6983
    - Classe 8 (Obstacle): 0.8287

--- Fase 2 / Epoch 22/23 (Global: 29) ---
Risultati Epoch 29: Train Loss: 0.3425 | Val Loss: 0.3702 | Val mIoU: 0.4227                                                                                                      
    - Classe 1 (Sky): 0.1662
    - Classe 2 (Rough): 0.3671
    - Classe 3 (Smooth): 0.4543
    - Classe 4 (Grass): 0.2092
    - Classe 5 (HighVeg): 0.3057
    - Classe 6 (LowVeg): 0.3489
    - Classe 7 (Puddle): 0.7001
    - Classe 8 (Obstacle): 0.8299

--- Fase 2 / Epoch 23/23 (Global: 30) ---
Risultati Epoch 30: Train Loss: 0.3428 | Val Loss: 0.3670 | Val mIoU: 0.4307                                                                                                      
    - Classe 1 (Sky): 0.1978
    - Classe 2 (Rough): 0.3815
    - Classe 3 (Smooth): 0.4538
    - Classe 4 (Grass): 0.2272
    - Classe 5 (HighVeg): 0.3104
    - Classe 6 (LowVeg): 0.3487
    - Classe 7 (Puddle): 0.7032
    - Classe 8 (Obstacle): 0.8235

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4307 <<<

--- Training Completato ---
Miglior mIoU di validazione ottenuto: 0.4307
Visualizzazione e salvataggio Matrice di Confusione...
Matrice di confusione salvata in 'confusion_matrix_best.png'

'''