# SCRIPT DI TRAINING FINALE - EDIZIONE "MIRATA" (AUGMENTATION AGGRESSIVA)

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
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "RANDOM_SEED": 42, "NUM_WORKERS_TRAIN": 2, "NUM_WORKERS_VAL": 2, "USE_AMP": True,
        
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_model_b5_aggressive_aug.pth"),

        "ENCODER": "efficientnet-b5", "ENCODER_WEIGHTS": "imagenet",
        "OUTPUT_SIZE": (384, 768), "BATCH_SIZE": 2, 
        "NUM_CLASSES": 9, "IGNORE_INDEX": 0, "VALIDATION_SPLIT": 0.2,

        "FREEZE_EPOCHS": 7, "UNFREEZE_EPOCHS": 23,
        "LR_HEAD": 1e-3, "LR_FULL": 1e-5, "WEIGHT_DECAY": 1e-5,
    }
    
    # FIX: Nomi delle classi anonimi come da richiesta
    config["CLASS_NAMES"] = ["Ignored"] + [f"Classe {i}" for i in range(1, config["NUM_CLASSES"])]

    # --- CALCOLO CORRETTO DI MEDIA E DEVIAZIONE STANDARD ---
    tqdm.write("Preparazione per il calcolo delle statistiche del dataset...")
    
    # 1. Pipeline di trasformazione SENZA A.Normalize
    stats_transforms = A.Compose([
        A.Resize(height=config["OUTPUT_SIZE"][0], width=config["OUTPUT_SIZE"][1]),
        # ToTensorV2 è fondamentale perché scala le immagini da [0, 255] a [0, 1]
        A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ToTensorV2(),
    ])

    # 2. Crea un Dataset temporaneo usando solo i dati originali
    train_original_dirs = [d for d in glob.glob(os.path.join(config["DATASET_PATH_TRAIN"], "*")) if os.path.isdir(d)]
    stats_dataset = SegmentationDataset(
        sample_dirs=train_original_dirs,
        transform=stats_transforms,
        config=config  # Passa la config per l'assert nel dataset
    )

    # 3. Avvolgi il Dataset in un DataLoader (IL FIX FONDAMENTALE)
    stats_loader = DataLoader(
        stats_dataset, 
        batch_size=config["BATCH_SIZE"] * 2, # Possiamo usare un batch size più grande qui
        shuffle=False, 
        num_workers=config["NUM_WORKERS_TRAIN"]
    )
    
    # 4. Ora chiama la funzione con il DataLoader
    mean, std = calculate_mean_std(stats_loader)

    assert 0.0 <= mean.min() and mean.max() <= 1.0,  "Mean fuori range 0-1"
    assert 0.0 <  std.min()  and std.max()  <  1.0,  "Std anomala"
    
    # 5. Salva i valori calcolati nella configurazione
    config["MEAN"] = mean
    config["STD"] = std

    tqdm.write("Statistiche personalizzate calcolate e caricate nella configurazione.")
    return config

# =============================================================================
# 4. DATA AUGMENTATION & GESTIONE DATI
# =============================================================================

def calculate_mean_std(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calcola la media e la deviazione standard per canale su un intero dataset.
    È numericamente stabile.
    """
    tqdm.write("Calcolo di media e deviazione standard in corso...")
    
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    pixel_count = 0

    for images, _ in tqdm(loader, desc="Analisi Statistiche Dataset"):
        batch_size, num_channels, height, width = images.shape
        psum += images.sum(axis=[0, 2, 3])
        psum_sq += (images ** 2).sum(axis=[0, 2, 3])
        pixel_count += batch_size * height * width

    total_mean = psum / pixel_count
    total_var = (psum_sq / pixel_count) - (total_mean ** 2)
    
    # --- FIX PER LA STABILITÀ NUMERICA ---
    # Assicura che la varianza non sia mai negativa a causa di errori di precisione.
    # Usiamo un minimo piccolo e positivo per evitare anche sqrt(0).
    total_std = torch.sqrt(torch.clamp(total_var, min=1e-6))

    mean_np = total_mean.numpy()
    std_np = total_std.numpy()

    tqdm.write("\n--- Statistiche Dataset Calcolate ---")
    tqdm.write(f"Media per canale (R, G, B): {mean_np}")
    tqdm.write(f"Deviazione Standard per canale (R, G, B): {std_np}")
    
    return mean_np, std_np

def get_train_transforms(config: Dict[str, Any]) -> A.Compose:
    """Pipeline di augmentation AGGRESSIVA per risolvere le confusioni specifiche."""
    
    output_size = config["OUTPUT_SIZE"]
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    return A.Compose([
        A.Resize(height=output_size[0], width=output_size[1], interpolation=cv2.INTER_NEAREST),
        A.HorizontalFlip(p=0.5),
        # Aggiunta per risolvere la confusione posizionale (es. Classe 1 vs 3)
        A.VerticalFlip(p=0.5), 
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.7, 
                           interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT),
        # Aggiunto per risolvere la confusione basata sul colore
        A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1, p=0.8), 
        # Aggiunto per forzare il modello a usare il contesto
        A.CoarseDropout(max_holes=16, max_height=32, max_width=32, min_holes=8, p=0.5), 
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

def get_val_transforms(output_size: Tuple[int, int], mean: float, std: float) -> A.Compose:
    """Pipeline di validazione (solo preprocessing)."""
    return A.Compose([
        A.Resize(height=output_size[0], width=output_size[1], interpolation=cv2.INTER_NEAREST),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

# La classe Dataset ora accetta una pipeline di transform specifica
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
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']

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
    # Per lo split stratificato, non ci basiamo più su nomi semantici ma su indici
    rare_class_indices = [2, 6] # Esempio di classi considerate "rare" per lo split
    group_labels = []
    for group in tqdm(image_groups, desc="Etichettatura gruppi per lo split"):
        group_labels.append(1 if any(rare_id in np.unique(np.array(Image.open(os.path.join(p, 'labels.png')))) for p in group for rare_id in rare_class_indices) else 0)
    train_groups, val_groups = train_test_split(image_groups, test_size=config["VALIDATION_SPLIT"], random_state=config["RANDOM_SEED"], stratify=group_labels)
    train_dirs, val_dirs = [p for group in train_groups for p in group], [p for group in val_groups for p in group]
    tqdm.write(f"Split stratificato creato: {len(train_dirs)} train, {len(val_dirs)} validation."); return train_dirs, val_dirs

# =============================================================================
# 5. MODELLO E FUNZIONI DI TRAINING/VALIDAZIONE (Logica invariata)
# =============================================================================
def get_model(config: Dict[str, Any]) -> nn.Module:
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    tqdm.write(f"Matrice di confusione salvata in '{save_path}'")

def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train(); running_loss = 0.0; progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        # Applica CutMix con una certa probabilità
        if torch.rand(1) < 0.5: # es. 50% di probabilità per batch
            images, masks = cutmix_batch(images, masks, alpha=1.0)

        images, masks = images.to(device), masks.to(device, dtype=torch.long)
        with autocast(enabled=use_amp, device_type=device.type if use_amp else 'cpu'):
            outputs = model(images); loss = criterion(outputs, masks)
        optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        running_loss += loss.item(); progress_bar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / len(loader)

def evaluate(model, loader, criterion, device, config) -> Tuple[float, float, np.ndarray, np.ndarray]:
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
    iou_per_class = (tp / (tp + fp + fn + 1e-6)).cpu().numpy()
    mean_iou = np.nanmean(iou_per_class[config["IGNORE_INDEX"]+1:]) if config["IGNORE_INDEX"] is not None else np.nanmean(iou_per_class)
    return total_loss / len(loader), mean_iou, iou_per_class, conf_matrix.cpu().numpy()


def cutmix_batch(images, masks, alpha=1.0):
    """
    Applica CutMix a un batch di immagini e maschere per la segmentazione.

    Args:
        images (torch.Tensor): Batch di immagini (B, C, H, W).
        masks (torch.Tensor): Batch di maschere (B, H, W).
        alpha (float): Parametro della distribuzione Beta per decidere la dimensione del ritaglio.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Il nuovo batch con CutMix applicato.
    """
    batch_size = images.size(0)
    
    # Genera gli indici per mescolare il batch
    # Es. [0, 1, 2, 3] -> [2, 0, 3, 1]
    indices = torch.randperm(batch_size).to(images.device)
    
    # Genera le coordinate del riquadro da tagliare e incollare
    # lambda decide la dimensione del riquadro
    lam = np.random.beta(alpha, alpha)
    
    # Calcola le coordinate (bounding box)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    
    # Prendi le immagini e maschere "sorgente" (da cui tagliare)
    images_shuffled = images[indices]
    masks_shuffled = masks[indices]
    
    # Crea una copia delle immagini e maschere originali
    new_images = images.clone()
    new_masks = masks.clone()
    
    # Incolla il riquadro
    new_images[:, :, bby1:bby2, bbx1:bbx2] = images_shuffled[:, :, bby1:bby2, bbx1:bbx2]
    new_masks[:, bby1:bby2, bbx1:bbx2] = masks_shuffled[:, bby1:bby2, bbx1:bbx2]

    return new_images, new_masks

def rand_bbox(size, lam):
    """Calcola le coordinate di un bounding box casuale."""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    # Coordinate casuali del centro
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
# =============================================================================
# 6. FUNZIONE PRINCIPALE (CON AUGMENTATION AGGRESSIVA)
# =============================================================================
def main():
    config = get_config(); seed_everything(config["RANDOM_SEED"]); device = config["DEVICE"]
    use_amp = config["USE_AMP"] and AMP_AVAILABLE and device.type == 'cuda'
    tqdm.write(f"Device: {device} | Mixed Precision: {use_amp}"); tqdm.write(f"Strategia Finale: {config['ENCODER']} con Augmentation Mirata")

    train_dirs, val_dirs = get_grouped_stratified_split(config)
    # Pipeline di augmentation diverse per training e validazione
    train_transforms = get_train_transforms(config)
    val_transforms = get_val_transforms(config["OUTPUT_SIZE"], config["MEAN"], config["STD"])
    
    train_dataset = SegmentationDataset(train_dirs, train_transforms, config)
    val_dataset = SegmentationDataset(val_dirs, val_transforms, config)
    
    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS_TRAIN"], pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS_VAL"], pin_memory=pin_mem)

    model = get_model(config).to(device)
    scaler = GradScaler(enabled=use_amp)
    criterion = TverskyLoss(mode='multiclass', alpha=0.7, beta=0.3, from_logits=True).to(device)

    best_iou = -1.0; best_conf_matrix = None; global_epoch = 0

    # --- FASE 1: FREEZE ENCODER, TRAIN DECODER ---
    tqdm.write("\n--- INIZIO FASE 1: TRAINING DEL DECODER (ENCODER CONGELATO) ---")
    for param in model.encoder.parameters(): param.requires_grad = False
    optimizer = optim.Adam(list(model.decoder.parameters()) + list(model.segmentation_head.parameters()), lr=config["LR_HEAD"], weight_decay=config["WEIGHT_DECAY"])
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
# 7. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()


'''
> & "C:/Program Files/Python312/python.exe" "d:/MLPoject-Git/autoencoder-onboard-segmentation/local_execution/Leo/7 - All_in_1.py"
Preparazione per il calcolo delle statistiche del dataset...
Calcolo di media e deviazione standard in corso...
Analisi Statistiche Dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 233/233 [00:23<00:00, 10.11it/s]

--- Statistiche Dataset Calcolate ---
Media per canale (R, G, B): [0.42948496 0.44826746 0.42454118]
Deviazione Standard per canale (R, G, B): [0.2645671  0.26363927 0.26595324]
Statistiche personalizzate calcolate e caricate nella configurazione.
Device: cuda | Mixed Precision: True
Strategia Finale: efficientnet-b5 con Augmentation Mirata
Caricamento percorsi dei dataset...
Trovati 931 campioni in 'original'.
Trovati 561 campioni in 'augmented'.
Trovati 1960 campioni in 'aggressive'.
Calcolo Hash: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3452/3452 [00:15<00:00, 226.08it/s] 
Creazione dei bucket per l'ottimizzazione del confronto...
Confronto delle immagini all'interno dei bucket (ottimizzato)...
Analisi Bucket: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 86/86 [00:00<00:00, 190.07it/s] 
Trovati 2524 gruppi di immagini uniche.
Etichettatura gruppi per lo split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 2524/2524 [00:21<00:00, 117.99it/s] 
Split stratificato creato: 2746 train, 706 validation.
C:\Users\leona\AppData\Roaming\Python\Python312\site-packages\albumentations\core\validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
d:\MLPoject-Git\autoencoder-onboard-segmentation\local_execution\Leo\7 - All_in_1.py:176: UserWarning: Argument(s) 'max_holes, max_height, max_width, min_holes' are not valid for transform CoarseDropout
  A.CoarseDropout(max_holes=16, max_height=32, max_width=32, min_holes=8, p=0.5),
Creazione del modello: U-Net con encoder 'efficientnet-b5'
Parametri addestrabili: 31,217,161

--- INIZIO FASE 1: TRAINING DEL DECODER (ENCODER CONGELATO) ---

--- Fase 1 / Epoch 1/7 (Global: 1) ---
Risultati Epoch 1: Train Loss: 0.5258 | Val Loss: 0.4565 | Val mIoU: 0.3001                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3001 <<<

--- Fase 1 / Epoch 2/7 (Global: 2) ---
Risultati Epoch 2: Train Loss: 0.4994 | Val Loss: 0.4523 | Val mIoU: 0.2993                                                                                                       

--- Fase 1 / Epoch 3/7 (Global: 3) ---
Risultati Epoch 3: Train Loss: 0.4881 | Val Loss: 0.4530 | Val mIoU: 0.3300                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3300 <<<

--- Fase 1 / Epoch 4/7 (Global: 4) ---
Risultati Epoch 4: Train Loss: 0.4921 | Val Loss: 0.4430 | Val mIoU: 0.3113                                                                                                       

--- Fase 1 / Epoch 5/7 (Global: 5) ---
Risultati Epoch 5: Train Loss: 0.4857 | Val Loss: 0.4397 | Val mIoU: 0.3263                                                                                                       

--- Fase 1 / Epoch 6/7 (Global: 6) ---
Risultati Epoch 6: Train Loss: 0.4766 | Val Loss: 0.4374 | Val mIoU: 0.3688                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3688 <<<

--- Fase 1 / Epoch 7/7 (Global: 7) ---
Risultati Epoch 7: Train Loss: 0.4797 | Val Loss: 0.4418 | Val mIoU: 0.3267                                                                                                       

--- INIZIO FASE 2: FINE-TUNING DELL'INTERA RETE (ENCODER SCONGELATO) ---

--- Fase 2 / Epoch 1/23 (Global: 8) ---
Risultati Epoch 8: Train Loss: 0.4717 | Val Loss: 0.4294 | Val mIoU: 0.3587                                                                                                       
    - Classe 1 (Classe 1): 0.1782
    - Classe 2 (Classe 2): 0.3107
    - Classe 3 (Classe 3): 0.4142
    - Classe 4 (Classe 4): 0.1495
    - Classe 5 (Classe 5): 0.2090
    - Classe 6 (Classe 6): 0.2493
    - Classe 7 (Classe 7): 0.5369
    - Classe 8 (Classe 8): 0.8222

--- Fase 2 / Epoch 2/23 (Global: 9) ---
Risultati Epoch 9: Train Loss: 0.4666 | Val Loss: 0.4248 | Val mIoU: 0.3714                                                                                                       
    - Classe 1 (Classe 1): 0.2146
    - Classe 2 (Classe 2): 0.3169
    - Classe 3 (Classe 3): 0.4139
    - Classe 4 (Classe 4): 0.1506
    - Classe 5 (Classe 5): 0.2248
    - Classe 6 (Classe 6): 0.2681
    - Classe 7 (Classe 7): 0.5537
    - Classe 8 (Classe 8): 0.8287

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3714 <<<

--- Fase 2 / Epoch 3/23 (Global: 10) ---
Risultati Epoch 10: Train Loss: 0.4555 | Val Loss: 0.4225 | Val mIoU: 0.3770                                                                                                      
    - Classe 1 (Classe 1): 0.2196
    - Classe 2 (Classe 2): 0.3178
    - Classe 3 (Classe 3): 0.4303
    - Classe 4 (Classe 4): 0.1405
    - Classe 5 (Classe 5): 0.2246
    - Classe 6 (Classe 6): 0.2700
    - Classe 7 (Classe 7): 0.5801
    - Classe 8 (Classe 8): 0.8329

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3770 <<<

--- Fase 2 / Epoch 4/23 (Global: 11) ---
Risultati Epoch 11: Train Loss: 0.4474 | Val Loss: 0.4164 | Val mIoU: 0.3801                                                                                                      
    - Classe 1 (Classe 1): 0.2293
    - Classe 2 (Classe 2): 0.3444
    - Classe 3 (Classe 3): 0.4230
    - Classe 4 (Classe 4): 0.1486
    - Classe 5 (Classe 5): 0.2381
    - Classe 6 (Classe 6): 0.2778
    - Classe 7 (Classe 7): 0.5459
    - Classe 8 (Classe 8): 0.8334

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3801 <<<

--- Fase 2 / Epoch 5/23 (Global: 12) ---
Risultati Epoch 12: Train Loss: 0.4451 | Val Loss: 0.4149 | Val mIoU: 0.3829                                                                                                      
    - Classe 1 (Classe 1): 0.2152
    - Classe 2 (Classe 2): 0.3304
    - Classe 3 (Classe 3): 0.4343
    - Classe 4 (Classe 4): 0.1579
    - Classe 5 (Classe 5): 0.2334
    - Classe 6 (Classe 6): 0.2756
    - Classe 7 (Classe 7): 0.5815
    - Classe 8 (Classe 8): 0.8351

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3829 <<<

--- Fase 2 / Epoch 6/23 (Global: 13) ---
Risultati Epoch 13: Train Loss: 0.4388 | Val Loss: 0.4092 | Val mIoU: 0.3920                                                                                                      
    - Classe 1 (Classe 1): 0.2425
    - Classe 2 (Classe 2): 0.3270
    - Classe 3 (Classe 3): 0.4348
    - Classe 4 (Classe 4): 0.1693
    - Classe 5 (Classe 5): 0.2426
    - Classe 6 (Classe 6): 0.2798
    - Classe 7 (Classe 7): 0.6148
    - Classe 8 (Classe 8): 0.8253

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3920 <<<

--- Fase 2 / Epoch 7/23 (Global: 14) ---
Risultati Epoch 14: Train Loss: 0.4426 | Val Loss: 0.4161 | Val mIoU: 0.3967                                                                                                      
    - Classe 1 (Classe 1): 0.2152
    - Classe 2 (Classe 2): 0.3596
    - Classe 3 (Classe 3): 0.4457
    - Classe 4 (Classe 4): 0.1618
    - Classe 5 (Classe 5): 0.2592
    - Classe 6 (Classe 6): 0.2673
    - Classe 7 (Classe 7): 0.6188
    - Classe 8 (Classe 8): 0.8458

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3967 <<<

--- Fase 2 / Epoch 8/23 (Global: 15) ---
Risultati Epoch 15: Train Loss: 0.4398 | Val Loss: 0.4070 | Val mIoU: 0.3940                                                                                                      
    - Classe 1 (Classe 1): 0.2113
    - Classe 2 (Classe 2): 0.3519
    - Classe 3 (Classe 3): 0.4458
    - Classe 4 (Classe 4): 0.1650
    - Classe 5 (Classe 5): 0.2532
    - Classe 6 (Classe 6): 0.2835
    - Classe 7 (Classe 7): 0.6126
    - Classe 8 (Classe 8): 0.8286

--- Fase 2 / Epoch 9/23 (Global: 16) ---
Risultati Epoch 16: Train Loss: 0.4317 | Val Loss: 0.4045 | Val mIoU: 0.3922                                                                                                      
    - Classe 1 (Classe 1): 0.1818
    - Classe 2 (Classe 2): 0.3521
    - Classe 3 (Classe 3): 0.4497
    - Classe 4 (Classe 4): 0.1577
    - Classe 5 (Classe 5): 0.2619
    - Classe 6 (Classe 6): 0.2848
    - Classe 7 (Classe 7): 0.6227
    - Classe 8 (Classe 8): 0.8268

--- Fase 2 / Epoch 10/23 (Global: 17) ---
Risultati Epoch 17: Train Loss: 0.4327 | Val Loss: 0.3973 | Val mIoU: 0.4020                                                                                                      
    - Classe 1 (Classe 1): 0.2414
    - Classe 2 (Classe 2): 0.3705
    - Classe 3 (Classe 3): 0.4310
    - Classe 4 (Classe 4): 0.1678
    - Classe 5 (Classe 5): 0.2689
    - Classe 6 (Classe 6): 0.2874
    - Classe 7 (Classe 7): 0.6164
    - Classe 8 (Classe 8): 0.8323

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4020 <<<

--- Fase 2 / Epoch 11/23 (Global: 18) ---
Risultati Epoch 18: Train Loss: 0.4250 | Val Loss: 0.4015 | Val mIoU: 0.4074                                                                                                      
    - Classe 1 (Classe 1): 0.2396
    - Classe 2 (Classe 2): 0.3829
    - Classe 3 (Classe 3): 0.4373
    - Classe 4 (Classe 4): 0.1833
    - Classe 5 (Classe 5): 0.2814
    - Classe 6 (Classe 6): 0.2761
    - Classe 7 (Classe 7): 0.6167
    - Classe 8 (Classe 8): 0.8417

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4074 <<<

--- Fase 2 / Epoch 12/23 (Global: 19) ---
Risultati Epoch 19: Train Loss: 0.4252 | Val Loss: 0.3995 | Val mIoU: 0.4071                                                                                                      
    - Classe 1 (Classe 1): 0.2251
    - Classe 2 (Classe 2): 0.3836
    - Classe 3 (Classe 3): 0.4452
    - Classe 4 (Classe 4): 0.1704
    - Classe 5 (Classe 5): 0.2795
    - Classe 6 (Classe 6): 0.2823
    - Classe 7 (Classe 7): 0.6282
    - Classe 8 (Classe 8): 0.8427

--- Fase 2 / Epoch 13/23 (Global: 20) ---
Risultati Epoch 20: Train Loss: 0.4230 | Val Loss: 0.3931 | Val mIoU: 0.4103                                                                                                      
    - Classe 1 (Classe 1): 0.2250
    - Classe 2 (Classe 2): 0.3907
    - Classe 3 (Classe 3): 0.4490
    - Classe 4 (Classe 4): 0.1828
    - Classe 5 (Classe 5): 0.2912
    - Classe 6 (Classe 6): 0.2862
    - Classe 7 (Classe 7): 0.6320
    - Classe 8 (Classe 8): 0.8259

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4103 <<<

--- Fase 2 / Epoch 14/23 (Global: 21) ---
Risultati Epoch 21: Train Loss: 0.4216 | Val Loss: 0.3916 | Val mIoU: 0.4114                                                                                                      
    - Classe 1 (Classe 1): 0.2061
    - Classe 2 (Classe 2): 0.3918
    - Classe 3 (Classe 3): 0.4486
    - Classe 4 (Classe 4): 0.1953
    - Classe 5 (Classe 5): 0.2840
    - Classe 6 (Classe 6): 0.2908
    - Classe 7 (Classe 7): 0.6362
    - Classe 8 (Classe 8): 0.8383

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4114 <<<

--- Fase 2 / Epoch 15/23 (Global: 22) ---
Risultati Epoch 22: Train Loss: 0.4208 | Val Loss: 0.3962 | Val mIoU: 0.4147                                                                                                      
    - Classe 1 (Classe 1): 0.2294
    - Classe 2 (Classe 2): 0.3973
    - Classe 3 (Classe 3): 0.4474
    - Classe 4 (Classe 4): 0.1924
    - Classe 5 (Classe 5): 0.2996
    - Classe 6 (Classe 6): 0.2739
    - Classe 7 (Classe 7): 0.6370
    - Classe 8 (Classe 8): 0.8408

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4147 <<<

--- Fase 2 / Epoch 16/23 (Global: 23) ---
Risultati Epoch 23: Train Loss: 0.4202 | Val Loss: 0.3901 | Val mIoU: 0.4225                                                                                                      
    - Classe 1 (Classe 1): 0.2805
    - Classe 2 (Classe 2): 0.3936
    - Classe 3 (Classe 3): 0.4468
    - Classe 4 (Classe 4): 0.1923
    - Classe 5 (Classe 5): 0.3041
    - Classe 6 (Classe 6): 0.2876
    - Classe 7 (Classe 7): 0.6345
    - Classe 8 (Classe 8): 0.8403

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4225 <<<

--- Fase 2 / Epoch 17/23 (Global: 24) ---
Risultati Epoch 24: Train Loss: 0.4173 | Val Loss: 0.3949 | Val mIoU: 0.4199                                                                                                      
    - Classe 1 (Classe 1): 0.2221
    - Classe 2 (Classe 2): 0.3976
    - Classe 3 (Classe 3): 0.4532
    - Classe 4 (Classe 4): 0.1912
    - Classe 5 (Classe 5): 0.3001
    - Classe 6 (Classe 6): 0.2865
    - Classe 7 (Classe 7): 0.6599
    - Classe 8 (Classe 8): 0.8490

--- Fase 2 / Epoch 18/23 (Global: 25) ---
Risultati Epoch 25: Train Loss: 0.4158 | Val Loss: 0.3877 | Val mIoU: 0.4190                                                                                                      
    - Classe 1 (Classe 1): 0.2183
    - Classe 2 (Classe 2): 0.3915
    - Classe 3 (Classe 3): 0.4526
    - Classe 4 (Classe 4): 0.2089
    - Classe 5 (Classe 5): 0.3101
    - Classe 6 (Classe 6): 0.2825
    - Classe 7 (Classe 7): 0.6410
    - Classe 8 (Classe 8): 0.8467

--- Fase 2 / Epoch 19/23 (Global: 26) ---
Risultati Epoch 26: Train Loss: 0.4157 | Val Loss: 0.3863 | Val mIoU: 0.4247                                                                                                      
    - Classe 1 (Classe 1): 0.2036
    - Classe 2 (Classe 2): 0.3878
    - Classe 3 (Classe 3): 0.4528
    - Classe 4 (Classe 4): 0.2168
    - Classe 5 (Classe 5): 0.3296
    - Classe 6 (Classe 6): 0.3072
    - Classe 7 (Classe 7): 0.6526
    - Classe 8 (Classe 8): 0.8470

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4247 <<<

--- Fase 2 / Epoch 20/23 (Global: 27) ---
Risultati Epoch 27: Train Loss: 0.4126 | Val Loss: 0.3837 | Val mIoU: 0.4192                                                                                                      
    - Classe 1 (Classe 1): 0.2040
    - Classe 2 (Classe 2): 0.3936
    - Classe 3 (Classe 3): 0.4531
    - Classe 4 (Classe 4): 0.2016
    - Classe 5 (Classe 5): 0.3028
    - Classe 6 (Classe 6): 0.3037
    - Classe 7 (Classe 7): 0.6460
    - Classe 8 (Classe 8): 0.8490

--- Fase 2 / Epoch 21/23 (Global: 28) ---
Risultati Epoch 28: Train Loss: 0.4134 | Val Loss: 0.3869 | Val mIoU: 0.4320                                                                                                      
    - Classe 1 (Classe 1): 0.2406
    - Classe 2 (Classe 2): 0.4035
    - Classe 3 (Classe 3): 0.4527
    - Classe 4 (Classe 4): 0.2276
    - Classe 5 (Classe 5): 0.3279
    - Classe 6 (Classe 6): 0.2965
    - Classe 7 (Classe 7): 0.6566
    - Classe 8 (Classe 8): 0.8502

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4320 <<<

--- Fase 2 / Epoch 22/23 (Global: 29) ---
Risultati Epoch 29: Train Loss: 0.4074 | Val Loss: 0.3785 | Val mIoU: 0.4331                                                                                                      
    - Classe 1 (Classe 1): 0.2576
    - Classe 2 (Classe 2): 0.4059
    - Classe 3 (Classe 3): 0.4503
    - Classe 4 (Classe 4): 0.2138
    - Classe 5 (Classe 5): 0.3481
    - Classe 6 (Classe 6): 0.3016
    - Classe 7 (Classe 7): 0.6459
    - Classe 8 (Classe 8): 0.8414

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4331 <<<

--- Fase 2 / Epoch 23/23 (Global: 30) ---
Risultati Epoch 30: Train Loss: 0.4094 | Val Loss: 0.3846 | Val mIoU: 0.4262                                                                                                      
    - Classe 1 (Classe 1): 0.1995
    - Classe 2 (Classe 2): 0.4102
    - Classe 3 (Classe 3): 0.4639
    - Classe 4 (Classe 4): 0.2386
    - Classe 5 (Classe 5): 0.3310
    - Classe 6 (Classe 6): 0.2717
    - Classe 7 (Classe 7): 0.6522
    - Classe 8 (Classe 8): 0.8424

--- Training Completato ---
Miglior mIoU di validazione ottenuto: 0.4331
Visualizzazione e salvataggio Matrice di Confusione...
Matrice di confusione salvata in 'confusion_matrix_best.png'

'''