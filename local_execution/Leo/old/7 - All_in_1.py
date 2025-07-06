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
from segmentation_models_pytorch.losses import FocalLoss

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
        "PERSISTENT_WORKERS": True, "RARE_CLASS_INDICES": [4], "RARE_BOOST_FACTOR": 20.0,
        
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_model_b5_aggressive_aug.pth"),

        "ENCODER": "efficientnet-b4", "ENCODER_WEIGHTS": "imagenet",
        "OUTPUT_SIZE": (384, 768), "BATCH_SIZE": 4, 
        "NUM_CLASSES": 9, "IGNORE_INDEX": 0, "VALIDATION_SPLIT": 0.2,

        "FREEZE_EPOCHS": 7, "UNFREEZE_EPOCHS": 23,
        "LR_HEAD": 1e-3, "LR_FULL": 1e-5, "WEIGHT_DECAY": 1e-6,

        "CONTRASTIVE_TEMP": 0.1,      # τ dell’InfoNCE
        "CONTRASTIVE_MAX_SAMPLES": 4096,  # pixel da campionare per batch
        "CONTRASTIVE_W": 0.2,         # peso della loss rispetto alle loss di segmento
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
        num_workers=config["NUM_WORKERS_TRAIN"],
        persistent_workers=config["PERSISTENT_WORKERS"]
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
    # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # image net
    mean, std = config["MEAN"], config["STD"]
    return A.Compose([
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=0.5),
        # A.RandomRotate90(p=0.7),
        # A.RandomBrightnessContrast(p=0.7),
        # A.RandomGamma(p=0.7),
        # A.ColorJitter(p=0.7),
        # A.GaussianBlur(blur_limit=(3, 5), p=0.4),
        # A.CoarseDropout(
        #     max_holes=16, 
        #     min_holes=8,
        #     max_height=32,
        #     min_height=16,
        #     max_width=32, 
        #     min_width=16,
        #     p=0.5
        # ),
        A.Resize(height=config["OUTPUT_SIZE"][0], width=config["OUTPUT_SIZE"][1]),
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
    # paths = {"original": config["DATASET_PATH_TRAIN"], "augmented": config["DATASET_PATH_AUG"], "aggressive": config["DATASET_PATH_AUG_AGGR"]}
    paths = {"original": config["DATASET_PATH_TRAIN"], "augmented": config["DATASET_PATH_AUG"]}
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

def calculate_class_weights(dataset: Dataset, config: Dict[str, Any]) -> torch.Tensor:
    """Calcola i pesi per la loss basandosi sulla frequenza delle classi."""
    print("Calcolo della distribuzione delle classi per i pesi della loss...")
    num_classes = config["NUM_CLASSES"]
    pixel_counts = np.zeros(num_classes, dtype=np.int64)
    
    loader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS_TRAIN"])
    for _, masks in tqdm(loader, desc="Analisi distribuzione pixel"):
        for mask in masks:
            pixel_counts += np.bincount(mask.flatten(), minlength=num_classes)
            
    # Calcolo dei pesi con Inverse Frequency + Boost per le classi rare
    class_frequencies = pixel_counts / pixel_counts.sum()
    weights = 1.0 / (class_frequencies + 1e-6)
    weights = weights / weights.mean() # Normalizza i pesi
    
    for idx in config["RARE_CLASS_INDICES"]:
        weights[idx] *= config["RARE_BOOST_FACTOR"]
        
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(config["DEVICE"])
    
    print("\nPesi calcolati per la loss (con boost per classi rare):")
    for i, w in enumerate(weights_tensor):
        print(f"  - Classe {i}: {w.item():.2f}")
    return weights_tensor


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

def train_one_epoch(model, loader, optimizer, seg_loss_fn, contrast_loss_fn, seg_w, scaler, device, use_amp):
    model.train(); running_loss = 0.0; progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        # Applica CutMix con una certa probabilità
        # if torch.rand(1) < 0.5: # es. 50% di probabilità per batch
        #     images, masks = cutmix_batch(images, masks, alpha=1.0)

        images, masks = images.to(device), masks.to(device, dtype=torch.long)
        with autocast(enabled=use_amp, device_type=device.type if use_amp else 'cpu'):
            outputs = model(images); 
            if contrast_loss_fn is not None:
                loss = seg_loss_fn(outputs, masks) + contrast_loss_fn(outputs, masks) * seg_w
            else:
                loss = seg_loss_fn(outputs, masks)
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
# 6. PixelContrastiveLoss
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelContrastiveLoss(nn.Module):
    """
    Supervised Pixel-wise InfoNCE.
    * Usa tutti i pixel di un batch fino a `max_samples`
    * Ignora i pixel con label == ignore_index
    """
    def __init__(self,
                 temperature: float = 0.1,
                 max_samples: int = 4096,
                 ignore_index: int = 0):
        super().__init__()
        self.temp = temperature
        self.max_samples = max_samples
        self.ignore_index = ignore_index

    def forward(self,
                feats: torch.Tensor,   # (B,C,H,W) – logit o feature
                labels: torch.Tensor   # (B,H,W)   – int64
                ) -> torch.Tensor:

        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(-1, C)      # (N,C)
        labels = labels.view(-1)                              # (N,)

        valid = labels != self.ignore_index
        feats, labels = feats[valid], labels[valid]

        # Campionamento per tenere sotto controllo la memoria GPU
        if self.max_samples is not None and feats.size(0) > self.max_samples:
            idx = torch.randperm(feats.size(0), device=feats.device)[:self.max_samples]
            feats, labels = feats[idx], labels[idx]

        feats = F.normalize(feats, dim=1)                     # unit-sphere
        logits = torch.mm(feats, feats.t()) / self.temp       # (N,N)

        # Mask per self-pair e per classe
        N = logits.size(0)
        self_mask = torch.eye(N, dtype=torch.bool, device=feats.device)
        class_mask = labels.unsqueeze(0) == labels.unsqueeze(1)   # (N,N)

        # Pos: stessa classe & non self-pair
        pos_mask = class_mask & (~self_mask)
        neg_mask = ~class_mask

        exp_logits = torch.exp(logits)
        pos_sum = (exp_logits * pos_mask).sum(1)              # (N,)
        all_sum = (exp_logits * (pos_mask | neg_mask)).sum(1)

        # InfoNCE: –log(∑_pos / ∑_all)
        loss = -(torch.log((pos_sum + 1e-8) / (all_sum + 1e-8))).mean()
        return loss

# =============================================================================
# FocalTverskyLoss
# =============================================================================

class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, gamma=2.0,
                 class_weights=None, ignore_index=0):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.cw = class_weights  # tensor (C,) normalizzato
        self.ignore = ignore_index

    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        true  = torch.nn.functional.one_hot(
                    targets, num_classes=logits.size(1)
                ).permute(0,3,1,2).float()

        if self.ignore is not None:
            mask = targets != self.ignore
            probs, true = probs*mask[:,None], true*mask[:,None]

        dims = (0,2,3)
        TP = (probs*true).sum(dims)
        FP = (probs*(1-true)).sum(dims)
        FN = ((1-probs)*true).sum(dims)

        TI = (TP+1e-7) / (TP + self.alpha*FN + self.beta*FP + 1e-7)
        loss = (1 - TI).pow(self.gamma)  # focalizzazione

        if self.cw is not None:
            loss = loss * self.cw

        return loss.mean()


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
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS_TRAIN"], pin_memory=pin_mem, persistent_workers=config["PERSISTENT_WORKERS"])
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS_VAL"], pin_memory=pin_mem, persistent_workers=config["PERSISTENT_WORKERS"])

    model = get_model(config).to(device)
    scaler = GradScaler(enabled=use_amp)
    # Calcola i pesi per la loss sul training set (la funzione va definita sopra)
    class_weights = calculate_class_weights(train_dataset, config)

    criterion_tversky = TverskyLoss(mode='multiclass', alpha=0.7, beta=0.3).to(device)
    criterion_ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=config["IGNORE_INDEX"])
    criterion_focal = FocalLoss(mode='multiclass', gamma=2.0).to(device)
    # 
    pixel_contrast = PixelContrastiveLoss(
        temperature=config["CONTRASTIVE_TEMP"],
        max_samples=config["CONTRASTIVE_MAX_SAMPLES"],
        ignore_index=config["IGNORE_INDEX"]
    ).to(device)

    criterion_ft  = FocalTverskyLoss(
                    alpha=0.9, beta=0.1, gamma=3.0,
                    class_weights=class_weights,
                    ignore_index=config["IGNORE_INDEX"]
                ).to(device)

    criterion_wce = nn.CrossEntropyLoss(
                        weight=class_weights,
                        ignore_index=config["IGNORE_INDEX"]
                    ).to(device)

    def combined_loss(pred, target):
        return criterion_ft(pred, target)

    best_iou = -1.0; best_conf_matrix = None; global_epoch = 0

    # --- FASE 1: FREEZE ENCODER, TRAIN DECODER ---
    # tqdm.write("\n--- INIZIO FASE 1: TRAINING DEL DECODER (ENCODER CONGELATO) ---")
    # for param in model.encoder.parameters(): param.requires_grad = False
    # optimizer = optim.AdamW(list(model.decoder.parameters()) + list(model.segmentation_head.parameters()), lr=config["LR_HEAD"], weight_decay=config["WEIGHT_DECAY"])
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    # for epoch in range(config["FREEZE_EPOCHS"]):
    #     global_epoch += 1; tqdm.write(f"\n--- Fase 1 / Epoch {epoch+1}/{config['FREEZE_EPOCHS']} (Global: {global_epoch}) ---")
    #     train_loss = train_one_epoch(model, train_loader, optimizer, combined_loss, scaler, device, use_amp)
    #     val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate(model, val_loader, combined_loss, device, config)
    #     scheduler.step(val_miou)
    #     tqdm.write(f"Risultati Epoch {global_epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
    #     if val_miou > best_iou:
    #         best_iou, best_conf_matrix = val_miou, val_conf_matrix
    #         torch.save({'model_state_dict': model.state_dict(), 'config': config}, config["MODEL_SAVE_PATH"])
    #         tqdm.write(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<")

    # # --- FASE 2: UNFREEZE ALL, FINE-TUNE ---
    tqdm.write("\n--- INIZIO FASE 2: FINE-TUNING DELL'INTERA RETE (ENCODER SCONGELATO) ---")
    # for param in model.encoder.parameters(): param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=config["LR_FULL"]) #, weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, cooldown=1, min_lr=1e-7)

    for epoch in range(config["UNFREEZE_EPOCHS"]):
        global_epoch += 1; tqdm.write(f"\n--- Fase 2 / Epoch {epoch+1}/{config['UNFREEZE_EPOCHS']} (Global: {global_epoch}) ---")
        
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, 
            seg_loss_fn=combined_loss, contrast_loss_fn=None, seg_w=config["CONTRASTIVE_W"], 
            scaler=scaler, device=device, use_amp=use_amp
        )
        val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate(model, val_loader, combined_loss, device, config)
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

 & "C:/Program Files/Python312/python.exe" "d:/MLPoject-Git/autoencoder-onboard-segmentation/local_execution/Leo/old/7 - All_in_1.py"
Preparazione per il calcolo delle statistiche del dataset...
Calcolo di media e deviazione standard in corso...
Analisi Statistiche Dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:22<00:00,  5.11it/s]

--- Statistiche Dataset Calcolate ---
Media per canale (R, G, B): [0.429485   0.44826734 0.42454103]
Deviazione Standard per canale (R, G, B): [0.26456702 0.26363945 0.2659536 ]
Statistiche personalizzate calcolate e caricate nella configurazione.
Device: cuda | Mixed Precision: True
Strategia Finale: efficientnet-b3 con Augmentation Mirata
Caricamento percorsi dei dataset...
Trovati 931 campioni in 'original'.
Trovati 561 campioni in 'augmented'.
Calcolo Hash: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1492/1492 [00:07<00:00, 213.14it/s]
Creazione dei bucket per l'ottimizzazione del confronto...
Confronto delle immagini all'interno dei bucket (ottimizzato)...
Analisi Bucket: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 72/72 [00:00<00:00, 609.56it/s] 
Trovati 904 gruppi di immagini uniche.
Etichettatura gruppi per lo split: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 904/904 [00:07<00:00, 117.58it/s] 
Split stratificato creato: 1208 train, 284 validation.
Creazione del modello: U-Net con encoder 'efficientnet-b3'
Parametri addestrabili: 13,160,193
Calcolo della distribuzione delle classi per i pesi della loss...
Analisi distribuzione pixel: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 302/302 [00:23<00:00, 12.78it/s] 

Pesi calcolati per la loss (con boost per classi rare):
  - Classe 0: 0.01
  - Classe 1: 0.03
  - Classe 2: 0.20
  - Classe 3: 0.02
  - Classe 4: 6.86
  - Classe 5: 1.63
  - Classe 6: 0.21
  - Classe 7: 0.01
  - Classe 8: 0.04

--- INIZIO FASE 2: FINE-TUNING DELL'INTERA RETE (ENCODER SCONGELATO) ---

--- Fase 2 / Epoch 1/23 (Global: 1) ---
Risultati Epoch 1: Train Loss: 2.5013 | Val Loss: 1.8259 | Val mIoU: 0.0417                                                                                                       
    - Classe 1 (Classe 1): 0.0051
    - Classe 2 (Classe 2): 0.0826
    - Classe 3 (Classe 3): 0.0230
    - Classe 4 (Classe 4): 0.0028
    - Classe 5 (Classe 5): 0.0169
    - Classe 6 (Classe 6): 0.0355
    - Classe 7 (Classe 7): 0.0372
    - Classe 8 (Classe 8): 0.1305

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.0417 <<<

--- Fase 2 / Epoch 2/23 (Global: 2) ---
Risultati Epoch 2: Train Loss: 2.2374 | Val Loss: 1.6107 | Val mIoU: 0.0773                                                                                                       
    - Classe 1 (Classe 1): 0.0070
    - Classe 2 (Classe 2): 0.1462
    - Classe 3 (Classe 3): 0.0325
    - Classe 4 (Classe 4): 0.0037
    - Classe 5 (Classe 5): 0.0770
    - Classe 6 (Classe 6): 0.0907
    - Classe 7 (Classe 7): 0.0188
    - Classe 8 (Classe 8): 0.2421

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.0773 <<<

--- Fase 2 / Epoch 3/23 (Global: 3) ---
Risultati Epoch 3: Train Loss: 2.0188 | Val Loss: 1.4678 | Val mIoU: 0.1341                                                                                                       
    - Classe 1 (Classe 1): 0.0096
    - Classe 2 (Classe 2): 0.1951
    - Classe 3 (Classe 3): 0.0625
    - Classe 4 (Classe 4): 0.0054
    - Classe 5 (Classe 5): 0.1366
    - Classe 6 (Classe 6): 0.1239
    - Classe 7 (Classe 7): 0.0135
    - Classe 8 (Classe 8): 0.5260

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1341 <<<

--- Fase 2 / Epoch 4/23 (Global: 4) ---
Risultati Epoch 4: Train Loss: 1.8703 | Val Loss: 1.3679 | Val mIoU: 0.1734                                                                                                       
    - Classe 1 (Classe 1): 0.0138
    - Classe 2 (Classe 2): 0.2265
    - Classe 3 (Classe 3): 0.0793
    - Classe 4 (Classe 4): 0.0059
    - Classe 5 (Classe 5): 0.1595
    - Classe 6 (Classe 6): 0.1370
    - Classe 7 (Classe 7): 0.0252
    - Classe 8 (Classe 8): 0.7397

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1734 <<<

--- Fase 2 / Epoch 5/23 (Global: 5) ---
Risultati Epoch 5: Train Loss: 1.7364 | Val Loss: 1.2950 | Val mIoU: 0.1922                                                                                                       
    - Classe 1 (Classe 1): 0.0219
    - Classe 2 (Classe 2): 0.2398
    - Classe 3 (Classe 3): 0.1327
    - Classe 4 (Classe 4): 0.0072
    - Classe 5 (Classe 5): 0.1712
    - Classe 6 (Classe 6): 0.1497
    - Classe 7 (Classe 7): 0.0607
    - Classe 8 (Classe 8): 0.7539

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1922 <<<

--- Fase 2 / Epoch 6/23 (Global: 6) ---
Risultati Epoch 6: Train Loss: 1.6455 | Val Loss: 1.2443 | Val mIoU: 0.2035                                                                                                       
    - Classe 1 (Classe 1): 0.0478
    - Classe 2 (Classe 2): 0.2552
    - Classe 3 (Classe 3): 0.1201
    - Classe 4 (Classe 4): 0.0077
    - Classe 5 (Classe 5): 0.1435
    - Classe 6 (Classe 6): 0.1664
    - Classe 7 (Classe 7): 0.1268
    - Classe 8 (Classe 8): 0.7606

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2035 <<<

--- Fase 2 / Epoch 7/23 (Global: 7) ---
Risultati Epoch 7: Train Loss: 1.5725 | Val Loss: 1.1890 | Val mIoU: 0.2340                                                                                                       
    - Classe 1 (Classe 1): 0.0562
    - Classe 2 (Classe 2): 0.2782
    - Classe 3 (Classe 3): 0.1727
    - Classe 4 (Classe 4): 0.0101
    - Classe 5 (Classe 5): 0.1631
    - Classe 6 (Classe 6): 0.1806
    - Classe 7 (Classe 7): 0.2379
    - Classe 8 (Classe 8): 0.7734

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2340 <<<

--- Fase 2 / Epoch 8/23 (Global: 8) ---
Risultati Epoch 8: Train Loss: 1.5132 | Val Loss: 1.1565 | Val mIoU: 0.2590                                                                                                       
    - Classe 1 (Classe 1): 0.0694
    - Classe 2 (Classe 2): 0.2983
    - Classe 3 (Classe 3): 0.1942
    - Classe 4 (Classe 4): 0.0123
    - Classe 5 (Classe 5): 0.1755
    - Classe 6 (Classe 6): 0.1993
    - Classe 7 (Classe 7): 0.3497
    - Classe 8 (Classe 8): 0.7735

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2590 <<<

--- Fase 2 / Epoch 9/23 (Global: 9) ---
Risultati Epoch 9: Train Loss: 1.4580 | Val Loss: 1.1244 | Val mIoU: 0.2862                                                                                                       
    - Classe 1 (Classe 1): 0.1088
    - Classe 2 (Classe 2): 0.3267
    - Classe 3 (Classe 3): 0.1847
    - Classe 4 (Classe 4): 0.0132
    - Classe 5 (Classe 5): 0.1954
    - Classe 6 (Classe 6): 0.2175
    - Classe 7 (Classe 7): 0.4670
    - Classe 8 (Classe 8): 0.7766

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2862 <<<

--- Fase 2 / Epoch 10/23 (Global: 10) ---
Risultati Epoch 10: Train Loss: 1.3914 | Val Loss: 1.0833 | Val mIoU: 0.3144                                                                                                      
    - Classe 1 (Classe 1): 0.1497
    - Classe 2 (Classe 2): 0.3547
    - Classe 3 (Classe 3): 0.1766
    - Classe 4 (Classe 4): 0.0142
    - Classe 5 (Classe 5): 0.2287
    - Classe 6 (Classe 6): 0.2438
    - Classe 7 (Classe 7): 0.5604
    - Classe 8 (Classe 8): 0.7871

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3144 <<<

--- Fase 2 / Epoch 11/23 (Global: 11) ---
Risultati Epoch 11: Train Loss: 1.3252 | Val Loss: 1.0522 | Val mIoU: 0.3372                                                                                                      
    - Classe 1 (Classe 1): 0.1773
    - Classe 2 (Classe 2): 0.3856
    - Classe 3 (Classe 3): 0.1822
    - Classe 4 (Classe 4): 0.0152
    - Classe 5 (Classe 5): 0.2616
    - Classe 6 (Classe 6): 0.2726
    - Classe 7 (Classe 7): 0.6132
    - Classe 8 (Classe 8): 0.7895

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3372 <<<

--- Fase 2 / Epoch 12/23 (Global: 12) ---
Risultati Epoch 12: Train Loss: 1.2787 | Val Loss: 1.0313 | Val mIoU: 0.3554                                                                                                      
    - Classe 1 (Classe 1): 0.1553
    - Classe 2 (Classe 2): 0.4240
    - Classe 3 (Classe 3): 0.1966
    - Classe 4 (Classe 4): 0.0143
    - Classe 5 (Classe 5): 0.2858
    - Classe 6 (Classe 6): 0.2962
    - Classe 7 (Classe 7): 0.6710
    - Classe 8 (Classe 8): 0.8000

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3554 <<<

--- Fase 2 / Epoch 13/23 (Global: 13) ---
Risultati Epoch 13: Train Loss: 1.2392 | Val Loss: 1.0092 | Val mIoU: 0.3731                                                                                                      
    - Classe 1 (Classe 1): 0.1884
    - Classe 2 (Classe 2): 0.4337
    - Classe 3 (Classe 3): 0.1999
    - Classe 4 (Classe 4): 0.0141
    - Classe 5 (Classe 5): 0.3362
    - Classe 6 (Classe 6): 0.3124
    - Classe 7 (Classe 7): 0.6982
    - Classe 8 (Classe 8): 0.8017

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3731 <<<

--- Fase 2 / Epoch 14/23 (Global: 14) ---
Risultati Epoch 14: Train Loss: 1.2173 | Val Loss: 0.9978 | Val mIoU: 0.3899                                                                                                      
    - Classe 1 (Classe 1): 0.2122
    - Classe 2 (Classe 2): 0.4354
    - Classe 3 (Classe 3): 0.2202
    - Classe 4 (Classe 4): 0.0140
    - Classe 5 (Classe 5): 0.3933
    - Classe 6 (Classe 6): 0.3162
    - Classe 7 (Classe 7): 0.7215
    - Classe 8 (Classe 8): 0.8065

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3899 <<<

--- Fase 2 / Epoch 15/23 (Global: 15) ---
Risultati Epoch 15: Train Loss: 1.1646 | Val Loss: 0.9815 | Val mIoU: 0.4010                                                                                                      
    - Classe 1 (Classe 1): 0.2395
    - Classe 2 (Classe 2): 0.4467
    - Classe 3 (Classe 3): 0.2352
    - Classe 4 (Classe 4): 0.0137
    - Classe 5 (Classe 5): 0.3832
    - Classe 6 (Classe 6): 0.3340
    - Classe 7 (Classe 7): 0.7390
    - Classe 8 (Classe 8): 0.8163

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4010 <<<

--- Fase 2 / Epoch 16/23 (Global: 16) ---
Risultati Epoch 16: Train Loss: 1.1338 | Val Loss: 0.9743 | Val mIoU: 0.4053                                                                                                      
    - Classe 1 (Classe 1): 0.2310
    - Classe 2 (Classe 2): 0.4518
    - Classe 3 (Classe 3): 0.2494
    - Classe 4 (Classe 4): 0.0123
    - Classe 5 (Classe 5): 0.4022
    - Classe 6 (Classe 6): 0.3305
    - Classe 7 (Classe 7): 0.7418
    - Classe 8 (Classe 8): 0.8236

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4053 <<<

--- Fase 2 / Epoch 17/23 (Global: 17) ---
Risultati Epoch 17: Train Loss: 1.1079 | Val Loss: 0.9691 | Val mIoU: 0.4182                                                                                                      
    - Classe 1 (Classe 1): 0.2734
    - Classe 2 (Classe 2): 0.4668
    - Classe 3 (Classe 3): 0.2328
    - Classe 4 (Classe 4): 0.0131
    - Classe 5 (Classe 5): 0.4467
    - Classe 6 (Classe 6): 0.3218
    - Classe 7 (Classe 7): 0.7597
    - Classe 8 (Classe 8): 0.8309

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4182 <<<

--- Fase 2 / Epoch 18/23 (Global: 18) ---
Risultati Epoch 18: Train Loss: 1.0801 | Val Loss: 0.9624 | Val mIoU: 0.4189                                                                                                      
    - Classe 1 (Classe 1): 0.2838
    - Classe 2 (Classe 2): 0.4650
    - Classe 3 (Classe 3): 0.2413
    - Classe 4 (Classe 4): 0.0131
    - Classe 5 (Classe 5): 0.4296
    - Classe 6 (Classe 6): 0.3347
    - Classe 7 (Classe 7): 0.7570
    - Classe 8 (Classe 8): 0.8266

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4189 <<<

--- Fase 2 / Epoch 19/23 (Global: 19) ---
Risultati Epoch 19: Train Loss: 1.0666 | Val Loss: 0.9490 | Val mIoU: 0.4207                                                                                                      
    - Classe 1 (Classe 1): 0.2823
    - Classe 2 (Classe 2): 0.4752
    - Classe 3 (Classe 3): 0.2328
    - Classe 4 (Classe 4): 0.0123
    - Classe 5 (Classe 5): 0.4498
    - Classe 6 (Classe 6): 0.3290
    - Classe 7 (Classe 7): 0.7568
    - Classe 8 (Classe 8): 0.8272

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4207 <<<

--- Fase 2 / Epoch 20/23 (Global: 20) ---
Risultati Epoch 20: Train Loss: 1.0385 | Val Loss: 0.9472 | Val mIoU: 0.4262                                                                                                      
    - Classe 1 (Classe 1): 0.3064
    - Classe 2 (Classe 2): 0.4607
    - Classe 3 (Classe 3): 0.2616
    - Classe 4 (Classe 4): 0.0150
    - Classe 5 (Classe 5): 0.4388
    - Classe 6 (Classe 6): 0.3426
    - Classe 7 (Classe 7): 0.7526
    - Classe 8 (Classe 8): 0.8317

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4262 <<<

--- Fase 2 / Epoch 21/23 (Global: 21) ---
Risultati Epoch 21: Train Loss: 1.0177 | Val Loss: 0.9519 | Val mIoU: 0.4301                                                                                                      
    - Classe 1 (Classe 1): 0.3296
    - Classe 2 (Classe 2): 0.4654
    - Classe 3 (Classe 3): 0.2614
    - Classe 4 (Classe 4): 0.0201
    - Classe 5 (Classe 5): 0.4169
    - Classe 6 (Classe 6): 0.3417
    - Classe 7 (Classe 7): 0.7652
    - Classe 8 (Classe 8): 0.8407

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4301 <<<

--- Fase 2 / Epoch 22/23 (Global: 22) ---
Risultati Epoch 22: Train Loss: 0.9912 | Val Loss: 0.9341 | Val mIoU: 0.4285                                                                                                      
    - Classe 1 (Classe 1): 0.3044
    - Classe 2 (Classe 2): 0.4838
    - Classe 3 (Classe 3): 0.2631
    - Classe 4 (Classe 4): 0.0157
    - Classe 5 (Classe 5): 0.4300
    - Classe 6 (Classe 6): 0.3420
    - Classe 7 (Classe 7): 0.7568
    - Classe 8 (Classe 8): 0.8324

--- Fase 2 / Epoch 23/23 (Global: 23) ---
Risultati Epoch 23: Train Loss: 0.9662 | Val Loss: 0.9386 | Val mIoU: 0.4331                                                                                                      
    - Classe 1 (Classe 1): 0.3205
    - Classe 2 (Classe 2): 0.4670
    - Classe 3 (Classe 3): 0.2704
    - Classe 4 (Classe 4): 0.0169
    - Classe 5 (Classe 5): 0.4379
    - Classe 6 (Classe 6): 0.3457
    - Classe 7 (Classe 7): 0.7668
    - Classe 8 (Classe 8): 0.8393

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4331 <<<

--- Training Completato ---
Miglior mIoU di validazione ottenuto: 0.4331
Visualizzazione e salvataggio Matrice di Confusione...
Matrice di confusione salvata in 'confusion_matrix_best.png'
PS D:\MLPoject-Git\autoencoder-onboard-segmentation> & "C:/Program Files/Python312/python.exe" "d:/MLPoject-Git/autoencoder-onboard-segmentation/local_execution/Leo/old/7 - All_in_1.py"
Preparazione per il calcolo delle statistiche del dataset...
Calcolo di media e deviazione standard in corso...
Analisi Statistiche Dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:21<00:00,  5.51it/s]

--- Statistiche Dataset Calcolate ---
Media per canale (R, G, B): [0.429485   0.44826734 0.42454103]
Deviazione Standard per canale (R, G, B): [0.26456702 0.26363945 0.2659536 ]
Statistiche personalizzate calcolate e caricate nella configurazione.
Device: cuda | Mixed Precision: True
Strategia Finale: efficientnet-b4 con Augmentation Mirata
Caricamento percorsi dei dataset...
Trovati 931 campioni in 'original'.
Trovati 561 campioni in 'augmented'.
Calcolo Hash: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1492/1492 [00:06<00:00, 239.05it/s] 
Creazione dei bucket per l'ottimizzazione del confronto...
Confronto delle immagini all'interno dei bucket (ottimizzato)...
Analisi Bucket: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 72/72 [00:00<00:00, 734.07it/s] 
Trovati 904 gruppi di immagini uniche.
Etichettatura gruppi per lo split: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████| 904/904 [00:06<00:00, 130.44it/s] 
Split stratificato creato: 1208 train, 284 validation.
Creazione del modello: U-Net con encoder 'efficientnet-b4'
Parametri addestrabili: 20,226,849
Calcolo della distribuzione delle classi per i pesi della loss...
Analisi distribuzione pixel: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 302/302 [00:22<00:00, 13.32it/s] 

Pesi calcolati per la loss (con boost per classi rare):
  - Classe 0: 0.01
  - Classe 1: 0.03
  - Classe 2: 0.20
  - Classe 3: 0.02
  - Classe 4: 6.86
  - Classe 5: 1.63
  - Classe 6: 0.21
  - Classe 7: 0.01
  - Classe 8: 0.04

--- INIZIO FASE 2: FINE-TUNING DELL'INTERA RETE (ENCODER SCONGELATO) ---

--- Fase 2 / Epoch 1/23 (Global: 1) ---
Risultati Epoch 1: Train Loss: 1.4209 | Val Loss: 1.1732 | Val mIoU: 0.0864
    - Classe 1 (Classe 1): 0.0436
    - Classe 2 (Classe 2): 0.1427
    - Classe 3 (Classe 3): 0.0868
    - Classe 4 (Classe 4): 0.0074
    - Classe 5 (Classe 5): 0.0280
    - Classe 6 (Classe 6): 0.1041
    - Classe 7 (Classe 7): 0.1013
    - Classe 8 (Classe 8): 0.1772

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.0864 <<<

--- Fase 2 / Epoch 2/23 (Global: 2) ---
Risultati Epoch 2: Train Loss: 1.3417 | Val Loss: 1.1124 | Val mIoU: 0.1386                                                                                                       
    - Classe 1 (Classe 1): 0.0204
    - Classe 2 (Classe 2): 0.1856
    - Classe 3 (Classe 3): 0.0905
    - Classe 4 (Classe 4): 0.0111
    - Classe 5 (Classe 5): 0.0263
    - Classe 6 (Classe 6): 0.1371
    - Classe 7 (Classe 7): 0.0714
    - Classe 8 (Classe 8): 0.5666

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1386 <<<

--- Fase 2 / Epoch 3/23 (Global: 3) ---
Risultati Epoch 3: Train Loss: 1.2504 | Val Loss: 1.0753 | Val mIoU: 0.1555                                                                                                       
    - Classe 1 (Classe 1): 0.0082
    - Classe 2 (Classe 2): 0.2166
    - Classe 3 (Classe 3): 0.0885
    - Classe 4 (Classe 4): 0.0121
    - Classe 5 (Classe 5): 0.0352
    - Classe 6 (Classe 6): 0.1624
    - Classe 7 (Classe 7): 0.1021
    - Classe 8 (Classe 8): 0.6188

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1555 <<<

--- Fase 2 / Epoch 4/23 (Global: 4) ---
Risultati Epoch 4: Train Loss: 1.1799 | Val Loss: 1.0479 | Val mIoU: 0.1714                                                                                                       
    - Classe 1 (Classe 1): 0.0047
    - Classe 2 (Classe 2): 0.2542
    - Classe 3 (Classe 3): 0.0898
    - Classe 4 (Classe 4): 0.0137
    - Classe 5 (Classe 5): 0.0417
    - Classe 6 (Classe 6): 0.1733
    - Classe 7 (Classe 7): 0.1383
    - Classe 8 (Classe 8): 0.6553

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1714 <<<

--- Fase 2 / Epoch 5/23 (Global: 5) ---
Risultati Epoch 5: Train Loss: 1.1354 | Val Loss: 1.0270 | Val mIoU: 0.1793                                                                                                       
    - Classe 1 (Classe 1): 0.0021
    - Classe 2 (Classe 2): 0.2713
    - Classe 3 (Classe 3): 0.0500
    - Classe 4 (Classe 4): 0.0116
    - Classe 5 (Classe 5): 0.0587
    - Classe 6 (Classe 6): 0.2006
    - Classe 7 (Classe 7): 0.2105
    - Classe 8 (Classe 8): 0.6291

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1793 <<<

--- Fase 2 / Epoch 6/23 (Global: 6) ---
Risultati Epoch 6: Train Loss: 1.1079 | Val Loss: 1.0203 | Val mIoU: 0.1877                                                                                                       
    - Classe 1 (Classe 1): 0.0024
    - Classe 2 (Classe 2): 0.3052
    - Classe 3 (Classe 3): 0.0438
    - Classe 4 (Classe 4): 0.0117
    - Classe 5 (Classe 5): 0.0632
    - Classe 6 (Classe 6): 0.2062
    - Classe 7 (Classe 7): 0.2642
    - Classe 8 (Classe 8): 0.6050

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1877 <<<

--- Fase 2 / Epoch 7/23 (Global: 7) ---
Risultati Epoch 7: Train Loss: 1.0847 | Val Loss: 1.0118 | Val mIoU: 0.1926                                                                                                       
    - Classe 1 (Classe 1): 0.0063
    - Classe 2 (Classe 2): 0.2966
    - Classe 3 (Classe 3): 0.0341
    - Classe 4 (Classe 4): 0.0137
    - Classe 5 (Classe 5): 0.0610
    - Classe 6 (Classe 6): 0.2171
    - Classe 7 (Classe 7): 0.3118
    - Classe 8 (Classe 8): 0.5999

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1926 <<<

--- Fase 2 / Epoch 8/23 (Global: 8) ---
Risultati Epoch 8: Train Loss: 1.0621 | Val Loss: 1.0045 | Val mIoU: 0.2077                                                                                                       
    - Classe 1 (Classe 1): 0.0081
    - Classe 2 (Classe 2): 0.3350
    - Classe 3 (Classe 3): 0.0199
    - Classe 4 (Classe 4): 0.0124
    - Classe 5 (Classe 5): 0.0695
    - Classe 6 (Classe 6): 0.2225
    - Classe 7 (Classe 7): 0.3832
    - Classe 8 (Classe 8): 0.6112

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2077 <<<

--- Fase 2 / Epoch 9/23 (Global: 9) ---
Risultati Epoch 9: Train Loss: 1.0354 | Val Loss: 0.9928 | Val mIoU: 0.2155                                                                                                       
    - Classe 1 (Classe 1): 0.0210
    - Classe 2 (Classe 2): 0.3685
    - Classe 3 (Classe 3): 0.0135
    - Classe 4 (Classe 4): 0.0156
    - Classe 5 (Classe 5): 0.0606
    - Classe 6 (Classe 6): 0.2276
    - Classe 7 (Classe 7): 0.3943
    - Classe 8 (Classe 8): 0.6230

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2155 <<<

--- Fase 2 / Epoch 10/23 (Global: 10) ---
Risultati Epoch 10: Train Loss: 1.0216 | Val Loss: 0.9872 | Val mIoU: 0.2349                                                                                                      
    - Classe 1 (Classe 1): 0.0322
    - Classe 2 (Classe 2): 0.3892
    - Classe 3 (Classe 3): 0.0105
    - Classe 4 (Classe 4): 0.0140
    - Classe 5 (Classe 5): 0.0704
    - Classe 6 (Classe 6): 0.2388
    - Classe 7 (Classe 7): 0.4643
    - Classe 8 (Classe 8): 0.6597

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2349 <<<

--- Fase 2 / Epoch 11/23 (Global: 11) ---
Risultati Epoch 11: Train Loss: 1.0013 | Val Loss: 0.9798 | Val mIoU: 0.2580                                                                                                      
    - Classe 1 (Classe 1): 0.0723
    - Classe 2 (Classe 2): 0.4150
    - Classe 3 (Classe 3): 0.0145
    - Classe 4 (Classe 4): 0.0183
    - Classe 5 (Classe 5): 0.0617
    - Classe 6 (Classe 6): 0.2511
    - Classe 7 (Classe 7): 0.5270
    - Classe 8 (Classe 8): 0.7042

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2580 <<<

--- Fase 2 / Epoch 12/23 (Global: 12) ---
Risultati Epoch 12: Train Loss: 0.9782 | Val Loss: 0.9746 | Val mIoU: 0.2710                                                                                                      
    - Classe 1 (Classe 1): 0.1004
    - Classe 2 (Classe 2): 0.4133
    - Classe 3 (Classe 3): 0.0139
    - Classe 4 (Classe 4): 0.0197
    - Classe 5 (Classe 5): 0.0648
    - Classe 6 (Classe 6): 0.2590
    - Classe 7 (Classe 7): 0.5708
    - Classe 8 (Classe 8): 0.7261

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2710 <<<

--- Fase 2 / Epoch 13/23 (Global: 13) ---
Risultati Epoch 13: Train Loss: 0.9677 | Val Loss: 0.9671 | Val mIoU: 0.2722                                                                                                      
    - Classe 1 (Classe 1): 0.1573
    - Classe 2 (Classe 2): 0.4265
    - Classe 3 (Classe 3): 0.0145
    - Classe 4 (Classe 4): 0.0240
    - Classe 5 (Classe 5): 0.0669
    - Classe 6 (Classe 6): 0.2538
    - Classe 7 (Classe 7): 0.5167
    - Classe 8 (Classe 8): 0.7179

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2722 <<<

--- Fase 2 / Epoch 14/23 (Global: 14) ---
Risultati Epoch 14: Train Loss: 0.9504 | Val Loss: 0.9559 | Val mIoU: 0.2879                                                                                                      
    - Classe 1 (Classe 1): 0.1875
    - Classe 2 (Classe 2): 0.4331
    - Classe 3 (Classe 3): 0.0162
    - Classe 4 (Classe 4): 0.0273
    - Classe 5 (Classe 5): 0.0690
    - Classe 6 (Classe 6): 0.2651
    - Classe 7 (Classe 7): 0.5625
    - Classe 8 (Classe 8): 0.7428

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2879 <<<

--- Fase 2 / Epoch 15/23 (Global: 15) ---
Risultati Epoch 15: Train Loss: 0.9300 | Val Loss: 0.9532 | Val mIoU: 0.2999                                                                                                      
    - Classe 1 (Classe 1): 0.2287
    - Classe 2 (Classe 2): 0.4310
    - Classe 3 (Classe 3): 0.0162
    - Classe 4 (Classe 4): 0.0291
    - Classe 5 (Classe 5): 0.0722
    - Classe 6 (Classe 6): 0.2741
    - Classe 7 (Classe 7): 0.5935
    - Classe 8 (Classe 8): 0.7546

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2999 <<<

--- Fase 2 / Epoch 16/23 (Global: 16) ---
Risultati Epoch 16: Train Loss: 0.9061 | Val Loss: 0.9454 | Val mIoU: 0.3142                                                                                                      
    - Classe 1 (Classe 1): 0.2986
    - Classe 2 (Classe 2): 0.4591
    - Classe 3 (Classe 3): 0.0125
    - Classe 4 (Classe 4): 0.0215
    - Classe 5 (Classe 5): 0.1027
    - Classe 6 (Classe 6): 0.2637
    - Classe 7 (Classe 7): 0.5968
    - Classe 8 (Classe 8): 0.7587

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3142 <<<

--- Fase 2 / Epoch 17/23 (Global: 17) ---
Risultati Epoch 17: Train Loss: 0.8910 | Val Loss: 0.9456 | Val mIoU: 0.3232                                                                                                      
    - Classe 1 (Classe 1): 0.3326
    - Classe 2 (Classe 2): 0.4494
    - Classe 3 (Classe 3): 0.0139
    - Classe 4 (Classe 4): 0.0295
    - Classe 5 (Classe 5): 0.0896
    - Classe 6 (Classe 6): 0.2944
    - Classe 7 (Classe 7): 0.6122
    - Classe 8 (Classe 8): 0.7640

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3232 <<<

--- Fase 2 / Epoch 18/23 (Global: 18) ---
Risultati Epoch 18: Train Loss: 0.8792 | Val Loss: 0.9432 | Val mIoU: 0.3334                                                                                                      
    - Classe 1 (Classe 1): 0.3784
    - Classe 2 (Classe 2): 0.4484
    - Classe 3 (Classe 3): 0.0172
    - Classe 4 (Classe 4): 0.0498
    - Classe 5 (Classe 5): 0.0957
    - Classe 6 (Classe 6): 0.2840
    - Classe 7 (Classe 7): 0.6253
    - Classe 8 (Classe 8): 0.7689

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3334 <<<

--- Fase 2 / Epoch 19/23 (Global: 19) ---
Risultati Epoch 19: Train Loss: 0.8650 | Val Loss: 0.9461 | Val mIoU: 0.3486                                                                                                      
    - Classe 1 (Classe 1): 0.4280
    - Classe 2 (Classe 2): 0.4408
    - Classe 3 (Classe 3): 0.0177
    - Classe 4 (Classe 4): 0.0497
    - Classe 5 (Classe 5): 0.1253
    - Classe 6 (Classe 6): 0.2954
    - Classe 7 (Classe 7): 0.6545
    - Classe 8 (Classe 8): 0.7772

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3486 <<<

--- Fase 2 / Epoch 20/23 (Global: 20) ---
Risultati Epoch 20: Train Loss: 0.8561 | Val Loss: 0.9428 | Val mIoU: 0.3585                                                                                                      
    - Classe 1 (Classe 1): 0.4450
    - Classe 2 (Classe 2): 0.4463
    - Classe 3 (Classe 3): 0.0166
    - Classe 4 (Classe 4): 0.0629
    - Classe 5 (Classe 5): 0.1319
    - Classe 6 (Classe 6): 0.3072
    - Classe 7 (Classe 7): 0.6734
    - Classe 8 (Classe 8): 0.7845

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3585 <<<

--- Fase 2 / Epoch 21/23 (Global: 21) ---
Risultati Epoch 21: Train Loss: 0.8438 | Val Loss: 0.9427 | Val mIoU: 0.3695                                                                                                      
    - Classe 1 (Classe 1): 0.4680
    - Classe 2 (Classe 2): 0.4432
    - Classe 3 (Classe 3): 0.0188
    - Classe 4 (Classe 4): 0.0704
    - Classe 5 (Classe 5): 0.1660
    - Classe 6 (Classe 6): 0.2984
    - Classe 7 (Classe 7): 0.6985
    - Classe 8 (Classe 8): 0.7924

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3695 <<<

--- Fase 2 / Epoch 22/23 (Global: 22) ---
Risultati Epoch 22: Train Loss: 0.8362 | Val Loss: 0.9293 | Val mIoU: 0.3806                                                                                                      
    - Classe 1 (Classe 1): 0.4759
    - Classe 2 (Classe 2): 0.4551
    - Classe 3 (Classe 3): 0.0203
    - Classe 4 (Classe 4): 0.0558
    - Classe 5 (Classe 5): 0.2312
    - Classe 6 (Classe 6): 0.2947
    - Classe 7 (Classe 7): 0.7016
    - Classe 8 (Classe 8): 0.8104

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3806 <<<

--- Fase 2 / Epoch 23/23 (Global: 23) ---
Risultati Epoch 23: Train Loss: 0.8354 | Val Loss: 0.9381 | Val mIoU: 0.3856                                                                                                      
    - Classe 1 (Classe 1): 0.4830
    - Classe 2 (Classe 2): 0.4527
    - Classe 3 (Classe 3): 0.0193
    - Classe 4 (Classe 4): 0.0549
    - Classe 5 (Classe 5): 0.2377
    - Classe 6 (Classe 6): 0.3054
    - Classe 7 (Classe 7): 0.7201
    - Classe 8 (Classe 8): 0.8118

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3856 <<<

--- Training Completato ---
Miglior mIoU di validazione ottenuto: 0.3856
Visualizzazione e salvataggio Matrice di Confusione...
Matrice di confusione salvata in 'confusion_matrix_best.png'
'''