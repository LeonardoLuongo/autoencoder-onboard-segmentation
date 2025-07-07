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
        "PERSISTENT_WORKERS": True, "RARE_CLASS_INDICES": [4], "RARE_BOOST_FACTOR": 15.0,
        
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_model_b5_aggressive_aug.pth"),

        "ENCODER": "efficientnet-b5", "ENCODER_WEIGHTS": "imagenet",
        "OUTPUT_SIZE": (384, 768), "BATCH_SIZE": 4, 
        "NUM_CLASSES": 9, "IGNORE_INDEX": 0, "VALIDATION_SPLIT": 0.2,

        "FREEZE_EPOCHS": 7, "UNFREEZE_EPOCHS": 53,
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
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.7),
        A.RandomBrightnessContrast(p=0.7),
        A.RandomGamma(p=0.7),
        A.ColorJitter(p=0.7),
        A.GaussianBlur(blur_limit=(3, 5), p=0.4),
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
    paths = {"original": config["DATASET_PATH_TRAIN"], "augmented": config["DATASET_PATH_AUG"], "aggressive": config["DATASET_PATH_AUG_AGGR"]}
    # paths = {"original": config["DATASET_PATH_TRAIN"], "augmented": config["DATASET_PATH_AUG"]}
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
        if torch.rand(1) < 0.5: # es. 50% di probabilità per batch
            images, masks = cutmix_batch(images, masks, alpha=1.0)

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
                    class_weights=None,
                    ignore_index=config["IGNORE_INDEX"]
                ).to(device)

    criterion_wce = nn.CrossEntropyLoss(
                        weight=class_weights,
                        ignore_index=config["IGNORE_INDEX"]
                    ).to(device)

    def combined_loss(pred, target):
        return 0.8 * criterion_ft(pred, target) + 0.2 * criterion_wce(pred, target) 

    best_iou = -1.0; best_conf_matrix = None; global_epoch = 0

    # --- FASE 1: FREEZE ENCODER, TRAIN DECODER ---
    tqdm.write("\n--- INIZIO FASE 1: TRAINING DEL DECODER (ENCODER CONGELATO) ---")
    for param in model.encoder.parameters(): param.requires_grad = False
    optimizer = optim.AdamW(list(model.decoder.parameters()) + list(model.segmentation_head.parameters()), lr=config["LR_HEAD"]) #, weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2, factor=0.5)

    for epoch in range(config["FREEZE_EPOCHS"]):
        global_epoch += 1; tqdm.write(f"\n--- Fase 1 / Epoch {epoch+1}/{config['FREEZE_EPOCHS']} (Global: {global_epoch}) ---")
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, 
            seg_loss_fn=combined_loss, contrast_loss_fn=pixel_contrast, seg_w=config["CONTRASTIVE_W"], 
            scaler=scaler, device=device, use_amp=use_amp
        )
        val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate(model, val_loader, combined_loss, device, config)
        scheduler.step(val_miou)
        tqdm.write(f"Risultati Epoch {global_epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        if val_miou > best_iou:
            best_iou, best_conf_matrix = val_miou, val_conf_matrix
            torch.save({'model_state_dict': model.state_dict(), 'config': config}, config["MODEL_SAVE_PATH"])
            tqdm.write(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<")

    # # --- FASE 2: UNFREEZE ALL, FINE-TUNE ---
    tqdm.write("\n--- INIZIO FASE 2: FINE-TUNING DELL'INTERA RETE (ENCODER SCONGELATO) ---")
    # for param in model.encoder.parameters(): param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=config["LR_FULL"]) #, weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, cooldown=1, min_lr=1e-7)

    for epoch in range(config["UNFREEZE_EPOCHS"]):
        global_epoch += 1; tqdm.write(f"\n--- Fase 2 / Epoch {epoch+1}/{config['UNFREEZE_EPOCHS']} (Global: {global_epoch}) ---")
        
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, 
            seg_loss_fn=combined_loss, contrast_loss_fn=pixel_contrast, seg_w=config["CONTRASTIVE_W"], 
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
> & "C:/Program Files/Python312/python.exe" "d:/MLPoject-Git/autoencoder-onboard-segmentation/local_execution/Leo/old/7 - All_in_1.py"
Preparazione per il calcolo delle statistiche del dataset...
Calcolo di media e deviazione standard in corso...
Analisi Statistiche Dataset: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 117/117 [00:21<00:00,  5.33it/s]

--- Statistiche Dataset Calcolate ---
Media per canale (R, G, B): [0.429485   0.44826734 0.42454103]
Deviazione Standard per canale (R, G, B): [0.26456702 0.26363945 0.2659536 ]
Statistiche personalizzate calcolate e caricate nella configurazione.
Device: cuda | Mixed Precision: True
Strategia Finale: efficientnet-b5 con Augmentation Mirata
Caricamento percorsi dei dataset...
Trovati 931 campioni in 'original'.
Trovati 561 campioni in 'augmented'.
Trovati 1960 campioni in 'aggressive'.
Calcolo Hash: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3452/3452 [00:15<00:00, 217.54it/s]
Creazione dei bucket per l'ottimizzazione del confronto...
Confronto delle immagini all'interno dei bucket (ottimizzato)...
Analisi Bucket: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 86/86 [00:00<00:00, 183.13it/s] 
Trovati 2524 gruppi di immagini uniche.
Etichettatura gruppi per lo split: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████| 2524/2524 [00:22<00:00, 112.66it/s] 
Split stratificato creato: 2746 train, 706 validation.
Creazione del modello: U-Net con encoder 'efficientnet-b5'
Parametri addestrabili: 31,217,161
Calcolo della distribuzione delle classi per i pesi della loss...
Analisi distribuzione pixel: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████| 687/687 [00:47<00:00, 14.39it/s] 

Pesi calcolati per la loss (con boost per classi rare):
  - Classe 0: 0.04
  - Classe 1: 0.51
  - Classe 2: 0.62
  - Classe 3: 0.54
  - Classe 4: 67.84
  - Classe 5: 1.01
  - Classe 6: 0.72
  - Classe 7: 0.23
  - Classe 8: 0.82

--- INIZIO FASE 1: TRAINING DEL DECODER (ENCODER CONGELATO) ---

--- Fase 1 / Epoch 1/7 (Global: 1) ---
Risultati Epoch 1: Train Loss: 0.9366 | Val Loss: 0.6970 | Val mIoU: 0.3254                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3254 <<<

--- Fase 1 / Epoch 2/7 (Global: 2) ---
Risultati Epoch 2: Train Loss: 0.8247 | Val Loss: 0.6453 | Val mIoU: 0.3620                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3620 <<<

--- Fase 1 / Epoch 3/7 (Global: 3) ---
Risultati Epoch 3: Train Loss: 0.7767 | Val Loss: 0.6152 | Val mIoU: 0.3695                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3695 <<<

--- Fase 1 / Epoch 4/7 (Global: 4) ---
Risultati Epoch 4: Train Loss: 0.7532 | Val Loss: 0.6128 | Val mIoU: 0.3910                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3910 <<<

--- Fase 1 / Epoch 5/7 (Global: 5) ---
Risultati Epoch 5: Train Loss: 0.7235 | Val Loss: 0.5913 | Val mIoU: 0.4040                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4040 <<<

--- Fase 1 / Epoch 6/7 (Global: 6) ---
Risultati Epoch 6: Train Loss: 0.7178 | Val Loss: 0.5706 | Val mIoU: 0.4048                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4048 <<<

--- Fase 1 / Epoch 7/7 (Global: 7) ---
Risultati Epoch 7: Train Loss: 0.7007 | Val Loss: 0.5780 | Val mIoU: 0.4241                                                                                                       

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4241 <<<

--- INIZIO FASE 2: FINE-TUNING DELL'INTERA RETE (ENCODER SCONGELATO) ---

--- Fase 2 / Epoch 1/53 (Global: 8) ---
Risultati Epoch 8: Train Loss: 0.6905 | Val Loss: 0.5578 | Val mIoU: 0.4460                                                                                                       
    - Classe 1 (Classe 1): 0.4410
    - Classe 2 (Classe 2): 0.3042
    - Classe 3 (Classe 3): 0.3905
    - Classe 4 (Classe 4): 0.3016
    - Classe 5 (Classe 5): 0.4336
    - Classe 6 (Classe 6): 0.2442
    - Classe 7 (Classe 7): 0.6108
    - Classe 8 (Classe 8): 0.8424

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4460 <<<

--- Fase 2 / Epoch 2/53 (Global: 9) ---
Risultati Epoch 9: Train Loss: 0.6725 | Val Loss: 0.5494 | Val mIoU: 0.4545                                                                                                       
    - Classe 1 (Classe 1): 0.4451
    - Classe 2 (Classe 2): 0.3129
    - Classe 3 (Classe 3): 0.3995
    - Classe 4 (Classe 4): 0.3173
    - Classe 5 (Classe 5): 0.4363
    - Classe 6 (Classe 6): 0.2548
    - Classe 7 (Classe 7): 0.6315
    - Classe 8 (Classe 8): 0.8389

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4545 <<<

--- Fase 2 / Epoch 3/53 (Global: 10) ---
Risultati Epoch 10: Train Loss: 0.6712 | Val Loss: 0.5455 | Val mIoU: 0.4538                                                                                                      
    - Classe 1 (Classe 1): 0.4550
    - Classe 2 (Classe 2): 0.3104
    - Classe 3 (Classe 3): 0.3845
    - Classe 4 (Classe 4): 0.2989
    - Classe 5 (Classe 5): 0.4374
    - Classe 6 (Classe 6): 0.2642
    - Classe 7 (Classe 7): 0.6349
    - Classe 8 (Classe 8): 0.8447

--- Fase 2 / Epoch 4/53 (Global: 11) ---
Risultati Epoch 11: Train Loss: 0.6687 | Val Loss: 0.5469 | Val mIoU: 0.4545                                                                                                      
    - Classe 1 (Classe 1): 0.4472
    - Classe 2 (Classe 2): 0.2977
    - Classe 3 (Classe 3): 0.3950
    - Classe 4 (Classe 4): 0.3074
    - Classe 5 (Classe 5): 0.4390
    - Classe 6 (Classe 6): 0.2611
    - Classe 7 (Classe 7): 0.6436
    - Classe 8 (Classe 8): 0.8453

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4545 <<<

--- Fase 2 / Epoch 5/53 (Global: 12) ---
Risultati Epoch 12: Train Loss: 0.6719 | Val Loss: 0.5408 | Val mIoU: 0.4597                                                                                                      
    - Classe 1 (Classe 1): 0.4407
    - Classe 2 (Classe 2): 0.3203
    - Classe 3 (Classe 3): 0.4079
    - Classe 4 (Classe 4): 0.3129
    - Classe 5 (Classe 5): 0.4410
    - Classe 6 (Classe 6): 0.2598
    - Classe 7 (Classe 7): 0.6413
    - Classe 8 (Classe 8): 0.8536

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4597 <<<

--- Fase 2 / Epoch 6/53 (Global: 13) ---
Risultati Epoch 13: Train Loss: 0.6595 | Val Loss: 0.5429 | Val mIoU: 0.4575                                                                                                      
    - Classe 1 (Classe 1): 0.4393
    - Classe 2 (Classe 2): 0.3119
    - Classe 3 (Classe 3): 0.4103
    - Classe 4 (Classe 4): 0.3139
    - Classe 5 (Classe 5): 0.4406
    - Classe 6 (Classe 6): 0.2631
    - Classe 7 (Classe 7): 0.6347
    - Classe 8 (Classe 8): 0.8463

--- Fase 2 / Epoch 7/53 (Global: 14) ---
Risultati Epoch 14: Train Loss: 0.6678 | Val Loss: 0.5403 | Val mIoU: 0.4553                                                                                                      
    - Classe 1 (Classe 1): 0.4275
    - Classe 2 (Classe 2): 0.3142
    - Classe 3 (Classe 3): 0.4177
    - Classe 4 (Classe 4): 0.3047
    - Classe 5 (Classe 5): 0.4410
    - Classe 6 (Classe 6): 0.2619
    - Classe 7 (Classe 7): 0.6247
    - Classe 8 (Classe 8): 0.8507

--- Fase 2 / Epoch 8/53 (Global: 15) ---
Risultati Epoch 15: Train Loss: 0.6693 | Val Loss: 0.5428 | Val mIoU: 0.4576                                                                                                      
    - Classe 1 (Classe 1): 0.4435
    - Classe 2 (Classe 2): 0.3098
    - Classe 3 (Classe 3): 0.4071
    - Classe 4 (Classe 4): 0.3040
    - Classe 5 (Classe 5): 0.4403
    - Classe 6 (Classe 6): 0.2666
    - Classe 7 (Classe 7): 0.6395
    - Classe 8 (Classe 8): 0.8497

--- Fase 2 / Epoch 9/53 (Global: 16) ---
Risultati Epoch 16: Train Loss: 0.6603 | Val Loss: 0.5379 | Val mIoU: 0.4610                                                                                                      
    - Classe 1 (Classe 1): 0.4344
    - Classe 2 (Classe 2): 0.3167
    - Classe 3 (Classe 3): 0.4168
    - Classe 4 (Classe 4): 0.3151
    - Classe 5 (Classe 5): 0.4435
    - Classe 6 (Classe 6): 0.2670
    - Classe 7 (Classe 7): 0.6413
    - Classe 8 (Classe 8): 0.8534

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4610 <<<

--- Fase 2 / Epoch 10/53 (Global: 17) ---
Risultati Epoch 17: Train Loss: 0.6631 | Val Loss: 0.5396 | Val mIoU: 0.4597                                                                                                      
    - Classe 1 (Classe 1): 0.4336
    - Classe 2 (Classe 2): 0.3145
    - Classe 3 (Classe 3): 0.4160
    - Classe 4 (Classe 4): 0.3114
    - Classe 5 (Classe 5): 0.4415
    - Classe 6 (Classe 6): 0.2677
    - Classe 7 (Classe 7): 0.6383
    - Classe 8 (Classe 8): 0.8544

--- Fase 2 / Epoch 11/53 (Global: 18) ---
Risultati Epoch 18: Train Loss: 0.6535 | Val Loss: 0.5371 | Val mIoU: 0.4587                                                                                                      
    - Classe 1 (Classe 1): 0.4409
    - Classe 2 (Classe 2): 0.3146
    - Classe 3 (Classe 3): 0.4104
    - Classe 4 (Classe 4): 0.3081
    - Classe 5 (Classe 5): 0.4443
    - Classe 6 (Classe 6): 0.2611
    - Classe 7 (Classe 7): 0.6354
    - Classe 8 (Classe 8): 0.8547

--- Fase 2 / Epoch 12/53 (Global: 19) ---
Risultati Epoch 19: Train Loss: 0.6528 | Val Loss: 0.5379 | Val mIoU: 0.4571                                                                                                      
    - Classe 1 (Classe 1): 0.4347
    - Classe 2 (Classe 2): 0.3213
    - Classe 3 (Classe 3): 0.4165
    - Classe 4 (Classe 4): 0.2997
    - Classe 5 (Classe 5): 0.4403
    - Classe 6 (Classe 6): 0.2605
    - Classe 7 (Classe 7): 0.6304
    - Classe 8 (Classe 8): 0.8535

--- Fase 2 / Epoch 13/53 (Global: 20) ---
Risultati Epoch 20: Train Loss: 0.6474 | Val Loss: 0.5352 | Val mIoU: 0.4617                                                                                                      
    - Classe 1 (Classe 1): 0.4349
    - Classe 2 (Classe 2): 0.3216
    - Classe 3 (Classe 3): 0.4191
    - Classe 4 (Classe 4): 0.3146
    - Classe 5 (Classe 5): 0.4438
    - Classe 6 (Classe 6): 0.2711
    - Classe 7 (Classe 7): 0.6354
    - Classe 8 (Classe 8): 0.8531

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4617 <<<

--- Fase 2 / Epoch 14/53 (Global: 21) ---
Risultati Epoch 21: Train Loss: 0.6570 | Val Loss: 0.5331 | Val mIoU: 0.4610                                                                                                      
    - Classe 1 (Classe 1): 0.4387
    - Classe 2 (Classe 2): 0.3196
    - Classe 3 (Classe 3): 0.4142
    - Classe 4 (Classe 4): 0.3130
    - Classe 5 (Classe 5): 0.4447
    - Classe 6 (Classe 6): 0.2717
    - Classe 7 (Classe 7): 0.6320
    - Classe 8 (Classe 8): 0.8540

--- Fase 2 / Epoch 15/53 (Global: 22) ---
Risultati Epoch 22: Train Loss: 0.6535 | Val Loss: 0.5387 | Val mIoU: 0.4609                                                                                                      
    - Classe 1 (Classe 1): 0.4245
    - Classe 2 (Classe 2): 0.3163
    - Classe 3 (Classe 3): 0.4301
    - Classe 4 (Classe 4): 0.3191
    - Classe 5 (Classe 5): 0.4431
    - Classe 6 (Classe 6): 0.2692
    - Classe 7 (Classe 7): 0.6306
    - Classe 8 (Classe 8): 0.8545

--- Fase 2 / Epoch 16/53 (Global: 23) ---
Risultati Epoch 23: Train Loss: 0.6569 | Val Loss: 0.5352 | Val mIoU: 0.4614                                                                                                      
    - Classe 1 (Classe 1): 0.4430
    - Classe 2 (Classe 2): 0.3138
    - Classe 3 (Classe 3): 0.4100
    - Classe 4 (Classe 4): 0.3122
    - Classe 5 (Classe 5): 0.4421
    - Classe 6 (Classe 6): 0.2704
    - Classe 7 (Classe 7): 0.6428
    - Classe 8 (Classe 8): 0.8566

--- Fase 2 / Epoch 17/53 (Global: 24) ---
Risultati Epoch 24: Train Loss: 0.6524 | Val Loss: 0.5352 | Val mIoU: 0.4646                                                                                                      
    - Classe 1 (Classe 1): 0.4194
    - Classe 2 (Classe 2): 0.3194
    - Classe 3 (Classe 3): 0.4326
    - Classe 4 (Classe 4): 0.3295
    - Classe 5 (Classe 5): 0.4464
    - Classe 6 (Classe 6): 0.2751
    - Classe 7 (Classe 7): 0.6400
    - Classe 8 (Classe 8): 0.8543

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4646 <<<

--- Fase 2 / Epoch 18/53 (Global: 25) ---
Risultati Epoch 25: Train Loss: 0.6552 | Val Loss: 0.5325 | Val mIoU: 0.4628                                                                                                      
    - Classe 1 (Classe 1): 0.4297
    - Classe 2 (Classe 2): 0.3165
    - Classe 3 (Classe 3): 0.4290
    - Classe 4 (Classe 4): 0.3199
    - Classe 5 (Classe 5): 0.4482
    - Classe 6 (Classe 6): 0.2714
    - Classe 7 (Classe 7): 0.6347
    - Classe 8 (Classe 8): 0.8533

--- Fase 2 / Epoch 19/53 (Global: 26) ---
Risultati Epoch 26: Train Loss: 0.6499 | Val Loss: 0.5306 | Val mIoU: 0.4655                                                                                                      
    - Classe 1 (Classe 1): 0.4292
    - Classe 2 (Classe 2): 0.3232
    - Classe 3 (Classe 3): 0.4301
    - Classe 4 (Classe 4): 0.3295
    - Classe 5 (Classe 5): 0.4474
    - Classe 6 (Classe 6): 0.2735
    - Classe 7 (Classe 7): 0.6361
    - Classe 8 (Classe 8): 0.8548

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4655 <<<

--- Fase 2 / Epoch 20/53 (Global: 27) ---
Risultati Epoch 27: Train Loss: 0.6568 | Val Loss: 0.5315 | Val mIoU: 0.4611                                                                                                      
    - Classe 1 (Classe 1): 0.4383
    - Classe 2 (Classe 2): 0.3227
    - Classe 3 (Classe 3): 0.4154
    - Classe 4 (Classe 4): 0.3101
    - Classe 5 (Classe 5): 0.4467
    - Classe 6 (Classe 6): 0.2650
    - Classe 7 (Classe 7): 0.6349
    - Classe 8 (Classe 8): 0.8558

--- Fase 2 / Epoch 21/53 (Global: 28) ---
Risultati Epoch 28: Train Loss: 0.6529 | Val Loss: 0.5328 | Val mIoU: 0.4661                                                                                                      
    - Classe 1 (Classe 1): 0.4120
    - Classe 2 (Classe 2): 0.3222
    - Classe 3 (Classe 3): 0.4441
    - Classe 4 (Classe 4): 0.3344
    - Classe 5 (Classe 5): 0.4484
    - Classe 6 (Classe 6): 0.2761
    - Classe 7 (Classe 7): 0.6365
    - Classe 8 (Classe 8): 0.8547

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4661 <<<

--- Fase 2 / Epoch 22/53 (Global: 29) ---
Risultati Epoch 29: Train Loss: 0.6428 | Val Loss: 0.5294 | Val mIoU: 0.4691                                                                                                      
    - Classe 1 (Classe 1): 0.4384
    - Classe 2 (Classe 2): 0.3294
    - Classe 3 (Classe 3): 0.4263
    - Classe 4 (Classe 4): 0.3331
    - Classe 5 (Classe 5): 0.4499
    - Classe 6 (Classe 6): 0.2728
    - Classe 7 (Classe 7): 0.6444
    - Classe 8 (Classe 8): 0.8586

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4691 <<<

--- Fase 2 / Epoch 23/53 (Global: 30) ---
Risultati Epoch 30: Train Loss: 0.6474 | Val Loss: 0.5316 | Val mIoU: 0.4630                                                                                                      
    - Classe 1 (Classe 1): 0.4252
    - Classe 2 (Classe 2): 0.3213
    - Classe 3 (Classe 3): 0.4346
    - Classe 4 (Classe 4): 0.3133
    - Classe 5 (Classe 5): 0.4464
    - Classe 6 (Classe 6): 0.2693
    - Classe 7 (Classe 7): 0.6357
    - Classe 8 (Classe 8): 0.8585

--- Fase 2 / Epoch 24/53 (Global: 31) ---
Risultati Epoch 31: Train Loss: 0.6463 | Val Loss: 0.5305 | Val mIoU: 0.4664                                                                                                      
    - Classe 1 (Classe 1): 0.4347
    - Classe 2 (Classe 2): 0.3271
    - Classe 3 (Classe 3): 0.4285
    - Classe 4 (Classe 4): 0.3235
    - Classe 5 (Classe 5): 0.4495
    - Classe 6 (Classe 6): 0.2720
    - Classe 7 (Classe 7): 0.6393
    - Classe 8 (Classe 8): 0.8568

--- Fase 2 / Epoch 25/53 (Global: 32) ---
Risultati Epoch 32: Train Loss: 0.6453 | Val Loss: 0.5347 | Val mIoU: 0.4668                                                                                                      
    - Classe 1 (Classe 1): 0.4272
    - Classe 2 (Classe 2): 0.3394
    - Classe 3 (Classe 3): 0.4317
    - Classe 4 (Classe 4): 0.3268
    - Classe 5 (Classe 5): 0.4475
    - Classe 6 (Classe 6): 0.2657
    - Classe 7 (Classe 7): 0.6367
    - Classe 8 (Classe 8): 0.8596

--- Fase 2 / Epoch 26/53 (Global: 33) ---
Risultati Epoch 33: Train Loss: 0.6497 | Val Loss: 0.5302 | Val mIoU: 0.4656                                                                                                      
    - Classe 1 (Classe 1): 0.4413
    - Classe 2 (Classe 2): 0.3254
    - Classe 3 (Classe 3): 0.4184
    - Classe 4 (Classe 4): 0.3183
    - Classe 5 (Classe 5): 0.4480
    - Classe 6 (Classe 6): 0.2735
    - Classe 7 (Classe 7): 0.6407
    - Classe 8 (Classe 8): 0.8596

--- Fase 2 / Epoch 27/53 (Global: 34) ---
Risultati Epoch 34: Train Loss: 0.6457 | Val Loss: 0.5289 | Val mIoU: 0.4673                                                                                                      
    - Classe 1 (Classe 1): 0.4275
    - Classe 2 (Classe 2): 0.3292
    - Classe 3 (Classe 3): 0.4345
    - Classe 4 (Classe 4): 0.3289
    - Classe 5 (Classe 5): 0.4494
    - Classe 6 (Classe 6): 0.2699
    - Classe 7 (Classe 7): 0.6388
    - Classe 8 (Classe 8): 0.8604

--- Fase 2 / Epoch 28/53 (Global: 35) ---
Risultati Epoch 35: Train Loss: 0.6482 | Val Loss: 0.5302 | Val mIoU: 0.4635                                                                                                      
    - Classe 1 (Classe 1): 0.4217
    - Classe 2 (Classe 2): 0.3247
    - Classe 3 (Classe 3): 0.4367
    - Classe 4 (Classe 4): 0.3173
    - Classe 5 (Classe 5): 0.4456
    - Classe 6 (Classe 6): 0.2718
    - Classe 7 (Classe 7): 0.6337
    - Classe 8 (Classe 8): 0.8565

--- Fase 2 / Epoch 29/53 (Global: 36) ---
Risultati Epoch 36: Train Loss: 0.6490 | Val Loss: 0.5303 | Val mIoU: 0.4643                                                                                                      
    - Classe 1 (Classe 1): 0.4290
    - Classe 2 (Classe 2): 0.3301
    - Classe 3 (Classe 3): 0.4303
    - Classe 4 (Classe 4): 0.3166
    - Classe 5 (Classe 5): 0.4472
    - Classe 6 (Classe 6): 0.2700
    - Classe 7 (Classe 7): 0.6350
    - Classe 8 (Classe 8): 0.8567

--- Fase 2 / Epoch 30/53 (Global: 37) ---
Risultati Epoch 37: Train Loss: 0.6518 | Val Loss: 0.5298 | Val mIoU: 0.4656                                                                                                      
    - Classe 1 (Classe 1): 0.4341
    - Classe 2 (Classe 2): 0.3215
    - Classe 3 (Classe 3): 0.4300
    - Classe 4 (Classe 4): 0.3198
    - Classe 5 (Classe 5): 0.4472
    - Classe 6 (Classe 6): 0.2731
    - Classe 7 (Classe 7): 0.6397
    - Classe 8 (Classe 8): 0.8594

--- Fase 2 / Epoch 31/53 (Global: 38) ---
Risultati Epoch 38: Train Loss: 0.6516 | Val Loss: 0.5320 | Val mIoU: 0.4629                                                                                                      
    - Classe 1 (Classe 1): 0.4032
    - Classe 2 (Classe 2): 0.3212
    - Classe 3 (Classe 3): 0.4456
    - Classe 4 (Classe 4): 0.3177
    - Classe 5 (Classe 5): 0.4477
    - Classe 6 (Classe 6): 0.2752
    - Classe 7 (Classe 7): 0.6358
    - Classe 8 (Classe 8): 0.8571

--- Fase 2 / Epoch 32/53 (Global: 39) ---
Risultati Epoch 39: Train Loss: 0.6450 | Val Loss: 0.5264 | Val mIoU: 0.4671                                                                                                      
    - Classe 1 (Classe 1): 0.4418
    - Classe 2 (Classe 2): 0.3256
    - Classe 3 (Classe 3): 0.4162
    - Classe 4 (Classe 4): 0.3206
    - Classe 5 (Classe 5): 0.4509
    - Classe 6 (Classe 6): 0.2758
    - Classe 7 (Classe 7): 0.6451
    - Classe 8 (Classe 8): 0.8608

--- Fase 2 / Epoch 33/53 (Global: 40) ---
Risultati Epoch 40: Train Loss: 0.6382 | Val Loss: 0.5316 | Val mIoU: 0.4656                                                                                                      
    - Classe 1 (Classe 1): 0.4423
    - Classe 2 (Classe 2): 0.3210
    - Classe 3 (Classe 3): 0.4178
    - Classe 4 (Classe 4): 0.3180
    - Classe 5 (Classe 5): 0.4464
    - Classe 6 (Classe 6): 0.2737
    - Classe 7 (Classe 7): 0.6460
    - Classe 8 (Classe 8): 0.8598

--- Fase 2 / Epoch 34/53 (Global: 41) ---
Risultati Epoch 41: Train Loss: 0.6473 | Val Loss: 0.5308 | Val mIoU: 0.4646                                                                                                      
    - Classe 1 (Classe 1): 0.4178
    - Classe 2 (Classe 2): 0.3276
    - Classe 3 (Classe 3): 0.4397
    - Classe 4 (Classe 4): 0.3177
    - Classe 5 (Classe 5): 0.4486
    - Classe 6 (Classe 6): 0.2707
    - Classe 7 (Classe 7): 0.6374
    - Classe 8 (Classe 8): 0.8574

--- Fase 2 / Epoch 35/53 (Global: 42) ---
Risultati Epoch 42: Train Loss: 0.6523 | Val Loss: 0.5324 | Val mIoU: 0.4631                                                                                                      
    - Classe 1 (Classe 1): 0.4264
    - Classe 2 (Classe 2): 0.3205
    - Classe 3 (Classe 3): 0.4349
    - Classe 4 (Classe 4): 0.3130
    - Classe 5 (Classe 5): 0.4443
    - Classe 6 (Classe 6): 0.2723
    - Classe 7 (Classe 7): 0.6379
    - Classe 8 (Classe 8): 0.8553

--- Fase 2 / Epoch 36/53 (Global: 43) ---
Risultati Epoch 43: Train Loss: 0.6453 | Val Loss: 0.5289 | Val mIoU: 0.4665                                                                                                      
    - Classe 1 (Classe 1): 0.4300
    - Classe 2 (Classe 2): 0.3266
    - Classe 3 (Classe 3): 0.4331
    - Classe 4 (Classe 4): 0.3269
    - Classe 5 (Classe 5): 0.4483
    - Classe 6 (Classe 6): 0.2707
    - Classe 7 (Classe 7): 0.6385
    - Classe 8 (Classe 8): 0.8580

--- Fase 2 / Epoch 37/53 (Global: 44) ---
Risultati Epoch 44: Train Loss: 0.6486 | Val Loss: 0.5336 | Val mIoU: 0.4685                                                                                                      
    - Classe 1 (Classe 1): 0.4281
    - Classe 2 (Classe 2): 0.3293
    - Classe 3 (Classe 3): 0.4353
    - Classe 4 (Classe 4): 0.3393
    - Classe 5 (Classe 5): 0.4486
    - Classe 6 (Classe 6): 0.2741
    - Classe 7 (Classe 7): 0.6387
    - Classe 8 (Classe 8): 0.8549

--- Fase 2 / Epoch 38/53 (Global: 45) ---
Risultati Epoch 45: Train Loss: 0.6489 | Val Loss: 0.5329 | Val mIoU: 0.4661                                                                                                      
    - Classe 1 (Classe 1): 0.4185
    - Classe 2 (Classe 2): 0.3281
    - Classe 3 (Classe 3): 0.4381
    - Classe 4 (Classe 4): 0.3281
    - Classe 5 (Classe 5): 0.4470
    - Classe 6 (Classe 6): 0.2749
    - Classe 7 (Classe 7): 0.6371
    - Classe 8 (Classe 8): 0.8573

--- Fase 2 / Epoch 39/53 (Global: 46) ---
Risultati Epoch 46: Train Loss: 0.6490 | Val Loss: 0.5280 | Val mIoU: 0.4631                                                                                                      
    - Classe 1 (Classe 1): 0.4299
    - Classe 2 (Classe 2): 0.3221
    - Classe 3 (Classe 3): 0.4326
    - Classe 4 (Classe 4): 0.3143
    - Classe 5 (Classe 5): 0.4470
    - Classe 6 (Classe 6): 0.2728
    - Classe 7 (Classe 7): 0.6287
    - Classe 8 (Classe 8): 0.8574

--- Fase 2 / Epoch 40/53 (Global: 47) ---
Risultati Epoch 47: Train Loss: 0.6397 | Val Loss: 0.5328 | Val mIoU: 0.4639                                                                                                      
    - Classe 1 (Classe 1): 0.4366
    - Classe 2 (Classe 2): 0.3230
    - Classe 3 (Classe 3): 0.4227
    - Classe 4 (Classe 4): 0.3151
    - Classe 5 (Classe 5): 0.4463
    - Classe 6 (Classe 6): 0.2734
    - Classe 7 (Classe 7): 0.6382
    - Classe 8 (Classe 8): 0.8560

--- Fase 2 / Epoch 41/53 (Global: 48) ---
Risultati Epoch 48: Train Loss: 0.6558 | Val Loss: 0.5298 | Val mIoU: 0.4666                                                                                                      
    - Classe 1 (Classe 1): 0.4308
    - Classe 2 (Classe 2): 0.3267
    - Classe 3 (Classe 3): 0.4345
    - Classe 4 (Classe 4): 0.3286
    - Classe 5 (Classe 5): 0.4474
    - Classe 6 (Classe 6): 0.2714
    - Classe 7 (Classe 7): 0.6352
    - Classe 8 (Classe 8): 0.8585

--- Fase 2 / Epoch 42/53 (Global: 49) ---
Risultati Epoch 49: Train Loss: 0.6529 | Val Loss: 0.5269 | Val mIoU: 0.4664                                                                                                      
    - Classe 1 (Classe 1): 0.4425
    - Classe 2 (Classe 2): 0.3288
    - Classe 3 (Classe 3): 0.4217
    - Classe 4 (Classe 4): 0.3252
    - Classe 5 (Classe 5): 0.4505
    - Classe 6 (Classe 6): 0.2702
    - Classe 7 (Classe 7): 0.6343
    - Classe 8 (Classe 8): 0.8582

--- Fase 2 / Epoch 43/53 (Global: 50) ---
Risultati Epoch 50: Train Loss: 0.6408 | Val Loss: 0.5298 | Val mIoU: 0.4650                                                                                                      
    - Classe 1 (Classe 1): 0.4355
    - Classe 2 (Classe 2): 0.3277
    - Classe 3 (Classe 3): 0.4284
    - Classe 4 (Classe 4): 0.3262
    - Classe 5 (Classe 5): 0.4478
    - Classe 6 (Classe 6): 0.2731
    - Classe 7 (Classe 7): 0.6264
    - Classe 8 (Classe 8): 0.8551

--- Fase 2 / Epoch 44/53 (Global: 51) ---
Risultati Epoch 51: Train Loss: 0.6515 | Val Loss: 0.5306 | Val mIoU: 0.4654                                                                                                      
    - Classe 1 (Classe 1): 0.4359
    - Classe 2 (Classe 2): 0.3253
    - Classe 3 (Classe 3): 0.4239
    - Classe 4 (Classe 4): 0.3169
    - Classe 5 (Classe 5): 0.4470
    - Classe 6 (Classe 6): 0.2720
    - Classe 7 (Classe 7): 0.6436
    - Classe 8 (Classe 8): 0.8590

--- Fase 2 / Epoch 45/53 (Global: 52) ---
Risultati Epoch 52: Train Loss: 0.6401 | Val Loss: 0.5273 | Val mIoU: 0.4667                                                                                                      
    - Classe 1 (Classe 1): 0.4279
    - Classe 2 (Classe 2): 0.3311
    - Classe 3 (Classe 3): 0.4331
    - Classe 4 (Classe 4): 0.3168
    - Classe 5 (Classe 5): 0.4488
    - Classe 6 (Classe 6): 0.2732
    - Classe 7 (Classe 7): 0.6423
    - Classe 8 (Classe 8): 0.8601

--- Fase 2 / Epoch 46/53 (Global: 53) ---
Risultati Epoch 53: Train Loss: 0.6454 | Val Loss: 0.5274 | Val mIoU: 0.4672                                                                                                      
    - Classe 1 (Classe 1): 0.4389
    - Classe 2 (Classe 2): 0.3287
    - Classe 3 (Classe 3): 0.4248
    - Classe 4 (Classe 4): 0.3257
    - Classe 5 (Classe 5): 0.4516
    - Classe 6 (Classe 6): 0.2701
    - Classe 7 (Classe 7): 0.6408
    - Classe 8 (Classe 8): 0.8569

--- Fase 2 / Epoch 47/53 (Global: 54) ---
Risultati Epoch 54: Train Loss: 0.6460 | Val Loss: 0.5293 | Val mIoU: 0.4664                                                                                                      
    - Classe 1 (Classe 1): 0.4416
    - Classe 2 (Classe 2): 0.3286
    - Classe 3 (Classe 3): 0.4188
    - Classe 4 (Classe 4): 0.3205
    - Classe 5 (Classe 5): 0.4486
    - Classe 6 (Classe 6): 0.2730
    - Classe 7 (Classe 7): 0.6408
    - Classe 8 (Classe 8): 0.8588

--- Fase 2 / Epoch 48/53 (Global: 55) ---
Risultati Epoch 55: Train Loss: 0.6395 | Val Loss: 0.5303 | Val mIoU: 0.4664                                                                                                      
    - Classe 1 (Classe 1): 0.4170
    - Classe 2 (Classe 2): 0.3284
    - Classe 3 (Classe 3): 0.4426
    - Classe 4 (Classe 4): 0.3315
    - Classe 5 (Classe 5): 0.4488
    - Classe 6 (Classe 6): 0.2721
    - Classe 7 (Classe 7): 0.6344
    - Classe 8 (Classe 8): 0.8564

--- Fase 2 / Epoch 49/53 (Global: 56) ---
Risultati Epoch 56: Train Loss: 0.6456 | Val Loss: 0.5300 | Val mIoU: 0.4658                                                                                                      
    - Classe 1 (Classe 1): 0.4342
    - Classe 2 (Classe 2): 0.3205
    - Classe 3 (Classe 3): 0.4302
    - Classe 4 (Classe 4): 0.3238
    - Classe 5 (Classe 5): 0.4491
    - Classe 6 (Classe 6): 0.2744
    - Classe 7 (Classe 7): 0.6375
    - Classe 8 (Classe 8): 0.8569

--- Fase 2 / Epoch 50/53 (Global: 57) ---
Risultati Epoch 57: Train Loss: 0.6376 | Val Loss: 0.5307 | Val mIoU: 0.4639                                                                                                      
    - Classe 1 (Classe 1): 0.4431
    - Classe 2 (Classe 2): 0.3311
    - Classe 3 (Classe 3): 0.4180
    - Classe 4 (Classe 4): 0.3206
    - Classe 5 (Classe 5): 0.4485
    - Classe 6 (Classe 6): 0.2660
    - Classe 7 (Classe 7): 0.6262
    - Classe 8 (Classe 8): 0.8573

--- Fase 2 / Epoch 51/53 (Global: 58) ---
Risultati Epoch 58: Train Loss: 0.6438 | Val Loss: 0.5329 | Val mIoU: 0.4657                                                                                                      
    - Classe 1 (Classe 1): 0.4287
    - Classe 2 (Classe 2): 0.3074
    - Classe 3 (Classe 3): 0.4313
    - Classe 4 (Classe 4): 0.3263
    - Classe 5 (Classe 5): 0.4491
    - Classe 6 (Classe 6): 0.2787
    - Classe 7 (Classe 7): 0.6487
    - Classe 8 (Classe 8): 0.8552

--- Fase 2 / Epoch 52/53 (Global: 59) ---
Risultati Epoch 59: Train Loss: 0.6525 | Val Loss: 0.5340 | Val mIoU: 0.4668                                                                                                      
    - Classe 1 (Classe 1): 0.4110
    - Classe 2 (Classe 2): 0.3225
    - Classe 3 (Classe 3): 0.4427
    - Classe 4 (Classe 4): 0.3324
    - Classe 5 (Classe 5): 0.4483
    - Classe 6 (Classe 6): 0.2750
    - Classe 7 (Classe 7): 0.6430
    - Classe 8 (Classe 8): 0.8592

--- Fase 2 / Epoch 53/53 (Global: 60) ---

--- Fase 2 / Epoch 53/53 (Global: 60) ---

--- Fase 2 / Epoch 53/53 (Global: 60) ---

--- Fase 2 / Epoch 53/53 (Global: 60) ---
Risultati Epoch 60: Train Loss: 0.6422 | Val Loss: 0.5286 | Val mIoU: 0.4662
    - Classe 1 (Classe 1): 0.4223
    - Classe 2 (Classe 2): 0.3230
    - Classe 3 (Classe 3): 0.4377
    - Classe 4 (Classe 4): 0.3284
    - Classe 5 (Classe 5): 0.4500
    - Classe 6 (Classe 6): 0.2779
    - Classe 7 (Classe 7): 0.6346
    - Classe 8 (Classe 8): 0.8558

--- Training Completato ---
Miglior mIoU di validazione ottenuto: 0.4691
Visualizzazione e salvataggio Matrice di Confusione...
Matrice di confusione salvata in 'confusion_matrix_best.png'

'''