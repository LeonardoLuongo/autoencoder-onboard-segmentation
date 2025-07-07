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
from lovasz import lovasz_softmax
import kornia.augmentation as K
import kornia.geometry as G

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
        "RANDOM_SEED": 42, "NUM_WORKERS_TRAIN": 3, "NUM_WORKERS_VAL": 2, "USE_AMP": True,
        "PERSISTENT_WORKERS": True, "RARE_CLASS_INDICES": [4, 6], "RARE_BOOST_FACTOR": 25.0,
        
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_model_b3.pth"),

        "ENCODER": "efficientnet-b3", "ENCODER_WEIGHTS": "imagenet",
        "OUTPUT_SIZE": (384, 768), "BATCH_SIZE": 4, 
        "NUM_CLASSES": 9, "IGNORE_INDEX": 0, "VALIDATION_SPLIT": 0.2,

        "FREEZE_EPOCHS": 7, "UNFREEZE_EPOCHS": 23, "FINE_TUNING_EPOCHS": 10,
        "LR_HEAD": 1e-3, "LR_FULL": 1e-4, "WEIGHT_DECAY": 1e-6,

        "CONTRASTIVE_TEMP": 0.1,      # τ dell’InfoNCE
        "CONTRASTIVE_MAX_SAMPLES": 4096,  # pixel da campionare per batch
        "CONTRASTIVE_W": 0.6,         # peso della loss rispetto alle loss di segmento
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

def get_cpu_transforms(config):
    return A.Compose([
        A.Resize(height=config["OUTPUT_SIZE"][0], width=config["OUTPUT_SIZE"][1]),
        ToTensorV2(),     
    ])

def get_gpu_augmentations(config: Dict[str, Any], aggressive_p: float = 0.5) -> nn.Module:
    """Pipeline GPU: tutte le augmentation pesanti, definite come un modulo nn.Sequential."""
    # Kornia lavora direttamente su tensori (B, C, H, W)
    gpu_aug = K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=aggressive_p / 2),
        K.RandomRotation(degrees=90, p=aggressive_p),
        K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        K.RandomGamma(gamma=(0.8, 1.2), p=0.7),
        K.RandomGaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0), p=0.4),
        # Aggiungiamo il flag per sincronizzare le trasformazioni tra immagine e maschera
        data_keys=["input", "mask"],
        same_on_batch=False # Applica una trasformazione diversa a ogni immagine nel batch
    )
    return gpu_aug.to(config["DEVICE"])

# La pipeline di validazione rimane su CPU perché non ha augmentation random
def get_val_transforms(output_size: Tuple[int, int], mean: float, std: float) -> A.Compose:
    """Pipeline di validazione (solo preprocessing su CPU)."""
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
    rare_class_indices = config["RARE_CLASS_INDICES"]
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
        
    weights = np.clip(weights, a_min=None, a_max=30.0)   # clamp prima
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

def train_one_epoch(model, loader, optimizer, scheduler, loss_function, scaler, device, use_amp, current_epoch, total_epochs, gpu_augmentations: nn.Module, mean, std, need_feats=False, clip_grad_norm=1.0) -> float:
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    
    for step, (images, masks) in enumerate(progress_bar):
        images = images.float() / 255.0
        images, raw_masks = gpu_augmentations(images, masks.float())
        masks_long = raw_masks.long()
        images = K.Normalize(mean=torch.tensor(mean, device=images.device),
                     std=torch.tensor(std,  device=images.device)
                    )(images)

        if current_epoch <= total_epochs - 5 and torch.rand(1) < 0.5:
            images, masks_long = cutmix_batch(images, masks_long, alpha=1.0)

        images, masks_long = images.to(device), masks_long.to(device, dtype=torch.long)
        
        optimizer.zero_grad()

        with autocast(enabled=use_amp, device_type=device.type if use_amp else 'cpu'):
            outputs = model(images)
            feats = model.encoder(images)[-1] if need_feats else None
            # Passiamo a loss_function le maschere nel formato che si aspetta (B, 1, H, W)
            loss = loss_function(outputs, masks_long, feats)

        # Scaliamo la loss e calcoliamo i gradienti
        scaler.scale(loss).backward()
        
        # --- FIX: GRADIENT CLIPPING ---
        # Prima di optimizer.step(), de-scaliamo i gradienti per poterli "vedere" e clippare
        scaler.unscale_(optimizer)
        # Applichiamo il clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad_norm)
        # -----------------------------

        # Ora l'optimizer può fare il suo passo
        scaler.step(optimizer)
        scaler.update()
        
        if scheduler is not None:
            scheduler.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
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

        # --- FIX QUI ---
        # Assicuriamoci che 'targets' sia (B, H, W) e non (B, 1, H, W)
        # prima di passarlo a one_hot.
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets_squeezed = targets.squeeze(1)
        else:
            targets_squeezed = targets
        
        true  = torch.nn.functional.one_hot(
                    targets_squeezed, num_classes=logits.size(1)
                ).permute(0,3,1,2).float()

        if self.ignore is not None:
            # Usiamo 'targets' originale (che potrebbe essere 4D) per la maschera
            mask = targets != self.ignore
            # Assicuriamo che la maschera possa essere trasmessa (broadcast) correttamente
            if mask.dim() == 3:
                mask = mask.unsqueeze(1) # Da (B, H, W) a (B, 1, H, W)
            
            probs, true = probs*mask, true*mask

        dims = (0,2,3)
        TP = (probs*true).sum(dims)
        FP = (probs*(1-true)).sum(dims)
        FN = ((1-probs)*true).sum(dims)

        TI = (TP+1e-7) / (TP + self.alpha*FN + self.beta*FP + 1e-7)
        loss = (1 - TI).pow(self.gamma)  # focalizzazione

        if self.cw is not None:
            # Applica i pesi alle classi. Assicurati che cw sia broadcast-able.
            loss = loss * self.cw.view(1, -1).squeeze(0)

        # Rimuovi le classi ignorate dal calcolo della media finale
        if self.ignore is not None and self.cw is not None:
            valid_classes_mask = torch.ones_like(self.cw, dtype=torch.bool)
            # Ignora la classe 0 e tutte le altre classi con peso 0 se necessario
            valid_classes_mask[self.ignore] = False
            
            # Calcola la media solo sulle classi valide
            return loss[valid_classes_mask].mean()
        
        return loss.mean()


# =============================================================================
# 6. FUNZIONE PRINCIPALE (CON AUGMENTATION AGGRESSIVA)
# =============================================================================
def main():
    config = get_config(); seed_everything(config["RANDOM_SEED"]); device = config["DEVICE"]
    use_amp = config["USE_AMP"] and AMP_AVAILABLE and device.type == 'cuda'
    tqdm.write(f"Device: {device} | Mixed Precision: {use_amp}"); tqdm.write(f"Strategia Finale: {config['ENCODER']} con Augmentation Mirata")

    train_dirs, val_dirs = get_grouped_stratified_split(config)
    
    cpu_transforms = get_cpu_transforms(config)
    val_transforms = get_val_transforms(config["OUTPUT_SIZE"], config["MEAN"], config["STD"])
    
    # FIX: Aggiunto align_corners=False per silenziare l'avviso di Kornia
    def get_gpu_augmentations(config: Dict[str, Any], aggressive_p: float = 0.5) -> nn.Module:
        gpu_aug = K.AugmentationSequential(
            K.RandomHorizontalFlip(p=0.5),
            K.RandomVerticalFlip(p=aggressive_p / 2),
            K.RandomRotation(degrees=90, p=aggressive_p, align_corners=False), # FIX
            K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
            K.RandomGamma(gamma=(0.8, 1.2), p=0.7),
            K.RandomGaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0), p=0.4),
            data_keys=["input", "mask"],
            same_on_batch=False
        )
        return gpu_aug.to(config["DEVICE"])

    gpu_augmentations_freeze = get_gpu_augmentations(config, aggressive_p=0.1)
    gpu_augmentations_unfreeze = get_gpu_augmentations(config, aggressive_p=0.5)

    train_dataset = SegmentationDataset(train_dirs, cpu_transforms, config)
    val_dataset = SegmentationDataset(val_dirs, val_transforms, config)
    
    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS_TRAIN"], pin_memory=pin_mem, persistent_workers=config["PERSISTENT_WORKERS"])
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS_VAL"], pin_memory=pin_mem, persistent_workers=config["PERSISTENT_WORKERS"])

    model = get_model(config).to(device)
    scaler = GradScaler(enabled=use_amp)
    class_weights = calculate_class_weights(train_dataset, config)

    pixel_contrast = PixelContrastiveLoss(
        temperature=config["CONTRASTIVE_TEMP"],
        max_samples=config["CONTRASTIVE_MAX_SAMPLES"],
        ignore_index=config["IGNORE_INDEX"]
    ).to(device)

    # Usa la classe FocalTverskyLoss corretta che hai definito
    criterion_ft  = FocalTverskyLoss(
                    alpha=0.9, beta=0.1, gamma=3.0,
                    class_weights=class_weights, # Passiamo i pesi qui
                    ignore_index=config["IGNORE_INDEX"]
                ).to(device)

    criterion_wce = nn.CrossEntropyLoss(weight=class_weights,ignore_index=config["IGNORE_INDEX"]).to(device)

    best_iou = -1.0; best_conf_matrix = None; global_epoch = 0

    # --- FASE 1: FREEZE ENCODER, TRAIN DECODER ---
    tqdm.write("\n--- INIZIO FASE 1: TRAINING DEL DECODER (ENCODER CONGELATO) ---")
    
    # Definisci la loss per la Fase 1
    def loss_phase1(pred, target, feats=None):
        loss_segmentation = criterion_ft(pred, target)
        if feats is None: # Per la validazione
            return loss_segmentation
        
        mask_ds = F.interpolate(
            target.unsqueeze(1).float(),
            size=feats.shape[2:], mode="nearest"
         ).squeeze(1).long()
        # FIX: Usa CONTRASTIVE_W come peso, non TEMP
        loss_contrast = pixel_contrast(feats, mask_ds) * config["CONTRASTIVE_W"]
        return loss_segmentation + loss_contrast

    for param in model.encoder.parameters(): param.requires_grad = False
    optimizer = optim.AdamW(list(model.decoder.parameters()) + list(model.segmentation_head.parameters()), lr=config["LR_HEAD"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config["LR_HEAD"], steps_per_epoch=len(train_loader), epochs=config["FREEZE_EPOCHS"], pct_start=0.1)

    # FIX: De-commentato il loop della Fase 1
    for epoch in range(config["FREEZE_EPOCHS"]):
        global_epoch += 1; tqdm.write(f"\n--- Fase 1 / Epoch {epoch+1}/{config['FREEZE_EPOCHS']} (Global: {global_epoch}) ---")
        # FIX: Corretto errore di sintassi (virgola mancante) e aggiunto need_feats=True
        # Esempio di chiamata in Fase 1 (e successive)
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler, 
            loss_function=loss_phase1, scaler=scaler, device=device, use_amp=use_amp, 
            current_epoch=epoch, total_epochs=config["FREEZE_EPOCHS"], 
            gpu_augmentations=gpu_augmentations_freeze, need_feats=True,
            clip_grad_norm=1.0, mean=config["MEAN"], std=config["STD"]
        )
        # FIX: Usa una loss di validazione che non richiede 'feats'
        val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate(model, val_loader, criterion_ft, device, config)
        
        tqdm.write(f"Risultati Epoch {global_epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        if val_miou > best_iou:
            best_iou, best_conf_matrix = val_miou, val_conf_matrix
            torch.save({'model_state_dict': model.state_dict(), 'config': config}, config["MODEL_SAVE_PATH"])
            tqdm.write(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<")

    # --- FASE 2: UNFREEZE ALL, FINE-TUNE ---
    tqdm.write("\n--- INIZIO FASE 2: FINE-TUNING DELL'INTERA RETE (ENCODER SCONGELATO) ---")

    for param in model.encoder.parameters(): param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=config["LR_FULL"], weight_decay=config["WEIGHT_DECAY"])
    scheduler_plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, cooldown=1, min_lr=1e-7)
            
    def loss_phase2(pred, target, feats=None):
        return 0.8 * criterion_ft(pred, target) + 0.2 * criterion_wce(pred, target)

    for epoch in range(config["UNFREEZE_EPOCHS"]):
        global_epoch += 1; tqdm.write(f"\n--- Fase 2 / Epoch {epoch+1}/{config['UNFREEZE_EPOCHS']} (Global: {global_epoch}) ---")
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scheduler=None, loss_function=loss_phase2, 
            scaler=scaler, device=device, use_amp=use_amp, current_epoch=epoch, total_epochs=config["UNFREEZE_EPOCHS"], 
            gpu_augmentations=gpu_augmentations_unfreeze, need_feats=False, clip_grad_norm=1.0, mean=config["MEAN"], std=config["STD"]
        )
        val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate(model, val_loader, loss_phase2, device, config)
        scheduler_plateau.step(val_miou)
        tqdm.write(f"Risultati Epoch {global_epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        for i, iou in enumerate(val_iou_per_class):
            if i != config["IGNORE_INDEX"]: tqdm.write(f"    - Classe {i} ({config['CLASS_NAMES'][i]}): {iou:.4f}")
        if val_miou > best_iou:
            best_iou, best_conf_matrix = val_miou, val_conf_matrix
            torch.save({'model_state_dict': model.state_dict(), 'config': config}, config["MODEL_SAVE_PATH"])
            tqdm.write(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<")
    
    # --- FASE 3: FINE-TUNING FINALE CON LOVASZ ---
    tqdm.write("\n--- INIZIO FASE 3: FINE-TUNING FINALE (LOVASZ LOSS) ---")
    
    def loss_phase3(pred, target, feats=None):
        # Lovasz lavora meglio su probabilità, quindi applichiamo Softmax
        # La lovasz_softmax interna applica già il log-softmax, quindi passiamo i logits
        loss_lovasz = lovasz_softmax(pred, target, ignore=config["IGNORE_INDEX"])
        return 0.2 * criterion_ft(pred, target) + 0.8 * loss_lovasz

    for epoch in range(config["FINE_TUNING_EPOCHS"]):
        global_epoch += 1; tqdm.write(f"\n--- Fase 3 / Epoch {epoch+1}/{config['FINE_TUNING_EPOCHS']} (Global: {global_epoch}) ---")
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scheduler=None, loss_function=loss_phase3, 
            scaler=scaler, device=device, use_amp=use_amp, current_epoch=epoch, total_epochs=config["FINE_TUNING_EPOCHS"], 
            gpu_augmentations=gpu_augmentations_unfreeze, need_feats=False, clip_grad_norm=1.0, mean=config["MEAN"], std=config["STD"]
        )
        val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate(model, val_loader, loss_phase3, device, config)
        scheduler_plateau.step(val_miou)
        tqdm.write(f"Risultati Epoch {global_epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
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

