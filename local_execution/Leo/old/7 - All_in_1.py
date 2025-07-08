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
import kornia.geometry.transform as KGT
from kornia.utils import image_to_tensor
import torch

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
        "RANDOM_SEED": 42, "NUM_WORKERS_TRAIN": 5, "NUM_WORKERS_VAL": 2, "USE_AMP": True,
        "PERSISTENT_WORKERS": True, "RARE_CLASS_INDICES": [4, 6], "RARE_BOOST_FACTOR": 25.0,
        
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_model_b3.pth"),

        "ENCODER": "efficientnet-b3", "ENCODER_WEIGHTS": "imagenet",
        "OUTPUT_SIZE": (384, 768), "BATCH_SIZE": 3, 
        "NUM_CLASSES": 9, "IGNORE_INDEX": 0, "VALIDATION_SPLIT": 0.2,

        "FREEZE_EPOCHS": 7, "UNFREEZE_EPOCHS": 23, "FINE_TUNING_EPOCHS": 10,
        "LR_HEAD": 1e-3, "LR_FULL": 1e-4, "WEIGHT_DECAY": 1e-6,

        "CONTRASTIVE_TEMP": 0.1,      # τ dell’InfoNCE
        "CONTRASTIVE_MAX_SAMPLES": 4096*3,  # pixel da campionare per batch
        "CONTRASTIVE_W": 1,         # peso della loss rispetto alle loss di segmento
    }
    
    # FIX: Nomi delle classi anonimi come da richiesta
    config["CLASS_NAMES"] = ["Ignored"] + [f"Classe {i}" for i in range(1, config["NUM_CLASSES"])]

    # --- CALCOLO CORRETTO DI MEDIA E DEVIAZIONE STANDARD ---
    # tqdm.write("Preparazione per il calcolo delle statistiche del dataset...")
    
    # # 1. Pipeline di trasformazione SENZA A.Normalize
    # stats_transforms = A.Compose([
    #     A.Resize(height=config["OUTPUT_SIZE"][0], width=config["OUTPUT_SIZE"][1]),
    #     # ToTensorV2 è fondamentale perché scala le immagini da [0, 255] a [0, 1]
    #     A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
    #     ToTensorV2(),
    # ])

    # # 2. Crea un Dataset temporaneo usando solo i dati originali
    # train_original_dirs = [d for d in glob.glob(os.path.join(config["DATASET_PATH_TRAIN"], "*")) if os.path.isdir(d)]
    # stats_dataset = SegmentationDataset(
    #     sample_dirs=train_original_dirs,
    #     transform=stats_transforms,
    #     config=config  # Passa la config per l'assert nel dataset
    # )

    # # 3. Avvolgi il Dataset in un DataLoader (IL FIX FONDAMENTALE)
    # stats_loader = DataLoader(
    #     stats_dataset, 
    #     batch_size=config["BATCH_SIZE"] * 2, # Possiamo usare un batch size più grande qui
    #     shuffle=False, 
    #     num_workers=config["NUM_WORKERS_TRAIN"],
    #     persistent_workers=config["PERSISTENT_WORKERS"]
    # )
    
    # # 4. Ora chiama la funzione con il DataLoader
    # mean, std = calculate_mean_std(stats_loader)

    # assert 0.0 <= mean.min() and mean.max() <= 1.0,  "Mean fuori range 0-1"
    # assert 0.0 <  std.min()  and std.max()  <  1.0,  "Std anomala"
    
    # # 5. Salva i valori calcolati nella configurazione
    # config["MEAN"] = mean
    # config["STD"] = std

    # tqdm.write("Statistiche personalizzate calcolate e caricate nella configurazione.")
    return config


import hashlib, json, torch, os, pickle

CACHE_PATH = "./cache/preprocessing_cache.pth"
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

def get_dataset_signature(paths: list[str]) -> str:
    """
    Crea una stringa‐hash semplice per capire se il dataset è cambiato.
    Usa nomi file + dimensione + mtime (veloce, niente lettura byte-a-byte).
    """
    h = hashlib.sha1()
    for p in sorted(paths):
        st = os.stat(p)
        h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()


def load_or_compute_preprocessing(config: dict) -> dict:
    """
    Se esiste la cache e combacia con il dataset attuale => la carico.
    Altrimenti ricalcolo tutto e salvo.
    """
    # ------------- 1) individua tutte le cartelle/immagini grezze ---------- #
    all_sample_dirs = [d for d in glob.glob(os.path.join(
        config["DATASET_PATH_TRAIN"], "*")) if os.path.isdir(d)]
    dataset_sig = get_dataset_signature(all_sample_dirs)

    # ------------- 2) tenta di caricare la cache --------------------------- #
    if os.path.isfile(CACHE_PATH):
        cache = torch.load(CACHE_PATH, weights_only=False)
        if cache.get("dataset_version") == dataset_sig:
            print("Cache trovata e valida — salto il preprocessing.")
            return cache
        else:
            print("Cache trovata ma il dataset è cambiato → ricalcolo.")

    # ------------- 3) se arrivi qui devi calcolare ex-novo ------------------ #
    print("Calcolo mean/std, split, pesi classi … (prima e unica volta)")
    
    # Usa la pipeline Kornia minimale
    cpu_transforms_kornia = get_cpu_transforms_kornia(config)
    
    # Il dataset ora usa la pipeline Kornia
    stats_dataset = SegmentationDataset(all_sample_dirs, cpu_transforms_kornia, config)
    
    mean, std = calculate_mean_std(
        DataLoader(
            stats_dataset,
            batch_size=config["BATCH_SIZE"] * 2,
            shuffle=False,
            num_workers=config["NUM_WORKERS_TRAIN"],
        )
    )

    train_dirs, val_dirs = get_grouped_stratified_split(config)
    class_weights = calculate_class_weights(
        SegmentationDataset(train_dirs, get_cpu_transforms_kornia(config), config),
        config
    )

    cache = {
        "dataset_version": dataset_sig,
        "mean": mean,
        "std": std,
        "train_dirs": train_dirs,
        "val_dirs": val_dirs,
        "class_weights": class_weights.cpu(),
    }
    torch.save(cache, CACHE_PATH)
    print(f"Cache salvata in {CACHE_PATH}")
    return cache


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

def get_train_transforms(config: Dict[str, Any], aggressive_p: float = 0.5) -> A.Compose:
    """Pipeline di augmentation AGGRESSIVA per risolvere le confusioni specifiche."""
    
    output_size = config["OUTPUT_SIZE"]
    # mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225] # image net
    mean, std = config["MEAN"], config["STD"]
    return A.Compose([
        # A.HorizontalFlip(p=0.5),
        # A.VerticalFlip(p=aggressive_p/2),
        # A.RandomRotate90(p=aggressive_p),
        # A.RandomBrightnessContrast(p=0.7),
        # A.RandomGamma(p=0.7),
        # A.ColorJitter(p=0.7),
        # A.GaussianBlur(blur_limit=(3, 5), p=0.4),
        # # A.CoarseDropout(
        # #     max_holes=16, 
        # #     min_holes=8,
        # #     max_height=32,
        # #     min_height=16,
        # #     max_width=32, 
        # #     min_width=16,
        # #     p=0.5
        # # ),
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






from kornia.constants import DataKey, Resample

def get_train_transforms_kornia(config, aggressive_p=0.5):
    output_size = config["OUTPUT_SIZE"]
    mean, std = config["MEAN"], config["STD"]

    return K.AugmentationSequential(
        # --- augmentations ---
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=aggressive_p / 2),
        K.RandomRotation(degrees=180, p=aggressive_p),
        K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.7),
        K.RandomGaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2.0), p=0.4),
        K.Resize(size=output_size),
        K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),

        data_keys=["input", "mask"],
        extra_args={
            DataKey.MASK: {          # <-- unica chiave, vale per TUTTI gli op
                "resample": Resample.NEAREST,
                "align_corners": False,
            }
        },
        same_on_batch=False,
    )


def get_val_transforms_kornia(config: Dict[str, Any]) -> torch.nn.Sequential:
    """Pipeline di validazione Kornia (solo preprocessing)."""
    output_size = config["OUTPUT_SIZE"]
    mean, std = config["MEAN"], config["STD"]
    
    return K.AugmentationSequential(
        K.Resize(size=(output_size[0], output_size[1])),
        K.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        
        data_keys=['input', 'mask'],
        extra_args={
            K.Resize: {"interpolation": "nearest", "align_corners": None}
        },
        same_on_batch=False
    )

def get_cpu_transforms_kornia(config: Dict[str, Any]) -> torch.nn.Sequential:
    """Pipeline Kornia minimale per il calcolo delle statistiche (solo resize)."""
    output_size = config["OUTPUT_SIZE"]

    return K.AugmentationSequential(
        K.Resize(size=(output_size[0], output_size[1])),
        
        data_keys=['input', 'mask'],
        extra_args={
            K.Resize: {"interpolation": "nearest", "align_corners": None}
        },
        same_on_batch=False
    )




# SOSTITUZIONE DEFINITIVA
class SegmentationDataset(Dataset):
    def __init__(self, sample_dirs: List[str], transform: nn.Module, config: Dict[str, Any]):
        self.sample_dirs = sample_dirs
        self.transform = transform
        self.config = config

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        
        image_np = np.array(Image.open(os.path.join(sample_dir, 'rgb.jpg')).convert("RGB"))
        mask_np = np.array(Image.open(os.path.join(sample_dir, 'labels.png')))
        
        assert mask_np.dtype == np.uint8, f"Tipo maschera non corretto: {mask_np.dtype}"
        assert mask_np.max() < self.config["NUM_CLASSES"], f"ID classe non valido in {os.path.join(sample_dir, 'labels.png')}"

        # Immagine a tensore (C, H, W), float [0, 1]
        image_tensor = image_to_tensor(image_np).float() / 255.0
        
        # Maschera a tensore (H, W), long
        mask_tensor = torch.from_numpy(mask_np.copy()).long()

        # Applica le trasformazioni Kornia.
        # Input: immagine (C,H,W), maschera (H,W)
        # Output di Kornia: immagine (1,C,H,W), maschera (1,1,H,W)
        aug_image, aug_mask = self.transform(image_tensor, mask_tensor)
        
        # --- LA CORREZIONE È QUI ---
        # Rimuovi la dimensione batch dall'immagine e TUTTE le dimensioni
        # singleton (batch E canale) dalla maschera.
        # L'output deve essere (C, H, W) per l'immagine e (H, W) per la maschera.
        return aug_image.squeeze(0), aug_mask.squeeze()
    



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

def train_one_epoch(model, loader, optimizer, scheduler, loss_function, scaler, device, use_amp, current_epoch, total_epochs, ignore_index=None) -> float:
    model.train(); running_loss = 0.0; progress_bar = tqdm(loader, desc="Training", leave=False)
    for step, (images, masks) in enumerate(progress_bar):
        # Applica CutMix con una certa probabilità
        if current_epoch <= total_epochs - 5 and torch.rand(1) < 0.5: # es. 50% di probabilità per batch
            images, masks = cutmix_batch(images, masks, alpha=1.0)

        images, masks = images.to(device), masks.to(device, dtype=torch.long)
        with autocast(enabled=use_amp, device_type=device.type if use_amp else 'cpu'):
            outputs = model(images); 
            feats = model.encoder(images)[-1]
            loss = loss_function(outputs, masks, feats)

        optimizer.zero_grad(); scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update(); 
        if scheduler is not None:
            scheduler.step()

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
    config = get_config(); 
    cache  = load_or_compute_preprocessing(config)

    # sovrascrivi i campi in config così il resto dello script non cambia
    config["MEAN"], config["STD"]   = cache["mean"], cache["std"]
    train_dirs, val_dirs            = cache["train_dirs"], cache["val_dirs"]
    class_weights                   = cache["class_weights"].to(config["DEVICE"])
    assert 0.0 <= config["MEAN"].min() and config["MEAN"].max() <= 1.0,  "Mean fuori range 0-1"
    assert 0.0 <  config["STD"].min()  and config["STD"].max()  <  1.0,  "Std anomala"
    
    
    seed_everything(config["RANDOM_SEED"]); device = config["DEVICE"]
    use_amp = config["USE_AMP"] and AMP_AVAILABLE and device.type == 'cuda'
    tqdm.write(f"Device: {device} | Mixed Precision: {use_amp}"); tqdm.write(f"Strategia Finale: {config['ENCODER']} con Augmentation Mirata")

    # train_dirs, val_dirs = get_grouped_stratified_split(config)
    # Pipeline di augmentation diverse per training e validazione
    train_transforms = get_train_transforms_kornia(config, aggressive_p=0.1) 
    val_transforms = get_val_transforms_kornia(config)
    
    train_dataset = SegmentationDataset(train_dirs, train_transforms, config)
    val_dataset = SegmentationDataset(val_dirs, val_transforms, config)
    
    pin_mem = device.type == 'cuda'
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS_TRAIN"], pin_memory=pin_mem, persistent_workers=config["PERSISTENT_WORKERS"])
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS_VAL"], pin_memory=pin_mem, persistent_workers=config["PERSISTENT_WORKERS"])

    model = get_model(config).to(device)
    scaler = GradScaler(enabled=use_amp)
    # Calcola i pesi per la loss sul training set (la funzione va definita sopra)
    # class_weights = calculate_class_weights(train_dataset, config)

    # criterion_tversky = TverskyLoss(mode='multiclass', alpha=0.7, beta=0.3).to(device)
    # criterion_ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=config["IGNORE_INDEX"])
    # criterion_focal = FocalLoss(mode='multiclass', gamma=2.0).to(device)
    # criterion_dice = DiceLoss(mode='multiclass', ignore_index=config["IGNORE_INDEX"]).to(device)
    
    pixel_contrast = PixelContrastiveLoss(
        temperature=config["CONTRASTIVE_TEMP"],
        max_samples=config["CONTRASTIVE_MAX_SAMPLES"],
        ignore_index=config["IGNORE_INDEX"]
    ).to(device)

    # Aggiungiamo la Lovasz
    # criterion_lovasz = LovaszLoss(mode='multiclass',ignore_index=config["IGNORE_INDEX"]).to(device)

    criterion_ft  = FocalTverskyLoss(
                    alpha=0.9, beta=0.1, gamma=3.0,
                    class_weights=None,
                    ignore_index=config["IGNORE_INDEX"]
                ).to(device)

    criterion_wce = nn.CrossEntropyLoss(weight=class_weights,ignore_index=config["IGNORE_INDEX"]).to(device)

    def combined_loss(pred, target, feats=None):
        return 0.8 * criterion_ft(pred, target) + 0.2 * criterion_wce(pred, target)

    best_iou = -1.0; best_conf_matrix = None; global_epoch = 0

    # --- FASE 1: FREEZE ENCODER, TRAIN DECODER ---
    tqdm.write("\n--- INIZIO FASE 1: TRAINING DEL DECODER (ENCODER CONGELATO) ---")
    for param in model.encoder.parameters(): param.requires_grad = False
    optimizer = optim.AdamW(list(model.decoder.parameters()) + list(model.segmentation_head.parameters()), lr=config["LR_HEAD"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader),
        epochs=config["UNFREEZE_EPOCHS"], pct_start=0.1, div_factor=10, final_div_factor=1e3
    )

    for epoch in range(config["FREEZE_EPOCHS"]):
        global_epoch += 1; tqdm.write(f"\n--- Fase 1 / Epoch {epoch+1}/{config['FREEZE_EPOCHS']} (Global: {global_epoch}) ---")
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scheduler=scheduler, loss_function=combined_loss, 
            scaler=scaler, device=device, use_amp=use_amp, current_epoch=epoch, total_epochs=config["FREEZE_EPOCHS"]
        )
        val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate(model, val_loader, combined_loss, device, config)
        # scheduler.step(val_miou)
        tqdm.write(f"Risultati Epoch {global_epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        if val_miou > best_iou:
            best_iou, best_conf_matrix = val_miou, val_conf_matrix
            torch.save({'model_state_dict': model.state_dict(), 'config': config}, config["MODEL_SAVE_PATH"])
            tqdm.write(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<")

    # # --- FASE 2: UNFREEZE ALL, FINE-TUNE ---
    tqdm.write("\n--- INIZIO FASE 2: FINE-TUNING DELL'INTERA RETE (ENCODER SCONGELATO) ---")
    train_loader.dataset.transform = get_train_transforms_kornia(config, aggressive_p=0.5) 

    for param in model.encoder.parameters(): param.requires_grad = True
    optimizer = optim.AdamW(model.parameters(), lr=config["LR_FULL"]) #, weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.5, cooldown=1, min_lr=1e-7)
            

    for epoch in range(config["UNFREEZE_EPOCHS"]):
        global_epoch += 1; tqdm.write(f"\n--- Fase 2 / Epoch {epoch+1}/{config['UNFREEZE_EPOCHS']} (Global: {global_epoch}) ---")
        
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scheduler=None, loss_function=combined_loss, 
            scaler=scaler, device=device, use_amp=use_amp, current_epoch=epoch, total_epochs=config["UNFREEZE_EPOCHS"]
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

    
    # --- FASE 3: FINE-TUNING FINALE ---
    tqdm.write("\n--- INIZIO FASE 3: FINE-TUNING FINALE ---")
    
    def combined_loss(pred, target, feats=None):
        return 0.2 * criterion_ft(pred, target) + 0.8 * lovasz_softmax(F.softmax(pred, dim=1), target, ignore=config["IGNORE_INDEX"]) 


    for epoch in range(config["FINE_TUNING_EPOCHS"]):
        global_epoch += 1; tqdm.write(f"\n--- Fase 3 / Epoch {epoch+1}/{config['FINE_TUNING_EPOCHS']} (Global: {global_epoch}) ---")
        
        train_loss = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scheduler=None, loss_function=combined_loss, 
            scaler=scaler, device=device, use_amp=use_amp, current_epoch=epoch, total_epochs=config["FINE_TUNING_EPOCHS"]
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

r'''
> & "C:/Program Files/Python312/python.exe" "d:/MLPoject-Git/autoencoder-onboard-segmentation/local_execution/Leo/old/7 - All_in_1.py"
Cache trovata e valida — salto il preprocessing.
Device: cuda | Mixed Precision: True
Strategia Finale: efficientnet-b3 con Augmentation Mirata
Creazione del modello: U-Net con encoder 'efficientnet-b3'
Parametri addestrabili: 13,160,193

--- INIZIO FASE 1: TRAINING DEL DECODER (ENCODER CONGELATO) ---

--- Fase 1 / Epoch 1/7 (Global: 1) ---
Risultati Epoch 1: Train Loss: 0.8126 | Val Loss: 0.6978 | Val mIoU: 0.3129                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3129 <<<

--- Fase 1 / Epoch 2/7 (Global: 2) ---
Risultati Epoch 2: Train Loss: 0.6904 | Val Loss: 0.6477 | Val mIoU: 0.3557                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3557 <<<

--- Fase 1 / Epoch 3/7 (Global: 3) ---
Risultati Epoch 3: Train Loss: 0.6613 | Val Loss: 0.6583 | Val mIoU: 0.3522                                                                                        

--- Fase 1 / Epoch 4/7 (Global: 4) ---
Risultati Epoch 4: Train Loss: 0.6271 | Val Loss: 0.6181 | Val mIoU: 0.4245                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4245 <<<

--- Fase 1 / Epoch 5/7 (Global: 5) ---
Risultati Epoch 5: Train Loss: 0.6112 | Val Loss: 0.6067 | Val mIoU: 0.4254                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4254 <<<

--- Fase 1 / Epoch 6/7 (Global: 6) ---
Risultati Epoch 6: Train Loss: 0.5944 | Val Loss: 0.5909 | Val mIoU: 0.4230                                                                                        

--- Fase 1 / Epoch 7/7 (Global: 7) ---
Risultati Epoch 7: Train Loss: 0.5765 | Val Loss: 0.5692 | Val mIoU: 0.4463                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4463 <<<

--- INIZIO FASE 2: FINE-TUNING DELL'INTERA RETE (ENCODER SCONGELATO) ---

--- Fase 2 / Epoch 1/23 (Global: 8) ---
Risultati Epoch 8: Train Loss: 0.5545 | Val Loss: 0.5389 | Val mIoU: 0.4916                                                                                        
    - Classe 1 (Classe 1): 0.3852
    - Classe 2 (Classe 2): 0.3030
    - Classe 3 (Classe 3): 0.4848
    - Classe 4 (Classe 4): 0.3970
    - Classe 5 (Classe 5): 0.6011
    - Classe 6 (Classe 6): 0.2869
    - Classe 7 (Classe 7): 0.6002
    - Classe 8 (Classe 8): 0.8745

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.4916 <<<

--- Fase 2 / Epoch 2/23 (Global: 9) ---
Risultati Epoch 9: Train Loss: 0.5207 | Val Loss: 0.5205 | Val mIoU: 0.5124                                                                                        
    - Classe 1 (Classe 1): 0.4108
    - Classe 2 (Classe 2): 0.3236
    - Classe 3 (Classe 3): 0.4999
    - Classe 4 (Classe 4): 0.4441
    - Classe 5 (Classe 5): 0.6072
    - Classe 6 (Classe 6): 0.2975
    - Classe 7 (Classe 7): 0.6216
    - Classe 8 (Classe 8): 0.8947

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5124 <<<

--- Fase 2 / Epoch 3/23 (Global: 10) ---
Risultati Epoch 10: Train Loss: 0.4924 | Val Loss: 0.5019 | Val mIoU: 0.5425                                                                                       
    - Classe 1 (Classe 1): 0.5074
    - Classe 2 (Classe 2): 0.4027
    - Classe 3 (Classe 3): 0.4884
    - Classe 4 (Classe 4): 0.5088
    - Classe 5 (Classe 5): 0.6561
    - Classe 6 (Classe 6): 0.2953
    - Classe 7 (Classe 7): 0.5931
    - Classe 8 (Classe 8): 0.8883

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5425 <<<

--- Fase 2 / Epoch 4/23 (Global: 11) ---
Risultati Epoch 11: Train Loss: 0.4754 | Val Loss: 0.4888 | Val mIoU: 0.5590                                                                                       
    - Classe 1 (Classe 1): 0.4851
    - Classe 2 (Classe 2): 0.3964
    - Classe 3 (Classe 3): 0.5362
    - Classe 4 (Classe 4): 0.4845
    - Classe 5 (Classe 5): 0.6891
    - Classe 6 (Classe 6): 0.3328
    - Classe 7 (Classe 7): 0.6595
    - Classe 8 (Classe 8): 0.8880

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5590 <<<

--- Fase 2 / Epoch 5/23 (Global: 12) ---
Risultati Epoch 12: Train Loss: 0.4611 | Val Loss: 0.4802 | Val mIoU: 0.5540                                                                                       
    - Classe 1 (Classe 1): 0.5211
    - Classe 2 (Classe 2): 0.3690
    - Classe 3 (Classe 3): 0.5253
    - Classe 4 (Classe 4): 0.5263
    - Classe 5 (Classe 5): 0.6824
    - Classe 6 (Classe 6): 0.3114
    - Classe 7 (Classe 7): 0.6081
    - Classe 8 (Classe 8): 0.8881

--- Fase 2 / Epoch 6/23 (Global: 13) ---
Risultati Epoch 13: Train Loss: 0.4392 | Val Loss: 0.4813 | Val mIoU: 0.5531                                                                                       
    - Classe 1 (Classe 1): 0.4637
    - Classe 2 (Classe 2): 0.3821
    - Classe 3 (Classe 3): 0.5457
    - Classe 4 (Classe 4): 0.4057
    - Classe 5 (Classe 5): 0.7445
    - Classe 6 (Classe 6): 0.3441
    - Classe 7 (Classe 7): 0.6633
    - Classe 8 (Classe 8): 0.8758

--- Fase 2 / Epoch 7/23 (Global: 14) ---
Risultati Epoch 14: Train Loss: 0.3864 | Val Loss: 0.3734 | Val mIoU: 0.5999                                                                                       
    - Classe 1 (Classe 1): 0.5676
    - Classe 2 (Classe 2): 0.4121
    - Classe 3 (Classe 3): 0.5119
    - Classe 4 (Classe 4): 0.6089
    - Classe 5 (Classe 5): 0.7196
    - Classe 6 (Classe 6): 0.3656
    - Classe 7 (Classe 7): 0.7198
    - Classe 8 (Classe 8): 0.8938

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.5999 <<<

--- Fase 2 / Epoch 8/23 (Global: 15) ---
Risultati Epoch 15: Train Loss: 0.3362 | Val Loss: 0.3755 | Val mIoU: 0.6058                                                                                       
    - Classe 1 (Classe 1): 0.5632
    - Classe 2 (Classe 2): 0.4277
    - Classe 3 (Classe 3): 0.5028
    - Classe 4 (Classe 4): 0.5909
    - Classe 5 (Classe 5): 0.7585
    - Classe 6 (Classe 6): 0.3766
    - Classe 7 (Classe 7): 0.7387
    - Classe 8 (Classe 8): 0.8884

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.6058 <<<

--- Fase 2 / Epoch 9/23 (Global: 16) ---
Risultati Epoch 16: Train Loss: 0.3262 | Val Loss: 0.3861 | Val mIoU: 0.5897                                                                                       
    - Classe 1 (Classe 1): 0.5631
    - Classe 2 (Classe 2): 0.4628
    - Classe 3 (Classe 3): 0.4848
    - Classe 4 (Classe 4): 0.4905
    - Classe 5 (Classe 5): 0.7305
    - Classe 6 (Classe 6): 0.3620
    - Classe 7 (Classe 7): 0.7312
    - Classe 8 (Classe 8): 0.8926

--- Fase 2 / Epoch 10/23 (Global: 17) ---
Risultati Epoch 17: Train Loss: 0.3180 | Val Loss: 0.3693 | Val mIoU: 0.6121                                                                                       
    - Classe 1 (Classe 1): 0.5828
    - Classe 2 (Classe 2): 0.4268
    - Classe 3 (Classe 3): 0.5661
    - Classe 4 (Classe 4): 0.5582
    - Classe 5 (Classe 5): 0.7805
    - Classe 6 (Classe 6): 0.3846
    - Classe 7 (Classe 7): 0.7192
    - Classe 8 (Classe 8): 0.8788

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.6121 <<<

--- Fase 2 / Epoch 11/23 (Global: 18) ---
Risultati Epoch 18: Train Loss: 0.3066 | Val Loss: 0.3823 | Val mIoU: 0.6176                                                                                       
    - Classe 1 (Classe 1): 0.5874
    - Classe 2 (Classe 2): 0.4764
    - Classe 3 (Classe 3): 0.5490
    - Classe 4 (Classe 4): 0.6109
    - Classe 5 (Classe 5): 0.7240
    - Classe 6 (Classe 6): 0.3747
    - Classe 7 (Classe 7): 0.7331
    - Classe 8 (Classe 8): 0.8853

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.6176 <<<

--- Fase 2 / Epoch 12/23 (Global: 19) ---
Risultati Epoch 19: Train Loss: 0.2991 | Val Loss: 0.3709 | Val mIoU: 0.6333                                                                                       
    - Classe 1 (Classe 1): 0.5915
    - Classe 2 (Classe 2): 0.4725
    - Classe 3 (Classe 3): 0.5676
    - Classe 4 (Classe 4): 0.6028
    - Classe 5 (Classe 5): 0.7845
    - Classe 6 (Classe 6): 0.3999
    - Classe 7 (Classe 7): 0.7612
    - Classe 8 (Classe 8): 0.8867

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.6333 <<<

--- Fase 2 / Epoch 13/23 (Global: 20) ---
Risultati Epoch 20: Train Loss: 0.2953 | Val Loss: 0.3666 | Val mIoU: 0.6344                                                                                       
    - Classe 1 (Classe 1): 0.6110
    - Classe 2 (Classe 2): 0.4740
    - Classe 3 (Classe 3): 0.6012
    - Classe 4 (Classe 4): 0.6246
    - Classe 5 (Classe 5): 0.7842
    - Classe 6 (Classe 6): 0.3603
    - Classe 7 (Classe 7): 0.7319
    - Classe 8 (Classe 8): 0.8878

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.6344 <<<

--- Fase 2 / Epoch 14/23 (Global: 21) ---
Risultati Epoch 21: Train Loss: 0.2868 | Val Loss: 0.3927 | Val mIoU: 0.6280                                                                                       
    - Classe 1 (Classe 1): 0.6080
    - Classe 2 (Classe 2): 0.4779
    - Classe 3 (Classe 3): 0.5496
    - Classe 4 (Classe 4): 0.6214
    - Classe 5 (Classe 5): 0.7904
    - Classe 6 (Classe 6): 0.3476
    - Classe 7 (Classe 7): 0.7546
    - Classe 8 (Classe 8): 0.8748

--- Fase 2 / Epoch 15/23 (Global: 22) ---
Risultati Epoch 22: Train Loss: 0.2849 | Val Loss: 0.3817 | Val mIoU: 0.6522                                                                                       
    - Classe 1 (Classe 1): 0.6225
    - Classe 2 (Classe 2): 0.5072
    - Classe 3 (Classe 3): 0.6008
    - Classe 4 (Classe 4): 0.6897
    - Classe 5 (Classe 5): 0.7817
    - Classe 6 (Classe 6): 0.3635
    - Classe 7 (Classe 7): 0.7581
    - Classe 8 (Classe 8): 0.8944

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.6522 <<<

--- Fase 2 / Epoch 16/23 (Global: 23) ---
Risultati Epoch 23: Train Loss: 0.2866 | Val Loss: 0.3710 | Val mIoU: 0.6298                                                                                       
    - Classe 1 (Classe 1): 0.6041
    - Classe 2 (Classe 2): 0.4939
    - Classe 3 (Classe 3): 0.5304
    - Classe 4 (Classe 4): 0.5919
    - Classe 5 (Classe 5): 0.7912
    - Classe 6 (Classe 6): 0.4013
    - Classe 7 (Classe 7): 0.7429
    - Classe 8 (Classe 8): 0.8825

--- Fase 2 / Epoch 17/23 (Global: 24) ---
Risultati Epoch 24: Train Loss: 0.2791 | Val Loss: 0.3744 | Val mIoU: 0.6441                                                                                       
    - Classe 1 (Classe 1): 0.6131
    - Classe 2 (Classe 2): 0.5050
    - Classe 3 (Classe 3): 0.5656
    - Classe 4 (Classe 4): 0.6542
    - Classe 5 (Classe 5): 0.7877
    - Classe 6 (Classe 6): 0.3832
    - Classe 7 (Classe 7): 0.7518
    - Classe 8 (Classe 8): 0.8924

--- Fase 2 / Epoch 18/23 (Global: 25) ---
Risultati Epoch 25: Train Loss: 0.2774 | Val Loss: 0.3609 | Val mIoU: 0.6622                                                                                       
    - Classe 1 (Classe 1): 0.6200
    - Classe 2 (Classe 2): 0.5087
    - Classe 3 (Classe 3): 0.6042
    - Classe 4 (Classe 4): 0.7136
    - Classe 5 (Classe 5): 0.8024
    - Classe 6 (Classe 6): 0.4193
    - Classe 7 (Classe 7): 0.7482
    - Classe 8 (Classe 8): 0.8812

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.6622 <<<

--- Fase 2 / Epoch 19/23 (Global: 26) ---
Risultati Epoch 26: Train Loss: 0.2700 | Val Loss: 0.3876 | Val mIoU: 0.6416                                                                                       
    - Classe 1 (Classe 1): 0.6018
    - Classe 2 (Classe 2): 0.4885
    - Classe 3 (Classe 3): 0.5357
    - Classe 4 (Classe 4): 0.6660
    - Classe 5 (Classe 5): 0.7900
    - Classe 6 (Classe 6): 0.3891
    - Classe 7 (Classe 7): 0.7737
    - Classe 8 (Classe 8): 0.8877

--- Fase 2 / Epoch 20/23 (Global: 27) ---
Risultati Epoch 27: Train Loss: 0.2453 | Val Loss: 0.3777 | Val mIoU: 0.6555                                                                                       
    - Classe 1 (Classe 1): 0.6399
    - Classe 2 (Classe 2): 0.5039
    - Classe 3 (Classe 3): 0.5960
    - Classe 4 (Classe 4): 0.6443
    - Classe 5 (Classe 5): 0.7987
    - Classe 6 (Classe 6): 0.4068
    - Classe 7 (Classe 7): 0.7763
    - Classe 8 (Classe 8): 0.8784

--- Fase 2 / Epoch 21/23 (Global: 28) ---
Risultati Epoch 28: Train Loss: 0.2453 | Val Loss: 0.3987 | Val mIoU: 0.6487                                                                                       
    - Classe 1 (Classe 1): 0.5967
    - Classe 2 (Classe 2): 0.5259
    - Classe 3 (Classe 3): 0.4840
    - Classe 4 (Classe 4): 0.6953
    - Classe 5 (Classe 5): 0.8004
    - Classe 6 (Classe 6): 0.4289
    - Classe 7 (Classe 7): 0.7755
    - Classe 8 (Classe 8): 0.8832

--- Fase 2 / Epoch 22/23 (Global: 29) ---
Risultati Epoch 29: Train Loss: 0.2473 | Val Loss: 0.3791 | Val mIoU: 0.6583                                                                                       
    - Classe 1 (Classe 1): 0.6586
    - Classe 2 (Classe 2): 0.5169
    - Classe 3 (Classe 3): 0.6188
    - Classe 4 (Classe 4): 0.6527
    - Classe 5 (Classe 5): 0.7839
    - Classe 6 (Classe 6): 0.3872
    - Classe 7 (Classe 7): 0.7602
    - Classe 8 (Classe 8): 0.8883

--- Fase 2 / Epoch 23/23 (Global: 30) ---
Risultati Epoch 30: Train Loss: 0.2380 | Val Loss: 0.3834 | Val mIoU: 0.6486                                                                                       
    - Classe 1 (Classe 1): 0.6235
    - Classe 2 (Classe 2): 0.5125
    - Classe 3 (Classe 3): 0.5601
    - Classe 4 (Classe 4): 0.6375
    - Classe 5 (Classe 5): 0.7879
    - Classe 6 (Classe 6): 0.4040
    - Classe 7 (Classe 7): 0.7713
    - Classe 8 (Classe 8): 0.8923

--- INIZIO FASE 3: FINE-TUNING FINALE ---

--- Fase 3 / Epoch 1/10 (Global: 31) ---
Risultati Epoch 31: Train Loss: 0.3114 | Val Loss: 0.3381 | Val mIoU: 0.6851                                                                                       
    - Classe 1 (Classe 1): 0.6487
    - Classe 2 (Classe 2): 0.5404
    - Classe 3 (Classe 3): 0.6044
    - Classe 4 (Classe 4): 0.7858
    - Classe 5 (Classe 5): 0.8291
    - Classe 6 (Classe 6): 0.3705
    - Classe 7 (Classe 7): 0.7961
    - Classe 8 (Classe 8): 0.9061

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.6851 <<<

--- Fase 3 / Epoch 2/10 (Global: 32) ---
Risultati Epoch 32: Train Loss: 0.2987 | Val Loss: 0.3356 | Val mIoU: 0.6821                                                                                       
    - Classe 1 (Classe 1): 0.6501
    - Classe 2 (Classe 2): 0.5400
    - Classe 3 (Classe 3): 0.5921
    - Classe 4 (Classe 4): 0.7937
    - Classe 5 (Classe 5): 0.8080
    - Classe 6 (Classe 6): 0.3578
    - Classe 7 (Classe 7): 0.8057
    - Classe 8 (Classe 8): 0.9096

--- Fase 3 / Epoch 3/10 (Global: 33) ---
Risultati Epoch 33: Train Loss: 0.2963 | Val Loss: 0.3384 | Val mIoU: 0.6800                                                                                       
    - Classe 1 (Classe 1): 0.6468
    - Classe 2 (Classe 2): 0.5261
    - Classe 3 (Classe 3): 0.6155
    - Classe 4 (Classe 4): 0.7770
    - Classe 5 (Classe 5): 0.8134
    - Classe 6 (Classe 6): 0.3489
    - Classe 7 (Classe 7): 0.8017
    - Classe 8 (Classe 8): 0.9107

--- Fase 3 / Epoch 4/10 (Global: 34) ---
Risultati Epoch 34: Train Loss: 0.2807 | Val Loss: 0.3299 | Val mIoU: 0.6928                                                                                       
    - Classe 1 (Classe 1): 0.6624
    - Classe 2 (Classe 2): 0.5376
    - Classe 3 (Classe 3): 0.6265
    - Classe 4 (Classe 4): 0.8236
    - Classe 5 (Classe 5): 0.8241
    - Classe 6 (Classe 6): 0.3504
    - Classe 7 (Classe 7): 0.8069
    - Classe 8 (Classe 8): 0.9109

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.6928 <<<

--- Fase 3 / Epoch 5/10 (Global: 35) ---
Risultati Epoch 35: Train Loss: 0.2803 | Val Loss: 0.3265 | Val mIoU: 0.6855                                                                                       
    - Classe 1 (Classe 1): 0.6552
    - Classe 2 (Classe 2): 0.5349
    - Classe 3 (Classe 3): 0.6116
    - Classe 4 (Classe 4): 0.7916
    - Classe 5 (Classe 5): 0.8326
    - Classe 6 (Classe 6): 0.3378
    - Classe 7 (Classe 7): 0.8068
    - Classe 8 (Classe 8): 0.9132

--- Fase 3 / Epoch 6/10 (Global: 36) ---
Risultati Epoch 36: Train Loss: 0.2750 | Val Loss: 0.3265 | Val mIoU: 0.6858                                                                                       
    - Classe 1 (Classe 1): 0.6596
    - Classe 2 (Classe 2): 0.5356
    - Classe 3 (Classe 3): 0.5989
    - Classe 4 (Classe 4): 0.8003
    - Classe 5 (Classe 5): 0.8105
    - Classe 6 (Classe 6): 0.3652
    - Classe 7 (Classe 7): 0.8022
    - Classe 8 (Classe 8): 0.9139

--- Fase 3 / Epoch 7/10 (Global: 37) ---
Risultati Epoch 37: Train Loss: 0.2556 | Val Loss: 0.3286 | Val mIoU: 0.6752                                                                                       
    - Classe 1 (Classe 1): 0.6391
    - Classe 2 (Classe 2): 0.5332
    - Classe 3 (Classe 3): 0.5674
    - Classe 4 (Classe 4): 0.7887
    - Classe 5 (Classe 5): 0.8264
    - Classe 6 (Classe 6): 0.3265
    - Classe 7 (Classe 7): 0.8061
    - Classe 8 (Classe 8): 0.9140

--- Fase 3 / Epoch 8/10 (Global: 38) ---
Risultati Epoch 38: Train Loss: 0.2494 | Val Loss: 0.3202 | Val mIoU: 0.6926                                                                                       
    - Classe 1 (Classe 1): 0.6736
    - Classe 2 (Classe 2): 0.5301
    - Classe 3 (Classe 3): 0.6455
    - Classe 4 (Classe 4): 0.8047
    - Classe 5 (Classe 5): 0.8260
    - Classe 6 (Classe 6): 0.3455
    - Classe 7 (Classe 7): 0.7998
    - Classe 8 (Classe 8): 0.9152

--- Fase 3 / Epoch 9/10 (Global: 39) ---
Risultati Epoch 39: Train Loss: 0.2449 | Val Loss: 0.3201 | Val mIoU: 0.6897                                                                                       
    - Classe 1 (Classe 1): 0.6614
    - Classe 2 (Classe 2): 0.5323
    - Classe 3 (Classe 3): 0.6443
    - Classe 4 (Classe 4): 0.8213
    - Classe 5 (Classe 5): 0.8293
    - Classe 6 (Classe 6): 0.3179
    - Classe 7 (Classe 7): 0.7970
    - Classe 8 (Classe 8): 0.9140

--- Fase 3 / Epoch 10/10 (Global: 40) ---
Risultati Epoch 40: Train Loss: 0.2412 | Val Loss: 0.3153 | Val mIoU: 0.6964                                                                                       
    - Classe 1 (Classe 1): 0.6703
    - Classe 2 (Classe 2): 0.5386
    - Classe 3 (Classe 3): 0.6449
    - Classe 4 (Classe 4): 0.8017
    - Classe 5 (Classe 5): 0.8320
    - Classe 6 (Classe 6): 0.3663
    - Classe 7 (Classe 7): 0.8018
    - Classe 8 (Classe 8): 0.9156

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.6964 <<<

--- Training Completato ---
Miglior mIoU di validazione ottenuto: 0.6964
Visualizzazione e salvataggio Matrice di Confusione...
Matrice di confusione salvata in 'confusion_matrix_best.png'

'''