# =============================================================================
# 1. IMPORTS
# =============================================================================
import os
import sys
import glob
from typing import Tuple, List, Dict, Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imagehash
import networkx as nx
from segmentation_models_pytorch.losses import FocalLoss 
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import WeightedRandomSampler

# =============================================================================
# 2. CONFIGURAZIONE
# =============================================================================
def get_config() -> Dict[str, Any]:
    """Restituisce un dizionario con tutte le configurazioni dello script."""
    # ... (gestione percorso base invariata) ...
    base_path = "./"

    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_finetuned_model.pth"),

        # Parametri del dataset e split (stabili, hanno funzionato bene)
        "OUTPUT_SIZE": (384, 768),
        "VALIDATION_SPLIT": 0.2,
        "RANDOM_SEED": 42,
        "RARE_CLASS_INDICES": [4, 5, 6],
        "NUM_WORKERS": 0,
        "NUM_WORKERS_TRAIN": 2,
        "NUM_WORKERS_VAL": 1,
        "RARE_BOOST_FACTOR": 5.0, # Moltiplicatore per il peso delle classi rare
        
        # Parametri del modello
        "ENCODER": "efficientnet-b3",
        "ENCODER_WEIGHTS": "imagenet",
        "NUM_CLASSES": 9,
        "IGNORE_INDEX": 0,

        # --- IPERPARAMETRI OTTIMIZZATI PER COMBATTERE L'OVERFITTING ---
        "BATCH_SIZE": 8,          # Aumentiamo un po' se la memoria lo consente
        "USE_AMP": True,
        
        # Parametri per il training a due fasi
        "NUM_EPOCHS": 30,        # Epoche per addestrare tutta la rete
        "LEARNING_RATE": 1e-4,      # Learning rate molto basso per il fine-tuning
        "WEIGHT_DECAY": 1e-5,     # Regolarizzazione L2 aumentata
    }

    config["CLASS_NAMES"] = []
    for i in range(config["NUM_CLASSES"]):
        if i == config["IGNORE_INDEX"]:
            config["CLASS_NAMES"].append("Ignore")
        else:
            config["CLASS_NAMES"].append(f"Class {i}")
    return config

class SegmentationDataset(Dataset):
    def __init__(self, sample_dirs: List[str], transform: A.Compose):
        self.sample_dirs = sample_dirs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        img_path = os.path.join(sample_dir, 'rgb.jpg')
        mask_path = os.path.join(sample_dir, 'labels.png')
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']
    
# =============================================================================
# 3. DATA AUGMENTATION -> ORA SOLO PREPROCESSING
# =============================================================================
def get_transforms(output_size: Tuple[int, int]) -> A.Compose:
    """
    Restituisce la pipeline di trasformazioni base (solo preprocessing).
    L'augmentation on-the-fly è stata rimossa, poiché i dati vengono già caricati
    da cartelle pre-aumentate.
    """
    mean=(0.0017, 0.0018, 0.0017)
    std=(0.0010, 0.0010, 0.0010)
    
    # MODIFICA: Rimossa la logica condizionale is_train.
    # Questa pipeline ora si applica sia al training che alla validazione.
    # Esegue solo le operazioni necessarie: resize, normalizzazione e conversione a tensore.
    return A.Compose([
        A.Resize(height=output_size[0], width=output_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

# =============================================================================
# 4. FUNZIONI HELPER (SPLIT E METRICHE)
# =============================================================================
def get_grouped_stratified_split(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    # Percorso al dataset di training originale
    dataset_path_train = config["DATASET_PATH_TRAIN"]

    # Percorso al dataset aumentato
    dataset_path_aug = config["DATASET_PATH_AUG"]

    # Percorso al dataset aumentato aggressivamente
    dataset_path_aug_aggr = config["DATASET_PATH_AUG_AGGR"]

    print("Caricamento percorsi dei dataset...")

    # Ottieni la lista delle directory originali
    original_dirs = [
        d for d in glob.glob(os.path.join(dataset_path_train, "*")) 
        if os.path.isdir(d) and not os.path.basename(d).startswith('__')
    ]
    print(f"Trovati {len(original_dirs)} campioni originali.")

    # Ottieni la lista delle directory aumentate
    augmented_dirs = [
        d for d in glob.glob(os.path.join(dataset_path_aug, "*")) 
        if os.path.isdir(d) and not os.path.basename(d).startswith('__')
    ]
    print(f"Trovati {len(augmented_dirs)} campioni aumentati.")

    # Ottieni la lista delle directory aumentate aggressive
    augmented_aggr_dirs = [
        d for d in glob.glob(os.path.join(dataset_path_aug_aggr, "*")) 
        if os.path.isdir(d) and not os.path.basename(d).startswith('__')
    ]
    print(f"Trovati {len(augmented_aggr_dirs)} campioni aumentati aggressivamente.")

    # Unisci le due liste in un'unica lista
    all_dirs = original_dirs + augmented_dirs + augmented_aggr_dirs
    print(f"Dimensione totale del dataset di training (originale + aumentato): {len(all_dirs)} campioni.")

    # Opzionale ma consigliato: ordina la lista finale per riproducibilità
    all_dirs = sorted(all_dirs)
    
    hashes = {path: imagehash.phash(Image.open(os.path.join(path, 'rgb.jpg'))) for path in tqdm(all_dirs, desc="Calcolo Hash", leave=False)}
    
    G = nx.Graph()
    G.add_nodes_from(hashes.keys())
    paths = list(hashes.keys())
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hashes[paths[i]] - hashes[paths[j]] < 5:
                G.add_edge(paths[i], paths[j])
                
    image_groups = list(nx.connected_components(G))
    print(f"Trovati {len(image_groups)} gruppi di immagini.")
    
    group_labels = []
    for group in tqdm(image_groups, desc="Etichettatura gruppi", leave=False):
        has_rare = any(
            rare_id in np.unique(Image.open(os.path.join(p, 'labels.png')))
            for p in group for rare_id in config["RARE_CLASS_INDICES"]
        )
        group_labels.append(1 if has_rare else 0)
        
    train_groups, val_groups = train_test_split(
        image_groups, test_size=config["VALIDATION_SPLIT"],
        random_state=config["RANDOM_SEED"], stratify=group_labels
    )
    
    train_dirs = [p for group in train_groups for p in group]
    val_dirs = [p for group in val_groups for p in group]
    
    print(f"Split stratificato basato su gruppi creato: {len(train_dirs)} train, {len(val_dirs)} validation.")
    return train_dirs, val_dirs


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
    
    for idx in config["RARE_CLASS_INDICES"]:
        weights[idx] *= config["RARE_BOOST_FACTOR"]
        
    weights = weights / weights.mean() # Normalizza i pesi
    weights_tensor = torch.tensor(weights, dtype=torch.float).to(config["DEVICE"])
    
    print("\nPesi calcolati per la loss (con boost per classi rare):")
    for i, w in enumerate(weights_tensor):
        print(f"  - Classe {i}: {w.item():.2f}")
    return weights_tensor

def compute_iou(preds, labels, num_classes, ignore_index):
    iou_per_class = np.zeros(num_classes)
    for cls in range(num_classes):
        if cls == ignore_index:
            continue
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            iou_per_class[cls] = np.nan # La classe non era presente
        else:
            iou_per_class[cls] = intersection / union
    return iou_per_class


def get_model(config: Dict[str, Any]) -> nn.Module:
    """
    Crea e restituisce il modello di segmentazione U-Net con un encoder pre-addestrato.
    
    Args:
        config (Dict): Dizionario di configurazione contenente i parametri del modello.
        
    Returns:
        nn.Module: Il modello PyTorch istanziato.
    """
    print(f"Creazione del modello: U-Net con encoder '{config['ENCODER']}'")
    
    model = smp.Unet(
        encoder_name=config["ENCODER"],
        encoder_weights=config["ENCODER_WEIGHTS"],
        in_channels=3,  # Canali di input (RGB)
        classes=config["NUM_CLASSES"],
    )
    
    # Stampa il numero di parametri per avere un'idea della complessità del modello
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri addestrabili del modello: {total_params:,}")
    
    return model

class CombinedLoss(nn.Module):
    def __init__(self, ce_weight_factor=0.5, dice_weight_factor=0.5, class_weights=None, ignore_index=0):
        super().__init__()
        self.ce_weight_factor = ce_weight_factor
        self.dice_weight_factor = dice_weight_factor
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index)
        self.dice = DiceLoss(mode='multiclass', from_logits=True)

    def forward(self, pred, target):
        return (self.ce_weight_factor * self.ce(pred, target)) + \
               (self.dice_weight_factor * self.dice(pred, target))
    
    
def compute_iou_per_class(preds_np: np.ndarray, labels_np: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    """Calcola l'IoU per ogni classe."""
    iou_scores = np.zeros(num_classes)
    for cls in range(num_classes):
        if cls == ignore_index:
            iou_scores[cls] = np.nan # Ignoriamo esplicitamente questa classe
            continue
            
        pred_inds = (preds_np == cls)
        target_inds = (labels_np == cls)
        
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        
        if union == 0:
            # La classe non era presente né nella predizione né nella GT
            iou_scores[cls] = np.nan 
        else:
            iou_scores[cls] = intersection / union
            
    return iou_scores


def create_colored_mask_from_indices(mask_np: np.ndarray, num_classes: int) -> Image.Image:
    """
    Funzione helper per creare un'immagine PIL colorata da una maschera di indici.
    Utile per visualizzare le predizioni del modello.
    """
    # Usa una colormap standard e riproducibile
    cmap = plt.get_cmap('nipy_spectral', num_classes)
    # Normalizza gli indici per la colormap
    norm = plt.Normalize(vmin=0, vmax=num_classes - 1)
    # Applica la colormap per ottenere un'immagine RGBA
    colored_mask_rgba = cmap(norm(mask_np))
    # Converti in RGB 8-bit
    colored_mask_rgb = (colored_mask_rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_mask_rgb)


def show_best_worst_predictions(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    config: Dict[str, Any]
):
    """
    Esegue il modello sul validation set, trova le predizioni migliori e peggiori
    in base all'mIoU per immagine e le visualizza.
    """
    print("\n--- Analisi Qualitativa: Ricerca Migliore e Peggiore Predizione ---")
    model.eval()
    
    results = [] # Lista per salvare (miou, immagine, gt, pred) per ogni campione

    with torch.no_grad():
        for images, masks_gt in tqdm(val_loader, desc="Analisi qualitativa", leave=False):
            images, masks_gt = images.to(device), masks_gt.to(device)
            
            # Ottieni le predizioni del modello
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            
            # Sposta i dati sulla CPU per l'analisi e la visualizzazione
            images_cpu = images.cpu()
            masks_gt_cpu = masks_gt.cpu().numpy()
            preds_cpu = preds.cpu().numpy()
            
            # Calcola l'mIoU per ogni immagine nel batch
            for i in range(images.size(0)):
                iou_per_class = compute_iou_per_class(
                    preds_cpu[i],
                    masks_gt_cpu[i],
                    config["NUM_CLASSES"],
                    config["IGNORE_INDEX"]
                )
                # mIoU per questa singola immagine (ignorando le classi non presenti)
                miou_image = np.nanmean(iou_per_class)
                
                # De-normalizza l'immagine per la visualizzazione
                # NOTA: Assicurati che le medie e std usate qui siano le stesse
                # usate nella tua funzione get_transforms!
                mean = np.array([0.0017, 0.0018, 0.0017])
                std = np.array([0.0010, 0.0010, 0.0010])
                img_denorm = images_cpu[i].permute(1, 2, 0).numpy()
                img_denorm = (img_denorm * std + mean)
                # Applica il clipping per assicurarsi che i valori siano in [0, 1]
                img_denorm = np.clip(img_denorm, 0, 1)

                results.append({
                    "miou": miou_image,
                    "image": img_denorm,
                    "ground_truth": masks_gt_cpu[i],
                    "prediction": preds_cpu[i]
                })

    # Ordina i risultati in base all'mIoU
    results.sort(key=lambda x: x["miou"])
    
    # Prendi il peggiore (primo) e il migliore (ultimo)
    worst_pred = results[0]
    best_pred = results[-1]
    
    # --- Visualizzazione ---
    
    # Prepara le maschere per la visualizzazione
    # Usa la palette originale della GT dove possibile, e una generata per la predizione
    gt_best_viz = create_colored_mask_from_indices(best_pred["ground_truth"], config["NUM_CLASSES"])
    pred_best_viz = create_colored_mask_from_indices(best_pred["prediction"], config["NUM_CLASSES"])
    
    gt_worst_viz = create_colored_mask_from_indices(worst_pred["ground_truth"], config["NUM_CLASSES"])
    pred_worst_viz = create_colored_mask_from_indices(worst_pred["prediction"], config["NUM_CLASSES"])

    # Plot della Migliore Predizione
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Migliore Predizione (mIoU: {best_pred['miou']:.4f})", fontsize=16)
    axs[0].imshow(best_pred["image"])
    axs[0].set_title("Immagine Originale")
    axs[0].axis('off')
    axs[1].imshow(gt_best_viz)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')
    axs[2].imshow(pred_best_viz)
    axs[2].set_title("Predizione del Modello")
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()
    
    # Plot della Peggiore Predizione
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f"Peggiore Predizione (mIoU: {worst_pred['miou']:.4f})", fontsize=16)
    axs[0].imshow(worst_pred["image"])
    axs[0].set_title("Immagine Originale")
    axs[0].axis('off')
    axs[1].imshow(gt_worst_viz)
    axs[1].set_title("Ground Truth")
    axs[1].axis('off')
    axs[2].imshow(pred_worst_viz)
    axs[2].set_title("Predizione del Modello")
    axs[2].axis('off')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(
    model: nn.Module, 
    loader: DataLoader, 
    device: torch.device, 
    config: Dict[str, Any],
    class_names: List[str] = None
):
    """
    Calcola e visualizza la matrice di confusione normalizzata per il modello.

    Args:
        model (nn.Module): Il modello addestrato da valutare.
        loader (DataLoader): Il DataLoader contenente i dati di validazione o test.
        device (torch.device): Il device su cui eseguire i calcoli (es. 'cuda').
        config (Dict): Il dizionario di configurazione.
        class_names (List[str], optional): Nomi delle classi per le etichette del grafico.
    """
    print("\n--- Calcolo della Matrice di Confusione ---")
    model.eval()
    
    num_classes = config["NUM_CLASSES"]
    ignore_index = config.get("IGNORE_INDEX", None) # Usa .get per sicurezza

    # Inizializza la matrice di confusione su CPU o GPU
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Generazione Matrice di Confusione", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            
            with torch.amp.autocast(enabled=True, device_type=device.type):
                outputs = model(images)
            
            preds = torch.argmax(outputs, dim=1)
            
            # Appiattiamo le maschere e le predizioni per un calcolo efficiente
            masks_flat = masks.flatten()
            preds_flat = preds.flatten()

            # Rimuoviamo i pixel da ignorare
            if ignore_index is not None:
                valid_indices = masks_flat != ignore_index
                masks_flat = masks_flat[valid_indices]
                preds_flat = preds_flat[valid_indices]
            
            # Calcolo degli indici per la matrice di confusione.
            # Questo è un trucco per aggiornare la matrice senza cicli for in Python.
            # Ogni coppia (true_class, pred_class) viene mappata a un indice univoco.
            indices = masks_flat * num_classes + preds_flat
            
            # Usiamo bincount per contare le occorrenze di ogni coppia
            counts = torch.bincount(indices, minlength=num_classes**2)
            
            # Aggiungiamo i conteggi del batch alla matrice complessiva
            conf_matrix += counts.reshape(num_classes, num_classes)

    # Sposta la matrice finale sulla CPU e convertila in NumPy
    conf_matrix_np = conf_matrix.cpu().numpy()

    # --- Preparazione per la visualizzazione ---
    
    # Rimuovi la riga/colonna dell'ignore_index se esiste, per una visualizzazione più pulita
    plot_matrix = conf_matrix_np
    plot_class_names = class_names

    if ignore_index is not None:
        plot_matrix = np.delete(plot_matrix, ignore_index, axis=0) # Rimuovi riga
        plot_matrix = np.delete(plot_matrix, ignore_index, axis=1) # Rimuovi colonna
        if plot_class_names:
            plot_class_names = [name for i, name in enumerate(class_names) if i != ignore_index]

    # Normalizza la matrice per riga per ottenere le percentuali (recall per ogni classe)
    # L'asse y (true label) mostrerà dove vanno a finire i pixel di quella classe.
    row_sums = plot_matrix.sum(axis=1)[:, np.newaxis]
    # Evita la divisione per zero se una classe non è mai presente nel set di validazione
    norm_conf_matrix = np.divide(plot_matrix, row_sums, out=np.zeros_like(plot_matrix, dtype=float), where=row_sums!=0)

    # --- Creazione del Grafico ---
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        norm_conf_matrix, 
        annot=True,          # Mostra i valori numerici nelle celle
        fmt=".2%",           # Formatta i valori come percentuali
        cmap="Blues",        # Scegli una colormap (es. 'Blues', 'viridis', 'coolwarm')
        xticklabels=plot_class_names or "auto",
        yticklabels=plot_class_names or "auto"
    )
    plt.title("Matrice di Confusione Normalizzata (per Riga)")
    plt.ylabel("Classe Vera (Ground Truth)")
    plt.xlabel("Classe Predetta dal Modello")
    plt.tight_layout()
    plt.show()
    


def create_class_balanced_sampler(
    sample_dirs: List[str], 
    config: Dict[str, Any]
) -> WeightedRandomSampler:
    """
    Crea un sampler che bilancia le classi campionando più spesso le immagini
    che contengono le classi più rare.
    """
    print("\nCreazione del Sampler Bilanciato...")
    
    # 1. Etichetta ogni immagine con la sua classe più rara
    print("  - Step 1: Analisi del contenuto di ogni maschera...")
    labels = []
    class_counts = np.zeros(config["NUM_CLASSES"])
    
    for path in tqdm(sample_dirs, desc="Etichettatura immagini", leave=False):
        mask_path = os.path.join(path, 'labels.png')
        mask = np.array(Image.open(mask_path))
        unique_classes = np.unique(mask)
        
        # Ignora la classe 0 (background/ignore) per la decisione
        unique_classes = unique_classes[unique_classes != config["IGNORE_INDEX"]]
        
        if len(unique_classes) == 0:
            # Se un'immagine contiene solo la classe ignore, le diamo un'etichetta speciale
            # o la trattiamo come una classe comune (es. la classe più comune del dataset)
            labels.append(7) # Assumiamo la classe 7 come default comune
            class_counts[7] += 1
            continue
            
        # Trova la classe più rara presente in questa immagine
        # (usiamo gli indici delle classi rare come proxy per la rarità)
        rarest_class_in_image = -1
        # Partiamo dalle classi più rare definite nella config
        for rare_idx in sorted(config["RARE_CLASS_INDICES"]): 
            if rare_idx in unique_classes:
                rarest_class_in_image = rare_idx
                break # Trovata la più rara, usciamo
        
        if rarest_class_in_image == -1:
            # Se non ci sono classi rare, prendiamo una delle classi comuni
            # (es. la più numerosa tra quelle presenti)
            rarest_class_in_image = unique_classes[0] # Semplice euristica
        
        labels.append(rarest_class_in_image)
        class_counts[rarest_class_in_image] += 1
        
    print("  - Step 2: Calcolo dei pesi per classe...")
    # Calcola il peso per ogni classe come l'inverso della sua frequenza
    class_weights = 1.0 / (class_counts + 1e-6) # Aggiungi epsilon per evitare divisione per zero
    
    # Applica il boost extra per le classi che ci interessano di più
    for idx in config["RARE_CLASS_INDICES"]:
        class_weights[idx] *= config["RARE_BOOST_FACTOR"]

    print("  - Step 3: Assegnazione del peso a ogni campione...")
    # Assegna a ogni campione il peso della sua classe-etichetta
    sample_weights = [class_weights[label] for label in labels]
    
    # 4. Crea e restituisci il sampler
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True # Permette di campionare la stessa immagine più volte in un'epoca
    )
    print("Sampler Bilanciato creato con successo.")
    return sampler


# =============================================================================
# 5. FUNZIONI DI TRAINING E VALIDAZIONE
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, scheduler, device, use_amp):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc="Training", leave=False)
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device, dtype=torch.long)
        
        with torch.amp.autocast(enabled=use_amp, device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        
        running_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")
        
    return running_loss / len(loader)

def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, 
             device: torch.device, num_classes: int, ignore_index: int) -> Tuple[float, float, np.ndarray]:
    """Valuta il modello, restituendo loss, mIoU e IoU per classe."""
    model.eval()
    total_loss = 0.0
    # Usiamo una lista per accumulare gli array di IoU per ogni immagine
    iou_list_per_image = [] 
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            
            with torch.amp.autocast(enabled=True, device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            
            # Calcola IoU per ogni immagine nel batch
            for i in range(images.size(0)):
                iou_scores = compute_iou_per_class(
                    preds[i].cpu().numpy(), 
                    masks[i].cpu().numpy(), 
                    num_classes,
                    ignore_index
                )
                iou_list_per_image.append(iou_scores)
    
    avg_loss = total_loss / len(loader)
    
    # Calcola la media dell'IoU per ogni classe su tutte le immagini
    iou_per_class = np.nanmean(np.array(iou_list_per_image), axis=0)
    
    # Il mIoU è la media degli IoU delle classi (ignorando i NaN delle classi assenti)
    mean_iou = np.nanmean(iou_per_class)
    
    return avg_loss, mean_iou, iou_per_class

# =============================================================================
# 6. FUNZIONE PRINCIPALE (MAIN - STRATEGIA BILANCIATA)
# =============================================================================
def main():
    """
    Funzione principale che orchestra il processo di addestramento.
    Questa versione implementa una strategia bilanciata per trovare una baseline robusta.
    """
    
    # 1. Carica la configurazione e imposta il device
    config = get_config()
    device = config["DEVICE"]
    use_amp = config["USE_AMP"] and device.type == 'cuda'
    print(f"Device: {device} | Mixed Precision: {use_amp}")

    # 2. Prepara i dati (split stratificato e grouped)
    # Questa funzione deve essere definita sopra, come nel tuo script
    train_dirs, val_dirs = get_grouped_stratified_split(config)
    
    # MODIFICA: Prepara le trasformazioni di base (solo preprocessing)
    # La stessa pipeline viene usata per training e validazione.
    base_transforms = get_transforms(output_size=config["OUTPUT_SIZE"])

    # Crea i Dataset
    train_dataset = SegmentationDataset(train_dirs, base_transforms)
    val_dataset = SegmentationDataset(val_dirs, base_transforms)
    
    # Crea il sampler SOLO per il training set!
    train_sampler = create_class_balanced_sampler(train_dirs, config)

    # Crea i DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["BATCH_SIZE"], 
        sampler=train_sampler,  # Usa il sampler bilanciato
        shuffle=False,
        num_workers=config["NUM_WORKERS_TRAIN"], 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["BATCH_SIZE"], 
        shuffle=False,
        num_workers=config["NUM_WORKERS_VAL"], 
        pin_memory=True
    )

    # 3. Prepara il Modello, la Loss, l'Ottimizzatore e lo Scheduler
    
    # Modello
    model = get_model(config).to(device)
    
    # DiceLoss è la nostra arma principale contro lo sbilanciamento
    criterion_dice = DiceLoss(mode='multiclass', from_logits=True) 
    
    # FocalLoss si concentra sui pixel difficili. Gamma=2 è un buon default.
    criterion_focal = FocalLoss(mode='multiclass', gamma=2)
    
    # Li combiniamo
    def combined_loss(pred, target):
        return 0.7 * criterion_dice(pred, target) + 0.3 * criterion_focal(pred, target)
    
    # Ottimizzatore
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    
    # Scheduler per adattare il learning rate
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
      optimizer, 
      max_lr=1e-3, 
      steps_per_epoch=len(train_loader), 
      epochs=config["NUM_EPOCHS"]
    )
    
    # Scaler per la Mixed Precision
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # 4. Ciclo di Training e Validazione
    
    best_iou = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
    
    print("\n--- Inizio Addestramento (Strategia Bilanciata) ---")
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        
        # Le funzioni train_one_epoch e evaluate vanno definite sopra
        train_loss = train_one_epoch(model, train_loader, optimizer, combined_loss, scaler, scheduler, device, use_amp)
        val_loss, val_miou, val_iou_per_class = evaluate(model, val_loader, combined_loss, device, config["NUM_CLASSES"], config["IGNORE_INDEX"])
        
        # Aggiorna lo scheduler con il mIoU di validazione
        # scheduler.step(val_miou)
        
        # Salva la storia per futuri grafici
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        
        # Stampa i risultati dell'epoca
        print(f"Risultati Epoch {epoch+1}:")
        print(f"  - Training Loss: {train_loss:.4f} | Validation Loss: {val_loss:.4f} | Validation mIoU: {val_miou:.4f}")
        print("  - IoU per Classe (Validation):")
        for i, iou in enumerate(val_iou_per_class):
            if not np.isnan(iou): # Stampa solo le classi presenti
                print(f"    - Classe {i}: {iou:.4f}")
        
        # Salva il modello se ha ottenuto il miglior mIoU finora
        if val_miou > best_iou:
            best_iou = val_miou
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<\n")

        # --- NUOVA SEZIONE: ANALISI QUALITATIVA FINALE ---
        # Carica il miglior modello salvato per l'analisi
        print("\nCaricamento del miglior modello per l'analisi finale...")
        model.load_state_dict(torch.load(config["MODEL_SAVE_PATH"]))
        
        # Mostra i risultati migliori e peggiori sul validation set
        show_best_worst_predictions(model, val_loader, device, config)

    print(f"\n--- Addestramento Completato ---")
    print(f"Miglior mIoU di validazione ottenuto: {best_iou:.4f}")
    
    # Qui potresti aggiungere codice per plottare i grafici dalla variabile 'history'

# =============================================================================
# 7. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()

'''
> & "C:/Program Files/Python312/python.exe" "d:/MLPoject-Git/autoencoder-onboard-segmentation/local_execution/3 - Loss/5 - single_phase_focal_loss.py"
Device: cuda | Mixed Precision: True
Caricamento percorsi dei dataset...
Trovati 931 campioni originali.
Trovati 561 campioni aumentati.
Trovati 1960 campioni aumentati aggressivamente.
Dimensione totale del dataset di training (originale + aumentato): 3452 campioni.
Trovati 2475 gruppi di immagini.                                                                                                                                                  
Split stratificato basato su gruppi creato: 2805 train, 647 validation.                                                                                                           
Creazione del modello: U-Net con encoder 'efficientnet-b2'
Parametri addestrabili del modello: 10,047,451

--- Inizio Addestramento (Strategia Bilanciata) ---

--- Epoch 1/30 ---
d:\MLPoject-Git\autoencoder-onboard-segmentation\local_execution\3 - Loss\5 - single_phase_focal_loss.py:344: RuntimeWarning: Mean of empty slice                                 
  iou_per_class = np.nanmean(np.array(iou_list_per_image), axis=0)
Risultati Epoch 1:
  - Training Loss: 1.0992 | Validation Loss: 0.7836 | Validation mIoU: 0.0133
  - IoU per Classe (Validation):
    - Classe 1: 0.0020
    - Classe 2: 0.0012
    - Classe 3: 0.0007
    - Classe 4: 0.0004
    - Classe 5: 0.0031
    - Classe 6: 0.0001
    - Classe 7: 0.0942
    - Classe 8: 0.0049

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.0133 <<<


--- Epoch 2/30 ---
Risultati Epoch 2:                                                                                                                                                                
  - Training Loss: 0.6867 | Validation Loss: 0.5679 | Validation mIoU: 0.0237
  - IoU per Classe (Validation):
    - Classe 1: 0.0015
    - Classe 2: 0.0054
    - Classe 3: 0.0009
    - Classe 4: 0.0000
    - Classe 5: 0.0000
    - Classe 6: 0.0004
    - Classe 7: 0.1796
    - Classe 8: 0.0017

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.0237 <<<


--- Epoch 3/30 ---
Risultati Epoch 3:                                                                                                                                                                
  - Training Loss: 0.5836 | Validation Loss: 0.5093 | Validation mIoU: 0.0913
  - IoU per Classe (Validation):
    - Classe 1: 0.0011
    - Classe 2: 0.0932
    - Classe 3: 0.0049
    - Classe 4: 0.0000
    - Classe 5: 0.0000
    - Classe 6: 0.0873
    - Classe 7: 0.2734
    - Classe 8: 0.2708

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.0913 <<<


--- Epoch 4/30 ---
Risultati Epoch 4:                                                                                                                                                                
  - Training Loss: 0.5408 | Validation Loss: 0.5083 | Validation mIoU: 0.1088
  - IoU per Classe (Validation):
    - Classe 1: 0.1180
    - Classe 2: 0.1416
    - Classe 3: 0.0195
    - Classe 4: 0.0000
    - Classe 5: 0.0001
    - Classe 6: 0.0223
    - Classe 7: 0.2499
    - Classe 8: 0.3194

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1088 <<<


--- Epoch 5/30 ---
Risultati Epoch 5:                                                                                                                                                                
  - Training Loss: 0.5285 | Validation Loss: 0.4869 | Validation mIoU: 0.1142
  - IoU per Classe (Validation):
    - Classe 1: 0.0495
    - Classe 2: 0.1295
    - Classe 3: 0.0075
    - Classe 4: 0.0000
    - Classe 5: 0.1123
    - Classe 6: 0.1084
    - Classe 7: 0.2002
    - Classe 8: 0.3065

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1142 <<<


--- Epoch 6/30 ---
Risultati Epoch 6:                                                                                                                                                                
  - Training Loss: 0.5068 | Validation Loss: 0.5186 | Validation mIoU: 0.1279
  - IoU per Classe (Validation):
    - Classe 1: 0.0727
    - Classe 2: 0.0452
    - Classe 3: 0.0424
    - Classe 4: 0.0000
    - Classe 5: 0.1843
    - Classe 6: 0.0799
    - Classe 7: 0.2623
    - Classe 8: 0.3359

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1279 <<<


--- Epoch 7/30 ---
Risultati Epoch 7:                                                                                                                                                                
  - Training Loss: 0.4935 | Validation Loss: 0.4703 | Validation mIoU: 0.1580
  - IoU per Classe (Validation):
    - Classe 1: 0.1836
    - Classe 2: 0.1706
    - Classe 3: 0.0813
    - Classe 4: 0.0000
    - Classe 5: 0.1611
    - Classe 6: 0.0785
    - Classe 7: 0.2289
    - Classe 8: 0.3597

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1580 <<<


--- Epoch 8/30 ---
Risultati Epoch 8:                                                                                                                                                                
  - Training Loss: 0.4827 | Validation Loss: 0.4264 | Validation mIoU: 0.1856
  - IoU per Classe (Validation):
    - Classe 1: 0.1529
    - Classe 2: 0.1943
    - Classe 3: 0.0983
    - Classe 4: 0.0404
    - Classe 5: 0.2629
    - Classe 6: 0.0587
    - Classe 7: 0.2820
    - Classe 8: 0.3957

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1856 <<<


--- Epoch 9/30 ---
Risultati Epoch 9:                                                                                                                                                                
  - Training Loss: 0.4631 | Validation Loss: 0.4549 | Validation mIoU: 0.1927
  - IoU per Classe (Validation):
    - Classe 1: 0.1616
    - Classe 2: 0.1731
    - Classe 3: 0.0516
    - Classe 4: 0.1168
    - Classe 5: 0.2551
    - Classe 6: 0.0541
    - Classe 7: 0.3285
    - Classe 8: 0.4007

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1927 <<<


--- Epoch 10/30 ---
Risultati Epoch 10:                                                                                                                                                               
  - Training Loss: 0.4457 | Validation Loss: 0.4187 | Validation mIoU: 0.1938
  - IoU per Classe (Validation):
    - Classe 1: 0.1021
    - Classe 2: 0.1059
    - Classe 3: 0.1366
    - Classe 4: 0.1219
    - Classe 5: 0.2702
    - Classe 6: 0.1321
    - Classe 7: 0.2886
    - Classe 8: 0.3931

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1938 <<<


--- Epoch 11/30 ---
Risultati Epoch 11:                                                                                                                                                               
  - Training Loss: 0.4281 | Validation Loss: 0.4465 | Validation mIoU: 0.1856
  - IoU per Classe (Validation):
    - Classe 1: 0.0702
    - Classe 2: 0.1130
    - Classe 3: 0.1266
    - Classe 4: 0.1487
    - Classe 5: 0.2035
    - Classe 6: 0.1190
    - Classe 7: 0.3289
    - Classe 8: 0.3752

--- Epoch 12/30 ---
Risultati Epoch 12:                                                                                                                                                               
  - Training Loss: 0.4164 | Validation Loss: 0.4221 | Validation mIoU: 0.2175
  - IoU per Classe (Validation):
    - Classe 1: 0.0782
    - Classe 2: 0.1335
    - Classe 3: 0.1535
    - Classe 4: 0.2787
    - Classe 5: 0.2279
    - Classe 6: 0.1070
    - Classe 7: 0.3259
    - Classe 8: 0.4351

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2175 <<<


--- Epoch 13/30 ---
Risultati Epoch 13:                                                                                                                                                               
  - Training Loss: 0.4006 | Validation Loss: 0.4474 | Validation mIoU: 0.2037
  - IoU per Classe (Validation):
    - Classe 1: 0.1400
    - Classe 2: 0.1509
    - Classe 3: 0.1346
    - Classe 4: 0.1717
    - Classe 5: 0.2695
    - Classe 6: 0.0872
    - Classe 7: 0.3461
    - Classe 8: 0.3297

--- Epoch 14/30 ---
Risultati Epoch 14:                                                                                                                                                               
  - Training Loss: 0.3973 | Validation Loss: 0.4351 | Validation mIoU: 0.2088
  - IoU per Classe (Validation):
    - Classe 1: 0.1045
    - Classe 2: 0.1468
    - Classe 3: 0.1448
    - Classe 4: 0.1867
    - Classe 5: 0.2956
    - Classe 6: 0.0804
    - Classe 7: 0.3241
    - Classe 8: 0.3871

--- Epoch 15/30 ---
Risultati Epoch 15:                                                                                                                                                               
  - Training Loss: 0.3873 | Validation Loss: 0.3950 | Validation mIoU: 0.2513
  - IoU per Classe (Validation):
    - Classe 1: 0.2008
    - Classe 2: 0.1706
    - Classe 3: 0.1182
    - Classe 4: 0.3041
    - Classe 5: 0.3071
    - Classe 6: 0.1030
    - Classe 7: 0.3646
    - Classe 8: 0.4420

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2513 <<<


--- Epoch 16/30 ---
Risultati Epoch 16:                                                                                                                                                               
  - Training Loss: 0.3670 | Validation Loss: 0.4369 | Validation mIoU: 0.2564
  - IoU per Classe (Validation):
    - Classe 1: 0.1000
    - Classe 2: 0.1946
    - Classe 3: 0.1663
    - Classe 4: 0.3848
    - Classe 5: 0.3277
    - Classe 6: 0.0834
    - Classe 7: 0.3775
    - Classe 8: 0.4172

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2564 <<<


--- Epoch 17/30 ---
Risultati Epoch 17:                                                                                                                                                               
  - Training Loss: 0.3600 | Validation Loss: 0.3839 | Validation mIoU: 0.2692
  - IoU per Classe (Validation):
    - Classe 1: 0.2087
    - Classe 2: 0.1767
    - Classe 3: 0.1455
    - Classe 4: 0.4044
    - Classe 5: 0.3411
    - Classe 6: 0.0950
    - Classe 7: 0.3655
    - Classe 8: 0.4167

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2692 <<<


--- Epoch 18/30 ---
Risultati Epoch 18:                                                                                                                                                               
  - Training Loss: 0.3413 | Validation Loss: 0.6341 | Validation mIoU: 0.2096
  - IoU per Classe (Validation):
    - Classe 1: 0.1492
    - Classe 2: 0.1639
    - Classe 3: 0.1331
    - Classe 4: 0.3322
    - Classe 5: 0.2953
    - Classe 6: 0.0250
    - Classe 7: 0.2564
    - Classe 8: 0.3217

--- Epoch 19/30 ---
Risultati Epoch 19:                                                                                                                                                               
  - Training Loss: 0.3477 | Validation Loss: 0.3936 | Validation mIoU: 0.3023
  - IoU per Classe (Validation):
    - Classe 1: 0.2381
    - Classe 2: 0.1995
    - Classe 3: 0.1515
    - Classe 4: 0.4561
    - Classe 5: 0.3764
    - Classe 6: 0.1035
    - Classe 7: 0.4243
    - Classe 8: 0.4690

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3023 <<<


--- Epoch 20/30 ---
Risultati Epoch 20:                                                                                                                                                               
  - Training Loss: 0.3221 | Validation Loss: 0.3823 | Validation mIoU: 0.3008
  - IoU per Classe (Validation):
    - Classe 1: 0.2324
    - Classe 2: 0.1958
    - Classe 3: 0.1546
    - Classe 4: 0.4435
    - Classe 5: 0.3876
    - Classe 6: 0.1065
    - Classe 7: 0.4124
    - Classe 8: 0.4736

--- Epoch 21/30 ---
Risultati Epoch 21:                                                                                                                                                               
  - Training Loss: 0.3098 | Validation Loss: 0.3890 | Validation mIoU: 0.2931
  - IoU per Classe (Validation):
    - Classe 1: 0.2203
    - Classe 2: 0.2194
    - Classe 3: 0.1620
    - Classe 4: 0.3934
    - Classe 5: 0.3875
    - Classe 6: 0.1110
    - Classe 7: 0.4046
    - Classe 8: 0.4467

--- Epoch 22/30 ---
Risultati Epoch 22:                                                                                                                                                               
  - Training Loss: 0.2953 | Validation Loss: 0.3960 | Validation mIoU: 0.3217
  - IoU per Classe (Validation):
    - Classe 1: 0.2685
    - Classe 2: 0.2291
    - Classe 3: 0.1821
    - Classe 4: 0.4855
    - Classe 5: 0.3874
    - Classe 6: 0.1229
    - Classe 7: 0.4431
    - Classe 8: 0.4551

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3217 <<<


--- Epoch 23/30 ---
Risultati Epoch 23:                                                                                                                                                               
  - Training Loss: 0.2811 | Validation Loss: 0.3872 | Validation mIoU: 0.3321
  - IoU per Classe (Validation):
    - Classe 1: 0.2750
    - Classe 2: 0.2366
    - Classe 3: 0.1727
    - Classe 4: 0.5253
    - Classe 5: 0.4129
    - Classe 6: 0.1124
    - Classe 7: 0.4456
    - Classe 8: 0.4767

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3321 <<<


--- Epoch 24/30 ---
Risultati Epoch 24:                                                                                                                                                               
  - Training Loss: 0.2709 | Validation Loss: 0.3758 | Validation mIoU: 0.3308
  - IoU per Classe (Validation):
    - Classe 1: 0.2757
    - Classe 2: 0.2515
    - Classe 3: 0.1847
    - Classe 4: 0.4505
    - Classe 5: 0.4352
    - Classe 6: 0.1105
    - Classe 7: 0.4615
    - Classe 8: 0.4763

--- Epoch 25/30 ---
Risultati Epoch 25:                                                                                                                                                               
  - Training Loss: 0.2606 | Validation Loss: 0.3869 | Validation mIoU: 0.3483
  - IoU per Classe (Validation):
    - Classe 1: 0.2831
    - Classe 2: 0.2481
    - Classe 3: 0.1975
    - Classe 4: 0.5553
    - Classe 5: 0.4512
    - Classe 6: 0.1057
    - Classe 7: 0.4467
    - Classe 8: 0.4984

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3483 <<<


--- Epoch 26/30 ---
Risultati Epoch 26:                                                                                                                                                               
  - Training Loss: 0.2507 | Validation Loss: 0.3843 | Validation mIoU: 0.3446
  - IoU per Classe (Validation):
    - Classe 1: 0.2915
    - Classe 2: 0.2554
    - Classe 3: 0.2040
    - Classe 4: 0.5027
    - Classe 5: 0.4460
    - Classe 6: 0.1196
    - Classe 7: 0.4525
    - Classe 8: 0.4850

--- Epoch 27/30 ---
Risultati Epoch 27:                                                                                                                                                               
  - Training Loss: 0.2453 | Validation Loss: 0.3806 | Validation mIoU: 0.3497
  - IoU per Classe (Validation):
    - Classe 1: 0.2874
    - Classe 2: 0.2432
    - Classe 3: 0.2005
    - Classe 4: 0.5559
    - Classe 5: 0.4512
    - Classe 6: 0.1148
    - Classe 7: 0.4493
    - Classe 8: 0.4958

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3497 <<<


--- Epoch 28/30 ---
Risultati Epoch 28:                                                                                                                                                               
  - Training Loss: 0.2391 | Validation Loss: 0.3961 | Validation mIoU: 0.3536
  - IoU per Classe (Validation):
    - Classe 1: 0.3008
    - Classe 2: 0.2505
    - Classe 3: 0.2013
    - Classe 4: 0.5534
    - Classe 5: 0.4558
    - Classe 6: 0.1148
    - Classe 7: 0.4529
    - Classe 8: 0.4993

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3536 <<<


--- Epoch 29/30 ---
Risultati Epoch 29:                                                                                                                                                               
  - Training Loss: 0.2383 | Validation Loss: 0.3916 | Validation mIoU: 0.3517
  - IoU per Classe (Validation):
    - Classe 1: 0.3025
    - Classe 2: 0.2505
    - Classe 3: 0.2026
    - Classe 4: 0.5393
    - Classe 5: 0.4572
    - Classe 6: 0.1159
    - Classe 7: 0.4471
    - Classe 8: 0.4982

--- Epoch 30/30 ---
Risultati Epoch 30:                                                                                                                                                               
  - Training Loss: 0.2372 | Validation Loss: 0.3990 | Validation mIoU: 0.3503
  - IoU per Classe (Validation):
    - Classe 1: 0.3007
    - Classe 2: 0.2532
    - Classe 3: 0.2011
    - Classe 4: 0.5356
    - Classe 5: 0.4480
    - Classe 6: 0.1199
    - Classe 7: 0.4493
    - Classe 8: 0.4944

--- Addestramento Completato ---
Miglior mIoU di validazione ottenuto: 0.3536

'''