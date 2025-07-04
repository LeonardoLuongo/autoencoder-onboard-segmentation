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
from torch.amp import GradScaler, autocast

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss
import albumentations as A
from albumentations.pytorch import ToTensorV2
import imagehash
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import WeightedRandomSampler

# =============================================================================
# 2. CONFIGURAZIONE E MAPPA COLORI
# =============================================================================

# La nostra "Stele di Rosetta": mappa da colori RGB a ID di classe numerici.
COLOR_TO_CLASS_ID = {
    (255, 255, 255): 0,  # Bianco -> background/veicolo (CLASSE DA IGNORARE)
    (1, 88, 255): 1,     # Blu -> sky
    (255, 0, 128): 2,    # Magenta -> rough trail
    (178, 176, 153): 3,  # Grigio -> smooth trail
    (128, 255, 0): 4,    # Verde Lime -> traversable grass
    (40, 80, 0): 5,      # Verde Scuro -> high vegetation
    (0, 160, 0): 6,      # Verde -> low vegetation
    (156, 76, 30): 7,    # Marrone -> "puddle"/terreno con detriti
    (255, 0, 0): 8,      # Rosso -> obstacle
}

def get_config() -> Dict[str, Any]:
    """Restituisce un dizionario con tutte le configurazioni dello script."""
    base_path = "./"

    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "DATASET_PATH_TRAIN": os.path.join(base_path, "dataset/training/train"),
        "DATASET_PATH_AUG": os.path.join(base_path, "dataset/training/augmented"),
        "DATASET_PATH_AUG_AGGR": os.path.join(base_path, "dataset/training/augmented_aggressive"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_model_with_color_map.pth"),

        "OUTPUT_SIZE": (384, 768),
        "VALIDATION_SPLIT": 0.2,
        "RANDOM_SEED": 42,
        "RARE_CLASS_INDICES": [2, 4, 5, 8], # rough trail, trav. grass, high veg, obstacle
        "NUM_WORKERS_TRAIN": 2,
        "NUM_WORKERS_VAL": 1,
        "RARE_BOOST_FACTOR": 5.0,
        
        "ENCODER": "efficientnet-b3",
        "ENCODER_WEIGHTS": "imagenet",
        "NUM_CLASSES": 9,
        "IGNORE_INDEX": 0, # Corrisponde a (255, 255, 255)

        "BATCH_SIZE": 8,
        "USE_AMP": True,
        "NUM_EPOCHS": 30,
        "LEARNING_RATE": 1e-4,
        "WEIGHT_DECAY": 1e-5,
    }

    config["CLASS_NAMES"] = [
        "Vehicle/Ignore", "Sky", "Rough Trail", "Smooth Trail", 
        "Traversable Grass", "High Vegetation", "Low Vegetation", 
        "Debris/Puddle", "Obstacle"
    ]
    return config

# =============================================================================
# 3. CONVERSIONE MASCHERE E DATASET
# =============================================================================

def rgb_to_class_id_mask(mask_rgb: np.ndarray, color_map: Dict[Tuple[int, int, int], int]) -> np.ndarray:
    """
    Converte una maschera RGB (H, W, 3) in una maschera di ID di classe (H, W).
    """
    # Crea una maschera vuota (con zeri) della stessa altezza e larghezza
    class_id_mask = np.zeros(mask_rgb.shape[:2], dtype=np.uint8)
    
    # Itera su ogni coppia colore-ID nella mappa
    for color, class_id in color_map.items():
        # Trova tutti i pixel che corrispondono al colore corrente
        # np.all(mask_rgb == color, axis=-1) crea una maschera booleana (True/False)
        matches = np.all(mask_rgb == np.array(color).reshape(1, 1, 3), axis=2)
        
        # Assegna l'ID di classe corretto in quelle posizioni
        class_id_mask[matches] = class_id
        
    return class_id_mask

def get_transforms(output_size: Tuple[int, int]) -> A.Compose:
    """
    Restituisce la pipeline di trasformazioni base (solo preprocessing).
    Per ora non usiamo augmentation on-the-fly, ma solo il preprocessing.
    """
    mean=(0.0017, 0.0018, 0.0017)
    std=(0.0010, 0.0010, 0.0010)
    
    return A.Compose([
        A.Resize(height=output_size[0], width=output_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])

class SegmentationDataset(Dataset):
    def __init__(self, sample_dirs: List[str], transform: A.Compose, color_map: Dict):
        self.sample_dirs = sample_dirs
        self.transform = transform
        self.color_map = color_map

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        img_path = os.path.join(sample_dir, 'rgb.jpg')
        mask_path = os.path.join(sample_dir, 'labels.png') # Questa è la maschera colorata
        
        image = np.array(Image.open(img_path).convert("RGB"))
        
        # --- MODIFICA CHIAVE ---
        # 1. Carica la maschera come immagine RGB
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        # 2. Convertila in una maschera di ID di classe
        mask_ids = rgb_to_class_id_mask(mask_rgb, self.color_map)
        
        # 3. Applica le trasformazioni all'immagine e alla nuova maschera di ID
        augmented = self.transform(image=image, mask=mask_ids)
        return augmented['image'], augmented['mask']
    
# =============================================================================
# 4. FUNZIONI HELPER (MODIFICATE PER USARE LA CONVERSIONE)
# =============================================================================

def get_grouped_stratified_split(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    # Unione dei vari dataset (logica invariata)
    all_dirs = (
        glob.glob(os.path.join(config["DATASET_PATH_TRAIN"], "*")) +
        glob.glob(os.path.join(config["DATASET_PATH_AUG"], "*")) +
        glob.glob(os.path.join(config["DATASET_PATH_AUG_AGGR"], "*"))
    )
    all_dirs = sorted([d for d in all_dirs if os.path.isdir(d) and not os.path.basename(d).startswith('__')])
    print(f"Dimensione totale del dataset di training: {len(all_dirs)} campioni.")

    # Grouping basato su hash (logica invariata)
    hashes = {path: imagehash.phash(Image.open(os.path.join(path, 'rgb.jpg'))) for path in tqdm(all_dirs, desc="Calcolo Hash")}
    G = nx.Graph()
    G.add_nodes_from(hashes.keys())
    paths = list(hashes.keys())
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if hashes[paths[i]] - hashes[paths[j]] < 5:
                G.add_edge(paths[i], paths[j])
    image_groups = list(nx.connected_components(G))
    print(f"Trovati {len(image_groups)} gruppi di immagini.")
    
    # Etichettatura dei gruppi basata sulla presenza di classi rare (logica invariata)
    group_labels = []
    for group in tqdm(image_groups, desc="Etichettatura gruppi con mappa colori"):
        has_rare = False
        for p in group:
            mask_rgb = np.array(Image.open(os.path.join(p, 'labels.png')).convert("RGB"))
            mask_ids = rgb_to_class_id_mask(mask_rgb, COLOR_TO_CLASS_ID)
            unique_classes = np.unique(mask_ids)
            
            if any(rare_id in unique_classes for rare_id in config["RARE_CLASS_INDICES"]):
                has_rare = True
                break
        group_labels.append(1 if has_rare else 0)
        
    # --- MODIFICA CHIAVE PER RISOLVERE L'ERRORE ---
    # Controlla se la stratificazione è possibile.
    # Calcola quanti campioni ci sono per ogni etichetta (0 e 1).
    unique_labels, counts = np.unique(group_labels, return_counts=True)
    
    # Se una delle classi ha meno di 2 campioni, la stratificazione fallirà.
    if np.any(counts < 2):
        print("\nATTENZIONE: La stratificazione non è possibile a causa di una classe con meno di 2 campioni.")
        print("Verrà eseguito uno split casuale semplice (non stratificato).")
        stratify_param = None # Disabilita la stratificazione
    else:
        print("\nStratificazione possibile. Verrà eseguito uno split stratificato.")
        stratify_param = group_labels # Usa le etichette per la stratificazione

    # Esegui lo split usando il parametro di stratificazione calcolato
    train_groups, val_groups = train_test_split(
        image_groups, 
        test_size=config["VALIDATION_SPLIT"],
        random_state=config["RANDOM_SEED"], 
        stratify=stratify_param # Può essere 'group_labels' o 'None'
    )
    # --- FINE MODIFICA ---
    
    train_dirs = [p for group in train_groups for p in group]
    val_dirs = [p for group in val_groups for p in group]
    
    print(f"Split creato: {len(train_dirs)} train, {len(val_dirs)} validation.")
    return train_dirs, val_dirs

def create_class_balanced_sampler(sample_dirs: List[str], config: Dict[str, Any]) -> WeightedRandomSampler:
    print("\nCreazione del Sampler Bilanciato con mappa colori...")
    labels = []
    class_counts = np.zeros(config["NUM_CLASSES"])
    
    for path in tqdm(sample_dirs, desc="Etichettatura immagini per sampler"):
        # --- MODIFICA CHIAVE ---
        mask_path = os.path.join(path, 'labels.png')
        mask_rgb = np.array(Image.open(mask_path).convert("RGB"))
        mask_ids = rgb_to_class_id_mask(mask_rgb, COLOR_TO_CLASS_ID)
        unique_classes = np.unique(mask_ids)
        
        unique_classes = unique_classes[unique_classes != config["IGNORE_INDEX"]]
        if len(unique_classes) == 0:
            labels.append(3) # Default a smooth trail, una classe comune
            class_counts[3] += 1
            continue
            
        rarest_class_in_image = -1
        for rare_idx in sorted(config["RARE_CLASS_INDICES"], reverse=True): # Dalla più rara
            if rare_idx in unique_classes:
                rarest_class_in_image = rare_idx
                break
        
        if rarest_class_in_image == -1:
            rarest_class_in_image = unique_classes[0] # Altrimenti una classe a caso
        
        labels.append(rarest_class_in_image)
        class_counts[rarest_class_in_image] += 1
        
    # Calcolo pesi (logica invariata)
    class_weights = 1.0 / (class_counts + 1e-6)
    for idx in config["RARE_CLASS_INDICES"]:
        class_weights[idx] *= config["RARE_BOOST_FACTOR"]
    sample_weights = [class_weights[label] for label in labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    print("Sampler Bilanciato creato.")
    return sampler

# Le seguenti funzioni helper (compute_iou, get_model, ecc.) non necessitano di modifiche
# poiché operano su maschere di ID già convertite o su tensori.
# (Codice delle altre funzioni helper da qui in poi...)
def get_model(config: Dict[str, Any]) -> nn.Module:
    print(f"Creazione del modello: U-Net con encoder '{config['ENCODER']}'")
    model = smp.Unet(
        encoder_name=config["ENCODER"],
        encoder_weights=config["ENCODER_WEIGHTS"],
        in_channels=3,
        classes=config["NUM_CLASSES"],
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parametri addestrabili del modello: {total_params:,}")
    return model

def compute_iou_per_class(preds_np: np.ndarray, labels_np: np.ndarray, num_classes: int, ignore_index: int) -> np.ndarray:
    iou_scores = np.zeros(num_classes)
    for cls in range(num_classes):
        if cls == ignore_index:
            iou_scores[cls] = np.nan
            continue
        pred_inds = (preds_np == cls)
        target_inds = (labels_np == cls)
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            iou_scores[cls] = np.nan 
        else:
            iou_scores[cls] = intersection / union
    return iou_scores

def create_colored_mask_from_indices(mask_np: np.ndarray, num_classes: int) -> Image.Image:
    cmap = plt.get_cmap('nipy_spectral', num_classes)
    norm = plt.Normalize(vmin=0, vmax=num_classes - 1)
    colored_mask_rgba = cmap(norm(mask_np))
    colored_mask_rgb = (colored_mask_rgba[:, :, :3] * 255).astype(np.uint8)
    return Image.fromarray(colored_mask_rgb)

def show_best_worst_predictions(model: nn.Module, val_loader: DataLoader, device: torch.device, config: Dict[str, Any]):
    print("\n--- Analisi Qualitativa: Ricerca Migliore e Peggiore Predizione ---")
    model.eval()
    results = []
    with torch.no_grad():
        for images, masks_gt in tqdm(val_loader, desc="Analisi qualitativa", leave=False):
            images, masks_gt = images.to(device), masks_gt.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            images_cpu, masks_gt_cpu, preds_cpu = images.cpu(), masks_gt.cpu().numpy(), preds.cpu().numpy()
            for i in range(images.size(0)):
                iou_per_class = compute_iou_per_class(preds_cpu[i], masks_gt_cpu[i], config["NUM_CLASSES"], config["IGNORE_INDEX"])
                miou_image = np.nanmean(iou_per_class)
                mean, std = np.array([0.0017, 0.0018, 0.0017]), np.array([0.0010, 0.0010, 0.0010])
                img_denorm = np.clip((images_cpu[i].permute(1, 2, 0).numpy() * std + mean), 0, 1)
                results.append({"miou": miou_image, "image": img_denorm, "ground_truth": masks_gt_cpu[i], "prediction": preds_cpu[i]})
    results.sort(key=lambda x: x["miou"])
    worst_pred, best_pred = results[0], results[-1]
    
    # Visualizzazione
    for pred_data, title_prefix in [(best_pred, "Migliore"), (worst_pred, "Peggiore")]:
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"{title_prefix} Predizione (mIoU: {pred_data['miou']:.4f})", fontsize=16)
        axs[0].imshow(pred_data["image"]); axs[0].set_title("Immagine Originale"); axs[0].axis('off')
        axs[1].imshow(create_colored_mask_from_indices(pred_data["ground_truth"], config["NUM_CLASSES"])); axs[1].set_title("Ground Truth"); axs[1].axis('off')
        axs[2].imshow(create_colored_mask_from_indices(pred_data["prediction"], config["NUM_CLASSES"])); axs[2].set_title("Predizione"); axs[2].axis('off')
        plt.tight_layout(); plt.show()

def plot_confusion_matrix(model: nn.Module, loader: DataLoader, device: torch.device, config: Dict[str, Any]):
    print("\n--- Calcolo della Matrice di Confusione ---")
    model.eval()
    num_classes, ignore_index = config["NUM_CLASSES"], config.get("IGNORE_INDEX")
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Generazione Matrice Confusione"):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            with torch.amp.autocast(enabled=True, device_type=device.type):
                outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            valid_indices = masks.flatten() != ignore_index
            indices = masks.flatten()[valid_indices] * num_classes + preds.flatten()[valid_indices]
            conf_matrix += torch.bincount(indices, minlength=num_classes**2).reshape(num_classes, num_classes)
    
    plot_matrix = np.delete(np.delete(conf_matrix.cpu().numpy(), ignore_index, axis=0), ignore_index, axis=1)
    plot_class_names = [name for i, name in enumerate(config["CLASS_NAMES"]) if i != ignore_index]
    norm_conf_matrix = np.divide(plot_matrix, plot_matrix.sum(axis=1)[:, np.newaxis], out=np.zeros_like(plot_matrix, dtype=float), where=plot_matrix.sum(axis=1)[:, np.newaxis]!=0)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(norm_conf_matrix, annot=True, fmt=".2%", cmap="Blues", xticklabels=plot_class_names, yticklabels=plot_class_names)
    plt.title("Matrice di Confusione Normalizzata (Recall per Classe)"); plt.ylabel("Classe Vera"); plt.xlabel("Classe Predetta"); plt.tight_layout(); plt.show()

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
    model.eval()
    total_loss = 0.0
    iou_list_per_image = [] 
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            with torch.amp.autocast(enabled=True, device_type=device.type):
                outputs = model(images)
                loss = criterion(outputs, masks)
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            for i in range(images.size(0)):
                iou_scores = compute_iou_per_class(preds[i].cpu().numpy(), masks[i].cpu().numpy(), num_classes, ignore_index)
                iou_list_per_image.append(iou_scores)
    
    avg_loss = total_loss / len(loader)
    iou_per_class = np.nanmean(np.array(iou_list_per_image), axis=0)
    mean_iou = np.nanmean(iou_per_class)
    return avg_loss, mean_iou, iou_per_class

# =============================================================================
# 6. FUNZIONE PRINCIPALE (MAIN)
# =============================================================================
def main():
    config = get_config()
    device = config["DEVICE"]
    use_amp = config["USE_AMP"] and device.type == 'cuda'
    print(f"Device: {device} | Mixed Precision: {use_amp}")

    train_dirs, val_dirs = get_grouped_stratified_split(config)
    
    base_transforms = get_transforms(output_size=config["OUTPUT_SIZE"])

    # --- MODIFICA CHIAVE: Passa la mappa colori ai Dataset ---
    train_dataset = SegmentationDataset(train_dirs, base_transforms, COLOR_TO_CLASS_ID)
    val_dataset = SegmentationDataset(val_dirs, base_transforms, COLOR_TO_CLASS_ID)
    
    train_sampler = create_class_balanced_sampler(train_dirs, config)

    train_loader = DataLoader(
        train_dataset, batch_size=config["BATCH_SIZE"], sampler=train_sampler,
        num_workers=config["NUM_WORKERS_TRAIN"], pin_memory=True, shuffle=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False,
        num_workers=config["NUM_WORKERS_VAL"], pin_memory=True
    )

    model = get_model(config).to(device)
    
    criterion_dice = DiceLoss(mode='multiclass', from_logits=True, ignore_index=config["IGNORE_INDEX"]) 
    criterion_focal = FocalLoss(mode='multiclass', gamma=2, ignore_index=config["IGNORE_INDEX"])
    def combined_loss(pred, target):
        return 0.7 * criterion_dice(pred, target) + 0.3 * criterion_focal(pred, target)
    
    optimizer = optim.AdamW(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_loader), epochs=config["NUM_EPOCHS"]
    )
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_iou = -1.0
    history = {'train_loss': [], 'val_loss': [], 'val_miou': []}
    
    print("\n--- Inizio Addestramento con Mappa Colori Corretta ---")
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, combined_loss, scaler, scheduler, device, use_amp)
        val_loss, val_miou, val_iou_per_class = evaluate(model, val_loader, combined_loss, device, config["NUM_CLASSES"], config["IGNORE_INDEX"])
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_miou'].append(val_miou)
        
        print(f"Risultati Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f}")
        print("  - IoU per Classe (Validation):")
        for i, iou in enumerate(val_iou_per_class):
            if i != config["IGNORE_INDEX"] and not np.isnan(iou):
                print(f"    - {config['CLASS_NAMES'][i]}: {iou:.4f}")
        
        if val_miou > best_iou:
            best_iou = val_miou
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<\n")

    print(f"\n--- Addestramento Completato ---")
    print(f"Miglior mIoU di validazione ottenuto: {best_iou:.4f}")
    
    # --- ANALISI FINALE SUL MODELLO MIGLIORE ---
    print("\nCaricamento del miglior modello per l'analisi finale...")
    model.load_state_dict(torch.load(config["MODEL_SAVE_PATH"]))
    
    show_best_worst_predictions(model, val_loader, device, config)
    plot_confusion_matrix(model, val_loader, device, config)

# =============================================================================
# 7. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()