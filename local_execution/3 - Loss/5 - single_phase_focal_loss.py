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

# =============================================================================
# 2. CONFIGURAZIONE
# =============================================================================
def get_config() -> Dict[str, Any]:
    """Restituisce un dizionario con tutte le configurazioni dello script."""
    # ... (gestione percorso base invariata) ...
    base_path = "./"

    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "DATASET_PATH": os.path.join(base_path, "dataset/training/train"),
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
        "ENCODER": "efficientnet-b2",
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
# 3. DATA AUGMENTATION (PIÙ AGGRESSIVA)
# =============================================================================
def get_transforms(is_train: bool, output_size: Tuple[int, int]) -> A.Compose:
    """Restituisce la pipeline di augmentation appropriata."""
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    
    if is_train:
        return A.Compose([
            A.Resize(height=output_size[0], width=output_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(p=0.5), # Parametri di default, meno estremi
            A.RandomBrightnessContrast(p=0.3),
            # Rimuoviamo CoarseDropout per ora, è molto aggressivo
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else: # Validation
        return A.Compose([
            A.Resize(height=output_size[0], width=output_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

# =============================================================================
# 4. FUNZIONI HELPER (SPLIT E METRICHE)
# =============================================================================
def get_grouped_stratified_split(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    dataset_path = config["DATASET_PATH"]
    print("Avvio analisi di similarità e split stratificato...")
    all_dirs = sorted([d for d in glob.glob(os.path.join(dataset_path, "*")) if os.path.isdir(d) and not os.path.basename(d).startswith('__')])
    
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
    
    loader = DataLoader(dataset, batch_size=config["BATCH_SIZE"], num_workers=config["NUM_WORKERS"])
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

    
# =============================================================================
# 5. FUNZIONI DI TRAINING E VALIDAZIONE
# =============================================================================
def train_one_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
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
    
    # Prepara le trasformazioni
    train_transforms = get_transforms(is_train=True, output_size=config["OUTPUT_SIZE"])
    val_transforms = get_transforms(is_train=False, output_size=config["OUTPUT_SIZE"])

    # Crea i Dataset
    train_dataset = SegmentationDataset(train_dirs, train_transforms)
    val_dataset = SegmentationDataset(val_dirs, val_transforms)
    
    # Crea i DataLoader
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config["BATCH_SIZE"], 
        shuffle=True,
        num_workers=config["NUM_WORKERS"], 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config["BATCH_SIZE"], 
        shuffle=False,
        num_workers=config["NUM_WORKERS"], 
        pin_memory=True
    )

    # 3. Prepara il Modello, la Loss, l'Ottimizzatore e lo Scheduler
    
    # Modello
    model = get_model(config).to(device)
    
    # Calcola i pesi per la loss sul training set (la funzione va definita sopra)
    class_weights = calculate_class_weights(train_dataset, config)
    
    # Alpha è il peso per le classi (simile a prima), gamma è il fattore di focusing
    criterion = FocalLoss(mode='multiclass', alpha=class_weights, gamma=2.0).to(device)
    
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
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_miou, val_iou_per_class = evaluate(model, val_loader, criterion, device, config["NUM_CLASSES"], config["IGNORE_INDEX"])
        
        # Aggiorna lo scheduler con il mIoU di validazione
        scheduler.step(val_miou)
        
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

    print(f"\n--- Addestramento Completato ---")
    print(f"Miglior mIoU di validazione ottenuto: {best_iou:.4f}")
    
    # Qui potresti aggiungere codice per plottare i grafici dalla variabile 'history'

# =============================================================================
# 7. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()

