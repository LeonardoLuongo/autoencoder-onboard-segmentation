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
        "OUTPUT_SIZE": (256, 512),
        "VALIDATION_SPLIT": 0.2,
        "RANDOM_SEED": 42,
        "RARE_CLASS_INDICES": [4, 5, 6, 8],
        "NUM_WORKERS": 0,
        
        # Parametri del modello
        "ENCODER": "resnet34",
        "ENCODER_WEIGHTS": "imagenet",
        "NUM_CLASSES": 9,
        
        # --- IPERPARAMETRI OTTIMIZZATI PER COMBATTERE L'OVERFITTING ---
        "BATCH_SIZE": 8,          # Aumentiamo un po' se la memoria lo consente
        "USE_AMP": True,
        
        # Parametri per il training a due fasi
        "FREEZE_EPOCHS": 5,       # Epoche per addestrare solo il decoder
        "FINETUNE_EPOCHS": 20,    # Epoche per addestrare tutta la rete
        "LR_DECODER": 1e-3,       # Learning rate più alto per il decoder
        "LR_FINETUNE": 1e-5,      # Learning rate molto basso per il fine-tuning
        "WEIGHT_DECAY": 1e-4,     # Regolarizzazione L2 aumentata
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
        # Pipeline di augmentation più forte per il training
        return A.Compose([
            A.Resize(height=output_size[0], width=output_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=25, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussNoise(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=24, max_width=24, fill_value=0, p=0.3),
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
    """
    Calcola una loss combinata pesando CrossEntropy e DiceLoss.
    È particolarmente utile per la segmentazione con classi sbilanciate.
    """
    def __init__(self, ce_weight=0.4, dice_weight=0.6, ignore_index=0):
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        
        # CrossEntropyLoss è utile per la stabilità e la convergenza pixel per pixel
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index)
        
        # DiceLoss è ottima per lo sbilanciamento delle classi e ottimizza l'overlap
        self.dice = DiceLoss(mode='multiclass')

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred (torch.Tensor): Logits prodotti dal modello (B, C, H, W).
            target (torch.Tensor): Maschere ground-truth (B, H, W).
        
        Returns:
            torch.Tensor: Il valore di loss scalare.
        """
        # Calcola le due componenti della loss
        ce_loss = self.ce(pred, target)
        dice_loss = self.dice(pred, target)
        
        # Combinale con i pesi specificati
        combined = (self.ce_weight * ce_loss) + (self.dice_weight * dice_loss)
        
        return combined
    
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
            
            with autocast(enabled=(device.type=='cuda')):
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
# 6. FUNZIONE PRINCIPALE (MODIFICATA PER IL TRAINING A DUE FASI)
# =============================================================================
def main():
    config = get_config()
    device = config["DEVICE"]
    use_amp = config["USE_AMP"] and device.type == 'cuda'
    
    print(f"Device: {device} | Mixed Precision: {use_amp}")

    train_dirs, val_dirs = get_grouped_stratified_split(config)
    
    train_transforms = get_transforms(is_train=True, output_size=config["OUTPUT_SIZE"])
    val_transforms = get_transforms(output_size=config["OUTPUT_SIZE"], is_train=False)
    
    train_dataset = SegmentationDataset(train_dirs, train_transforms)
    val_dataset = SegmentationDataset(val_dirs, val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS_TRAIN"], pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS_VAL"], pin_memory=True)

    # Lista dei nomi delle classi per il logging
    class_names = []
    for i in range(config["NUM_CLASSES"]):
        if i == config["IGNORE_INDEX"]:
            class_names.append("Ignore")
        else:
            class_names.append(f"Class {i}")

    # --- Modello e Loss ---
    model = get_model(config).to(device)
    criterion = CombinedLoss(ignore_index=config["IGNORE_INDEX"])
    scaler = GradScaler(enabled=config["USE_AMP"])
    best_iou = -1.0
    
    # --- FASE 1: TRAINING DEL SOLO DECODER (ENCODER CONGELATO) ---
    print(f"\n--- INIZIO FASE 1: TRAINING DECODER ({config['FREEZE_EPOCHS']} epoche) ---")
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.decoder.parameters(), lr=config["LR_DECODER"])
    
    for epoch in range(config["FREEZE_EPOCHS"]):
        print(f"\n--- Epoch (Freeze) {epoch+1}/{config['FREEZE_EPOCHS']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, config["USE_AMP"])
        val_loss, val_iou = evaluate(model, val_loader, criterion, device, config["NUM_CLASSES"], config["IGNORE_INDEX"])

        print(f"Risultati Freeze Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4} | Val mIoU: {val_iou:.4f}")
        # Stampa l'IoU per ogni classe
        print("  - IoU per Classe:")
        for i, iou in enumerate(val_iou_per_class):
            if i != config["IGNORE_INDEX"]: # Non stampare l'IoU per la classe ignorata
                print(f"    - Classe {i} ({class_names[i]}): {iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"🎉 Modello migliore salvato con mIoU: {best_iou:.4f}")

    # --- FASE 2: FINE-TUNING DI TUTTA LA RETE ---
    print(f"\n--- INIZIO FASE 2: FINE-TUNING COMPLETO ({config['FINETUNE_EPOCHS']} epoche) ---")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=config["LR_FINETUNE"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3, verbose=True)

    for epoch in range(config["FINETUNE_EPOCHS"]):
        print(f"\n--- Epoch (Finetune) {epoch+1}/{config['FINETUNE_EPOCHS']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, config["USE_AMP"])
        val_loss, val_iou, val_iou_per_class = evaluate(model, val_loader, criterion, device, config["NUM_CLASSES"], config["IGNORE_INDEX"])
        scheduler.step(val_iou)
        
        print(f"Risultati Finetune Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4} | Val mIoU: {val_iou:.4f}")
        # Stampa l'IoU per ogni classe
        print("  - IoU per Classe:")
        for i, iou in enumerate(val_iou_per_class):
            if i != config["IGNORE_INDEX"]: # Non stampare l'IoU per la classe ignorata
                print(f"    - Classe {i} ({class_names[i]}): {iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"🎉 Nuovo modello migliore salvato con mIoU: {best_iou:.4f}")

    print(f"\n--- Addestramento Completato --- \nMiglior mIoU di validazione: {best_iou:.4f}")

# =============================================================================
# 7. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    main()