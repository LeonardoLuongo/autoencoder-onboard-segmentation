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
    # Imposta il percorso base a seconda dell'ambiente
    base_path = "./" 
    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        base_path = "/content/drive/MyDrive/ML_Project/" # Esempio per Colab
    
    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "DATASET_PATH": os.path.join(base_path, "dataset/training/train"),
        "MODEL_SAVE_PATH": os.path.join(base_path, "best_advanced_model.pth"),
        "NUM_EPOCHS": 25,
        "BATCH_SIZE": 4,          # Un valore sicuro per iniziare
        "LEARNING_RATE": 1e-4,    # Un buon LR per il fine-tuning
        "NUM_CLASSES": 9,
        "IGNORE_INDEX": 0,
        "OUTPUT_SIZE": (256, 512),# La risoluzione che ci ha dato la migliore baseline
        "VALIDATION_SPLIT": 0.2,
        "RANDOM_SEED": 42,
        "RARE_CLASS_INDICES": [4, 5, 6, 8], # Classi su cui stratificare
        "USE_AMP": True,
        "NUM_WORKERS_TRAIN": 2,
        "NUM_WORKERS_VAL": 1,
        "ENCODER": "resnet34",
        "ENCODER_WEIGHTS": "imagenet",
        "WEIGHT_DECAY": 1e-5,     # Aggiungiamo la regolarizzazione L2
    }
    return config

# =============================================================================
# 3. DATA AUGMENTATION & DATASET CLASS
# =============================================================================
def get_transforms(is_train: bool, output_size: Tuple[int, int]) -> A.Compose:
    """Restituisce la pipeline di augmentation appropriata."""
    mean=(0.485, 0.456, 0.406)
    std=(0.229, 0.224, 0.225)
    
    if is_train:
        return A.Compose([
            A.Resize(height=output_size[0], width=output_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.RandomBrightnessContrast(p=0.4),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else: # Validation
        return A.Compose([
            A.Resize(height=output_size[0], width=output_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

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
# 4. FUNZIONI HELPER (SPLIT E METRICHE)
# =============================================================================
def get_grouped_stratified_split(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    # ... (codice identico a quello precedente per lo split con networkx) ...
    # Questa funzione ora è completa e robusta
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

def evaluate(model, loader, criterion, device, num_classes, ignore_index):
    model.eval()
    total_loss = 0.0
    all_class_ious = []
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione", leave=False):
            images, masks = images.to(device), masks.to(device, dtype=torch.long)
            
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            
            for i in range(images.size(0)):
                iou = compute_iou(preds[i].cpu().numpy(), masks[i].cpu().numpy(), num_classes, ignore_index)
                all_class_ious.append(iou)
                
    avg_loss = total_loss / len(loader)
    # Calcola la media per ogni classe, poi la media di queste medie (mIoU)
    mean_iou = np.nanmean(np.array(all_class_ious), axis=0).mean()
    
    return avg_loss, mean_iou

# =============================================================================
# 6. FUNZIONE PRINCIPALE (MAIN)
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

    model = get_model(config).to(device)
    
    criterion = CombinedLoss(ignore_index=config["IGNORE_INDEX"])
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)
    scaler = torch.amp.GradScaler(enabled=use_amp)

    best_iou = -1.0
    
    print("\n--- Inizio Addestramento Avanzato ---")
    for epoch in range(config["NUM_EPOCHS"]):
        print(f"\n--- Epoch {epoch+1}/{config['NUM_EPOCHS']} ---")
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
        val_loss, val_iou = evaluate(model, val_loader, criterion, device, config["NUM_CLASSES"], config["IGNORE_INDEX"])
        
        scheduler.step(val_iou)
        
        print(f"Epoch {epoch+1} -> Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_iou:.4f}")
        
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



'''


'''