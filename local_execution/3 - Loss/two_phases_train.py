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
        "RARE_CLASS_INDICES": [4, 5, 6],
        "NUM_WORKERS": 0,
        "NUM_WORKERS_TRAIN": 2,
        "NUM_WORKERS_VAL": 1,
        "RARE_BOOST_FACTOR": 5.0, # Moltiplicatore per il peso delle classi rare
        
        # Parametri del modello
        "ENCODER": "resnet34",
        "ENCODER_WEIGHTS": "imagenet",
        "NUM_CLASSES": 9,
        "IGNORE_INDEX": 0,

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

    # Calcola i pesi SOLO sul training set
    class_weights = calculate_class_weights(train_dataset, config)

    # --- Modello e Loss ---
    model = get_model(config).to(device)
    criterion = CombinedLoss(class_weights=class_weights, ignore_index=config["IGNORE_INDEX"]).to(device)
    scaler = torch.amp.GradScaler(enabled=config["USE_AMP"])
    best_iou = -1.0
    
    # --- FASE 1: TRAINING DEL SOLO DECODER (ENCODER CONGELATO) ---
    print(f"\n--- INIZIO FASE 1: TRAINING DECODER ({config['FREEZE_EPOCHS']} epoche) ---")
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(model.decoder.parameters(), lr=config["LR_DECODER"])
    
    for epoch in range(config["FREEZE_EPOCHS"]):
        print(f"\n--- Epoch (Freeze) {epoch+1}/{config['FREEZE_EPOCHS']} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, config["USE_AMP"])
        val_loss, val_iou, val_iou_per_class = evaluate(model, val_loader, criterion, device, config["NUM_CLASSES"], config["IGNORE_INDEX"])

        print(f"Risultati Freeze Epoch {epoch+1}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4} | Val mIoU: {val_iou:.4f}")
        # Stampa l'IoU per ogni classe
        print("  - IoU per Classe:")
        for i, iou in enumerate(val_iou_per_class):
            if i != config["IGNORE_INDEX"]: # Non stampare l'IoU per la classe ignorata
                print(f"    - Classe {i}: {iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"🎉 Modello migliore salvato con mIoU: {best_iou:.4f}")

    # --- FASE 2: FINE-TUNING DI TUTTA LA RETE ---
    print(f"\n--- INIZIO FASE 2: FINE-TUNING COMPLETO ({config['FINETUNE_EPOCHS']} epoche) ---")
    for param in model.parameters():
        param.requires_grad = True

    optimizer = optim.Adam(model.parameters(), lr=config["LR_FINETUNE"], weight_decay=config["WEIGHT_DECAY"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

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
                print(f"    - Classe {i}: {iou:.4f}")

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
Device: cuda | Mixed Precision: True
Avvio analisi di similarità e split stratificato...
Trovati 902 gruppi di immagini.
Etichettatura gruppi:   0%|                                                                                                            | 0/902 [00:00<?, ?it/s] 












Split stratificato basato su gruppi creato: 743 train, 188 validation.
C:\Users\leona\AppData\Roaming\Python\Python312\site-packages\albumentations\core\validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
d:\MLPoject-Git\autoencoder-onboard-segmentation\local_execution\3 - Loss\two_phases_train.py:101: UserWarning: Argument(s) 'max_holes, max_height, max_width, fill_value' are not valid for transform CoarseDropout
  A.CoarseDropout(max_holes=8, max_height=24, max_width=24, fill_value=0, p=0.3),
Calcolo della distribuzione delle classi per i pesi della loss...
Analisi distribuzione pixel: 100%|█████████████████████████████████████████████████████████████████████████████████████████████| 93/93 [00:08<00:00, 11.34it/s] 

Pesi calcolati per la loss (con boost per classi rare):
  - Classe 0: 0.02
  - Classe 1: 0.02
  - Classe 2: 0.02
  - Classe 3: 0.02
  - Classe 4: 6.99
  - Classe 5: 1.66
  - Classe 6: 0.24
  - Classe 7: 0.01
  - Classe 8: 0.03
Creazione del modello: U-Net con encoder 'resnet34'
Parametri addestrabili del modello: 24,437,529

--- INIZIO FASE 1: TRAINING DECODER (5 epoche) ---

--- Epoch (Freeze) 1/5 ---
d:\MLPoject-Git\autoencoder-onboard-segmentation\local_execution\3 - Loss\two_phases_train.py:312: RuntimeWarning: Mean of empty slice                          
  iou_per_class = np.nanmean(np.array(iou_list_per_image), axis=0)
Risultati Freeze Epoch 1: Train Loss: 1.3908 | Val Loss: 1.261 | Val mIoU: 0.1045
  - IoU per Classe:
    - Classe 1: 0.0000
    - Classe 2: 0.0022
    - Classe 3: 0.1422
    - Classe 4: 0.0000
    - Classe 5: 0.0018
    - Classe 6: 0.1019
    - Classe 7: 0.0228
    - Classe 8: 0.5651
🎉 Modello migliore salvato con mIoU: 0.1045

--- Epoch (Freeze) 2/5 ---
Risultati Freeze Epoch 2: Train Loss: 1.2955 | Val Loss: 1.161 | Val mIoU: 0.1904                                             
  - IoU per Classe:
    - Classe 1: 0.0036
    - Classe 2: 0.0024
    - Classe 3: 0.1379
    - Classe 4: 0.0005
    - Classe 5: 0.0128
    - Classe 6: 0.1306
    - Classe 7: 0.4950
    - Classe 8: 0.7404

--- Epoch (Freeze) 3/5 ---
--- Epoch (Freeze) 3/5 ---
Risultati Freeze Epoch 3: Train Loss: 1.2636 | Val Loss: 1.129 | Val mIoU: 0.1618
  - IoU per Classe:
    - Classe 1: 0.0043
    - Classe 2: 0.0036
    - Classe 3: 0.1729
    - Classe 4: 0.0007
    - Classe 5: 0.0163
    - Classe 6: 0.1151
    - Classe 7: 0.2494
    - Classe 8: 0.7325

--- Epoch (Freeze) 4/5 ---
Validazione:   0%|                                                                                                                      | 0/24 [00:00<?, ?it/s]
Validazione:   4%|████▌                                                                                           Validazione:  12%|█████████████▊                                                                                  Validazione:  21%|██████████████████████▉                                                                            Validazione:  25%|███████████████████████████▌                                                                     
Validazione:  33%|████████████████████████████████████▋                                                            
Validazione:  42%|█████████████████████████████████████████████▍                                                   
Validazione:  50%|██████████████████████████████████████████████████████▌                                          
Validazione:  58%|███████████████████████████████████████████████████████████████▌                                 
Validazione:  67%|████████████████████████████████████████████████████████████████████████▋                        
Validazione:  75%|█████████████████████████████████████████████████████████████████████████████████▊               
Validazione:  83%|██████████████████████████████████████████████████████████████████████████████████████████▊      Validazione:  92%|█████████████████████████████████████████████████████████████████████████████████████████████████Validazione: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████                                                                                                                   
Risultati Freeze Epoch 4: Train Loss: 1.2257 | Val Loss: 1.076 | Val mIoU: 0.1906
  - IoU per Classe:
    - Classe 1: 0.0006
    - Classe 2: 0.0016
    - Classe 3: 0.1533
    - Classe 4: 0.0025
    - Classe 5: 0.0168
    - Classe 6: 0.1451
    - Classe 7: 0.4793
    - Classe 8: 0.7259
🎉 Modello migliore salvato con mIoU: 0.1906

--- Epoch (Freeze) 5/5 ---
Risultati Freeze Epoch 5: Train Loss: 1.1907 | Val Loss: 0.9821 | Val mIoU: 0.2325                                 
  - IoU per Classe:
    - Classe 1: 0.1827
    - Classe 2: 0.0116
    - Classe 3: 0.1243
    - Classe 4: 0.0016
    - Classe 5: 0.0197
    - Classe 6: 0.1457
    - Classe 7: 0.6232
    - Classe 8: 0.7512
🎉 Modello migliore salvato con mIoU: 0.2325

--- INIZIO FASE 2: FINE-TUNING COMPLETO (20 epoche) ---

--- Epoch (Finetune) 1/20 ---
Risultati Finetune Epoch 1: Train Loss: 1.1355 | Val Loss: 0.9686 | Val mIoU: 0.2306                               
  - IoU per Classe:
    - Classe 1: 0.1523
    - Classe 2: 0.0041
    - Classe 3: 0.1531
    - Classe 4: 0.0020
    - Classe 5: 0.0208
    - Classe 6: 0.1479
    - Classe 7: 0.6047
    - Classe 8: 0.7598

--- Epoch (Finetune) 2/20 ---
Risultati Finetune Epoch 2: Train Loss: 1.0995 | Val Loss: 0.9478 | Val mIoU: 0.2305                               
  - IoU per Classe:
    - Classe 1: 0.1460
    - Classe 2: 0.0029
    - Classe 3: 0.1609
    - Classe 4: 0.0017
    - Classe 5: 0.0207
    - Classe 6: 0.1536
    - Classe 7: 0.5965
    - Classe 8: 0.7620

--- Epoch (Finetune) 3/20 ---
Risultati Finetune Epoch 3: Train Loss: 1.0806 | Val Loss: 0.9368 | Val mIoU: 0.2284                               
  - IoU per Classe:
    - Classe 1: 0.1248
    - Classe 2: 0.0020
    - Classe 3: 0.1555
    - Classe 4: 0.0020
    - Classe 5: 0.0194
    - Classe 6: 0.1538
    - Classe 7: 0.6082
    - Classe 8: 0.7616

--- Epoch (Finetune) 4/20 ---
Risultati Finetune Epoch 4: Train Loss: 1.0726 | Val Loss: 0.9164 | Val mIoU: 0.2264                               
  - IoU per Classe:
    - Classe 1: 0.1505
    - Classe 2: 0.0015
    - Classe 3: 0.1510
    - Classe 4: 0.0023
    - Classe 5: 0.0224
    - Classe 6: 0.1519
    - Classe 7: 0.5730
    - Classe 8: 0.7587

--- Epoch (Finetune) 5/20 ---
Risultati Finetune Epoch 5: Train Loss: 1.0523 | Val Loss: 0.9037 | Val mIoU: 0.2359                               
  - IoU per Classe:
    - Classe 1: 0.1571
    - Classe 2: 0.0013
    - Classe 3: 0.1699
    - Classe 4: 0.0025
    - Classe 5: 0.0218
    - Classe 6: 0.1568
    - Classe 7: 0.6169
    - Classe 8: 0.7606
🎉 Nuovo modello migliore salvato con mIoU: 0.2359

--- Epoch (Finetune) 6/20 ---
Risultati Finetune Epoch 6: Train Loss: 1.0336 | Val Loss: 0.894 | Val mIoU: 0.2332                                
  - IoU per Classe:
    - Classe 1: 0.1753
    - Classe 2: 0.0010
    - Classe 3: 0.1664
    - Classe 4: 0.0025
    - Classe 5: 0.0198
    - Classe 6: 0.1544
    - Classe 7: 0.5844
    - Classe 8: 0.7619

--- Epoch (Finetune) 7/20 ---
Risultati Finetune Epoch 7: Train Loss: 1.0170 | Val Loss: 0.8838 | Val mIoU: 0.2430                               
  - IoU per Classe:
    - Classe 1: 0.2206
    - Classe 2: 0.0010
    - Classe 3: 0.1722
    - Classe 4: 0.0019
    - Classe 5: 0.0205
    - Classe 6: 0.1593
    - Classe 7: 0.6077
    - Classe 8: 0.7608
🎉 Nuovo modello migliore salvato con mIoU: 0.2430

--- Epoch (Finetune) 8/20 ---
Risultati Finetune Epoch 8: Train Loss: 1.0121 | Val Loss: 0.8704 | Val mIoU: 0.2396                               
  - IoU per Classe:
    - Classe 1: 0.1997
    - Classe 2: 0.0008
    - Classe 3: 0.1714
    - Classe 4: 0.0022
    - Classe 5: 0.0214
    - Classe 6: 0.1595
    - Classe 7: 0.6002
    - Classe 8: 0.7613

--- Epoch (Finetune) 9/20 ---
Risultati Finetune Epoch 9: Train Loss: 0.9825 | Val Loss: 0.8641 | Val mIoU: 0.2318                               
  - IoU per Classe:
    - Classe 1: 0.1810
    - Classe 2: 0.0007
    - Classe 3: 0.1645
    - Classe 4: 0.0026
    - Classe 5: 0.0221
    - Classe 6: 0.1562
    - Classe 7: 0.5646
    - Classe 8: 0.7631

--- Epoch (Finetune) 10/20 ---
Risultati Finetune Epoch 10: Train Loss: 0.9868 | Val Loss: 0.8686 | Val mIoU: 0.2438                              
  - IoU per Classe:
    - Classe 1: 0.2292
    - Classe 2: 0.0007
    - Classe 3: 0.1685
    - Classe 4: 0.0028
    - Classe 5: 0.0214
    - Classe 6: 0.1581
    - Classe 7: 0.6046
    - Classe 8: 0.7648
🎉 Nuovo modello migliore salvato con mIoU: 0.2438

--- Epoch (Finetune) 11/20 ---
Risultati Finetune Epoch 11: Train Loss: 0.9568 | Val Loss: 0.8449 | Val mIoU: 0.2425                              
  - IoU per Classe:
    - Classe 1: 0.2393
    - Classe 2: 0.0006
    - Classe 3: 0.1564
    - Classe 4: 0.0028
    - Classe 5: 0.0220
    - Classe 6: 0.1603
    - Classe 7: 0.5951
    - Classe 8: 0.7637

--- Epoch (Finetune) 12/20 ---
Risultati Finetune Epoch 12: Train Loss: 0.9542 | Val Loss: 0.845 | Val mIoU: 0.2480                               
  - IoU per Classe:
    - Classe 1: 0.2395
    - Classe 2: 0.0007
    - Classe 3: 0.1538
    - Classe 4: 0.0027
    - Classe 5: 0.0223
    - Classe 6: 0.1595
    - Classe 7: 0.6401
    - Classe 8: 0.7651
🎉 Nuovo modello migliore salvato con mIoU: 0.2480

--- Epoch (Finetune) 13/20 ---
Risultati Finetune Epoch 13: Train Loss: 0.9399 | Val Loss: 0.8442 | Val mIoU: 0.2506                              
  - IoU per Classe:
    - Classe 1: 0.2530
    - Classe 2: 0.0006
    - Classe 3: 0.1532
    - Classe 4: 0.0030
    - Classe 5: 0.0213
    - Classe 6: 0.1624
    - Classe 7: 0.6485
    - Classe 8: 0.7626
🎉 Nuovo modello migliore salvato con mIoU: 0.2506

--- Epoch (Finetune) 14/20 ---
Risultati Finetune Epoch 14: Train Loss: 0.9153 | Val Loss: 0.8205 | Val mIoU: 0.2485                              
  - IoU per Classe:
    - Classe 1: 0.2750
    - Classe 2: 0.0005
    - Classe 3: 0.1440
    - Classe 4: 0.0027
    - Classe 5: 0.0238
    - Classe 6: 0.1598
    - Classe 7: 0.6196
    - Classe 8: 0.7625

--- Epoch (Finetune) 15/20 ---
Risultati Finetune Epoch 15: Train Loss: 0.9388 | Val Loss: 0.8124 | Val mIoU: 0.2510                              
  - IoU per Classe:
    - Classe 1: 0.2610
    - Classe 2: 0.0004
    - Classe 3: 0.1426
    - Classe 4: 0.0029
    - Classe 5: 0.0239
    - Classe 6: 0.1627
    - Classe 7: 0.6476
    - Classe 8: 0.7670
🎉 Nuovo modello migliore salvato con mIoU: 0.2510

--- Epoch (Finetune) 16/20 ---
Risultati Finetune Epoch 16: Train Loss: 0.9210 | Val Loss: 0.8153 | Val mIoU: 0.2480                              
  - IoU per Classe:
    - Classe 1: 0.2675
    - Classe 2: 0.0005
    - Classe 3: 0.1339
    - Classe 4: 0.0029
    - Classe 5: 0.0226
    - Classe 6: 0.1600
    - Classe 7: 0.6387
    - Classe 8: 0.7577

--- Epoch (Finetune) 17/20 ---
Risultati Finetune Epoch 17: Train Loss: 0.9059 | Val Loss: 0.8155 | Val mIoU: 0.2537                              
  - IoU per Classe:
    - Classe 1: 0.2722
    - Classe 2: 0.0004
    - Classe 3: 0.1426
    - Classe 4: 0.0021
    - Classe 5: 0.0221
    - Classe 6: 0.1620
    - Classe 7: 0.6587
    - Classe 8: 0.7694
🎉 Nuovo modello migliore salvato con mIoU: 0.2537

--- Epoch (Finetune) 18/20 ---
Risultati Finetune Epoch 18: Train Loss: 0.8984 | Val Loss: 0.799 | Val mIoU: 0.2508                               
  - IoU per Classe:
    - Classe 1: 0.2751
    - Classe 2: 0.0004
    - Classe 3: 0.1264
    - Classe 4: 0.0030
    - Classe 5: 0.0221
    - Classe 6: 0.1629
    - Classe 7: 0.6483
    - Classe 8: 0.7682

--- Epoch (Finetune) 19/20 ---
Risultati Finetune Epoch 19: Train Loss: 0.8731 | Val Loss: 0.8037 | Val mIoU: 0.2509                              
  - IoU per Classe:
    - Classe 1: 0.2888
    - Classe 2: 0.0004
    - Classe 3: 0.1330
    - Classe 4: 0.0030
    - Classe 5: 0.0213
    - Classe 6: 0.1613
    - Classe 7: 0.6310
    - Classe 8: 0.7687

--- Epoch (Finetune) 20/20 ---
Risultati Finetune Epoch 20: Train Loss: 0.8915 | Val Loss: 0.7957 | Val mIoU: 0.2537                              
  - IoU per Classe:
    - Classe 1: 0.2801
    - Classe 2: 0.0003
    - Classe 3: 0.1301
    - Classe 4: 0.0026
    - Classe 5: 0.0233
    - Classe 6: 0.1615
    - Classe 7: 0.6631
    - Classe 8: 0.7683
🎉 Nuovo modello migliore salvato con mIoU: 0.2537

--- Addestramento Completato ---
Miglior mIoU di validazione: 0.2537


'''