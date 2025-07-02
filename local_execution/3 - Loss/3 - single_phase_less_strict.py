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
    
    # Loss Function (solo CrossEntropy pesata per ora, per semplicità e stabilità)
    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=config["IGNORE_INDEX"]).to(device)
    
    # Ottimizzatore
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"])
    
    # Scheduler per adattare il learning rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max',      # Si basa su una metrica che deve essere massimizzata (il nostro mIoU)
        factor=0.2,      # Riduci LR a 1/5 del valore attuale
        patience=2       # Aspetta 2 epoche senza miglioramenti prima di agire
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


'''

Device: cuda | Mixed Precision: True
Avvio analisi di similarità e split stratificato...
Trovati 902 gruppi di immagini.
Etichettatura gruppi:   0%|                                                                                                                              | 0/902 [00:00<?, ?it/s] 












Split stratificato basato su gruppi creato: 743 train, 188 validation.
C:\Users\leona\AppData\Roaming\Python\Python312\site-packages\albumentations\core\validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
Creazione del modello: U-Net con encoder 'efficientnet-b2'
Parametri addestrabili del modello: 10,047,451
Calcolo della distribuzione delle classi per i pesi della loss...
Analisi distribuzione pixel: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████| 93/93 [00:09<00:00,  9.76it/s] 

Pesi calcolati per la loss (con boost per classi rare):
  - Classe 0: 0.02
  - Classe 1: 0.02
  - Classe 2: 0.02
  - Classe 3: 0.02
  - Classe 4: 7.10
  - Classe 5: 1.55
  - Classe 6: 0.23
  - Classe 7: 0.01
  - Classe 8: 0.03

--- Inizio Addestramento (Strategia Bilanciata) ---

--- Epoch 1/30 ---
d:\MLPoject-Git\autoencoder-onboard-segmentation\local_execution\3 - Loss\3 - single_phase_less_strict.py:308: RuntimeWarning: Mean of empty slice
  iou_per_class = np.nanmean(np.array(iou_list_per_image), axis=0)
Risultati Epoch 1:
  - Training Loss: 2.1008 | Validation Loss: 1.6798 | Validation mIoU: 0.1064
  - IoU per Classe (Validation):
    - Classe 1: 0.0582
    - Classe 2: 0.0025
    - Classe 3: 0.0139
    - Classe 4: 0.0005
    - Classe 5: 0.0150
    - Classe 6: 0.0966
    - Classe 7: 0.0755
    - Classe 8: 0.5892

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1064 <<<


--- Epoch 2/30 ---
Risultati Epoch 2:                                                                                                                                                                
  - Training Loss: 1.6583 | Validation Loss: 1.3609 | Validation mIoU: 0.1209
  - IoU per Classe (Validation):
    - Classe 1: 0.0441
    - Classe 2: 0.0016
    - Classe 3: 0.0386
    - Classe 4: 0.0011
    - Classe 5: 0.0151
    - Classe 6: 0.1111
    - Classe 7: 0.1043
    - Classe 8: 0.6508

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1209 <<<


--- Epoch 3/30 ---
Risultati Epoch 3:                                                                                                                                                                
  - Training Loss: 1.4426 | Validation Loss: 1.2109 | Validation mIoU: 0.1320
  - IoU per Classe (Validation):
    - Classe 1: 0.0888
    - Classe 2: 0.0022
    - Classe 3: 0.0604
    - Classe 4: 0.0020
    - Classe 5: 0.0136
    - Classe 6: 0.1164
    - Classe 7: 0.1021
    - Classe 8: 0.6703

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1320 <<<


--- Epoch 4/30 ---
Risultati Epoch 4:                                                                                                                                                                
  - Training Loss: 1.2693 | Validation Loss: 1.2006 | Validation mIoU: 0.1607
  - IoU per Classe (Validation):
    - Classe 1: 0.1525
    - Classe 2: 0.0035
    - Classe 3: 0.0714
    - Classe 4: 0.0022
    - Classe 5: 0.0132
    - Classe 6: 0.1355
    - Classe 7: 0.2197
    - Classe 8: 0.6873

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1607 <<<


--- Epoch 5/30 ---
Risultati Epoch 5:                                                                                                                                                                
  - Training Loss: 1.1258 | Validation Loss: 1.0437 | Validation mIoU: 0.1830
  - IoU per Classe (Validation):
    - Classe 1: 0.3087
    - Classe 2: 0.0031
    - Classe 3: 0.0906
    - Classe 4: 0.0037
    - Classe 5: 0.0135
    - Classe 6: 0.1405
    - Classe 7: 0.2171
    - Classe 8: 0.6869

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1830 <<<


--- Epoch 6/30 ---
Risultati Epoch 6:                                                                                                                                                                
  - Training Loss: 1.0397 | Validation Loss: 0.9600 | Validation mIoU: 0.1928
  - IoU per Classe (Validation):
    - Classe 1: 0.3147
    - Classe 2: 0.0041
    - Classe 3: 0.1167
    - Classe 4: 0.0035
    - Classe 5: 0.0136
    - Classe 6: 0.1382
    - Classe 7: 0.2545
    - Classe 8: 0.6969

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1928 <<<


--- Epoch 7/30 ---
Risultati Epoch 7:                                                                                                                                                                
  - Training Loss: 0.9367 | Validation Loss: 0.9547 | Validation mIoU: 0.1914
  - IoU per Classe (Validation):
    - Classe 1: 0.3279
    - Classe 2: 0.0099
    - Classe 3: 0.1139
    - Classe 4: 0.0040
    - Classe 5: 0.0155
    - Classe 6: 0.1242
    - Classe 7: 0.2440
    - Classe 8: 0.6919

--- Epoch 8/30 ---
Risultati Epoch 8:                                                                                                                                                                
  - Training Loss: 0.8861 | Validation Loss: 0.8896 | Validation mIoU: 0.2173
  - IoU per Classe (Validation):
    - Classe 1: 0.3959
    - Classe 2: 0.0264
    - Classe 3: 0.1264
    - Classe 4: 0.0057
    - Classe 5: 0.0142
    - Classe 6: 0.1427
    - Classe 7: 0.3006
    - Classe 8: 0.7267

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2173 <<<


--- Epoch 9/30 ---
Risultati Epoch 9:                                                                                                                                                                
  - Training Loss: 0.8588 | Validation Loss: 0.8976 | Validation mIoU: 0.2096
  - IoU per Classe (Validation):
    - Classe 1: 0.2666
    - Classe 2: 0.0353
    - Classe 3: 0.1950
    - Classe 4: 0.0046
    - Classe 5: 0.0143
    - Classe 6: 0.1290
    - Classe 7: 0.3068
    - Classe 8: 0.7251

--- Epoch 10/30 ---
Risultati Epoch 10:                                                                                                                                                               
  - Training Loss: 0.7829 | Validation Loss: 0.8742 | Validation mIoU: 0.2337
  - IoU per Classe (Validation):
    - Classe 1: 0.3317
    - Classe 2: 0.0575
    - Classe 3: 0.1720
    - Classe 4: 0.0041
    - Classe 5: 0.0161
    - Classe 6: 0.1360
    - Classe 7: 0.4175
    - Classe 8: 0.7343

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2337 <<<


--- Epoch 11/30 ---
Risultati Epoch 11:                                                                                                                                                               
  - Training Loss: 0.7471 | Validation Loss: 0.8594 | Validation mIoU: 0.2404
  - IoU per Classe (Validation):
    - Classe 1: 0.3067
    - Classe 2: 0.0602
    - Classe 3: 0.1926
    - Classe 4: 0.0044
    - Classe 5: 0.0179
    - Classe 6: 0.1371
    - Classe 7: 0.4583
    - Classe 8: 0.7458

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2404 <<<


--- Epoch 12/30 ---
Risultati Epoch 12:                                                                                                                                                               
  - Training Loss: 0.7224 | Validation Loss: 0.8607 | Validation mIoU: 0.2446
  - IoU per Classe (Validation):
    - Classe 1: 0.3442
    - Classe 2: 0.0978
    - Classe 3: 0.1551
    - Classe 4: 0.0045
    - Classe 5: 0.0186
    - Classe 6: 0.1337
    - Classe 7: 0.4588
    - Classe 8: 0.7444

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2446 <<<


--- Epoch 13/30 ---
Risultati Epoch 13:                                                                                                                                                               
  - Training Loss: 0.6917 | Validation Loss: 0.8622 | Validation mIoU: 0.2300
  - IoU per Classe (Validation):
    - Classe 1: 0.3079
    - Classe 2: 0.0856
    - Classe 3: 0.1904
    - Classe 4: 0.0033
    - Classe 5: 0.0157
    - Classe 6: 0.1327
    - Classe 7: 0.3945
    - Classe 8: 0.7103

--- Epoch 14/30 ---
Risultati Epoch 14:                                                                                                                                                               
  - Training Loss: 0.6310 | Validation Loss: 0.8158 | Validation mIoU: 0.2840
  - IoU per Classe (Validation):
    - Classe 1: 0.3716
    - Classe 2: 0.1938
    - Classe 3: 0.1784
    - Classe 4: 0.0050
    - Classe 5: 0.0178
    - Classe 6: 0.1456
    - Classe 7: 0.5906
    - Classe 8: 0.7697

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2840 <<<


--- Epoch 15/30 ---
Risultati Epoch 15:                                                                                                                                                               
  - Training Loss: 0.5949 | Validation Loss: 0.7688 | Validation mIoU: 0.2858
  - IoU per Classe (Validation):
    - Classe 1: 0.2833
    - Classe 2: 0.2055
    - Classe 3: 0.2397
    - Classe 4: 0.0043
    - Classe 5: 0.0224
    - Classe 6: 0.1526
    - Classe 7: 0.6159
    - Classe 8: 0.7626

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2858 <<<


--- Epoch 16/30 ---
Risultati Epoch 16:                                                                                                                                                               
  - Training Loss: 0.5679 | Validation Loss: 0.8312 | Validation mIoU: 0.2969
  - IoU per Classe (Validation):
    - Classe 1: 0.3166
    - Classe 2: 0.2415
    - Classe 3: 0.2319
    - Classe 4: 0.0055
    - Classe 5: 0.0219
    - Classe 6: 0.1449
    - Classe 7: 0.6513
    - Classe 8: 0.7613

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2969 <<<


--- Epoch 17/30 ---
Risultati Epoch 17:                                                                                                                                                               
  - Training Loss: 0.5752 | Validation Loss: 0.8228 | Validation mIoU: 0.2900
  - IoU per Classe (Validation):
    - Classe 1: 0.3725
    - Classe 2: 0.2169
    - Classe 3: 0.2035
    - Classe 4: 0.0055
    - Classe 5: 0.0245
    - Classe 6: 0.1446
    - Classe 7: 0.5865
    - Classe 8: 0.7662

--- Epoch 18/30 ---
Risultati Epoch 18:                                                                                                                                                               
  - Training Loss: 0.5589 | Validation Loss: 0.7370 | Validation mIoU: 0.3005
  - IoU per Classe (Validation):
    - Classe 1: 0.3826
    - Classe 2: 0.2361
    - Classe 3: 0.1908
    - Classe 4: 0.0054
    - Classe 5: 0.0244
    - Classe 6: 0.1519
    - Classe 7: 0.6389
    - Classe 8: 0.7741

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3005 <<<


--- Epoch 19/30 ---
Risultati Epoch 19:                                                                                                                                                               
  - Training Loss: 0.5160 | Validation Loss: 0.7383 | Validation mIoU: 0.2945
  - IoU per Classe (Validation):
    - Classe 1: 0.3139
    - Classe 2: 0.2427
    - Classe 3: 0.2435
    - Classe 4: 0.0053
    - Classe 5: 0.0262
    - Classe 6: 0.1481
    - Classe 7: 0.5924
    - Classe 8: 0.7844

--- Epoch 20/30 ---
Risultati Epoch 20:                                                                                                                                                               
  - Training Loss: 0.5012 | Validation Loss: 0.7638 | Validation mIoU: 0.3055
  - IoU per Classe (Validation):
    - Classe 1: 0.3243
    - Classe 2: 0.2687
    - Classe 3: 0.2314
    - Classe 4: 0.0050
    - Classe 5: 0.0329
    - Classe 6: 0.1518
    - Classe 7: 0.6566
    - Classe 8: 0.7735

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3055 <<<


--- Epoch 21/30 ---
Risultati Epoch 21:                                                                                                                                                               
  - Training Loss: 0.4418 | Validation Loss: 0.7130 | Validation mIoU: 0.3150
  - IoU per Classe (Validation):
    - Classe 1: 0.3250
    - Classe 2: 0.2575
    - Classe 3: 0.2471
    - Classe 4: 0.0055
    - Classe 5: 0.0330
    - Classe 6: 0.1615
    - Classe 7: 0.7140
    - Classe 8: 0.7765

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3150 <<<


--- Epoch 22/30 ---
Risultati Epoch 22:                                                                                                                                                               
  - Training Loss: 0.4821 | Validation Loss: 0.7234 | Validation mIoU: 0.3056
  - IoU per Classe (Validation):
    - Classe 1: 0.2925
    - Classe 2: 0.2342
    - Classe 3: 0.2569
    - Classe 4: 0.0058
    - Classe 5: 0.0278
    - Classe 6: 0.1583
    - Classe 7: 0.6780
    - Classe 8: 0.7914

--- Epoch 23/30 ---
Risultati Epoch 23:                                                                                                                                                               
  - Training Loss: 0.4522 | Validation Loss: 0.7474 | Validation mIoU: 0.2994
  - IoU per Classe (Validation):
    - Classe 1: 0.3928
    - Classe 2: 0.2139
    - Classe 3: 0.1715
    - Classe 4: 0.0059
    - Classe 5: 0.0322
    - Classe 6: 0.1470
    - Classe 7: 0.6470
    - Classe 8: 0.7848

--- Epoch 24/30 ---
Risultati Epoch 24:                                                                                                                                                               
  - Training Loss: 0.5186 | Validation Loss: 0.6719 | Validation mIoU: 0.2961
  - IoU per Classe (Validation):
    - Classe 1: 0.3517
    - Classe 2: 0.1997
    - Classe 3: 0.2191
    - Classe 4: 0.0053
    - Classe 5: 0.0337
    - Classe 6: 0.1558
    - Classe 7: 0.6287
    - Classe 8: 0.7747

--- Epoch 25/30 ---
Risultati Epoch 25:                                                                                                                                                               
  - Training Loss: 0.4297 | Validation Loss: 0.6631 | Validation mIoU: 0.3117
  - IoU per Classe (Validation):
    - Classe 1: 0.3415
    - Classe 2: 0.2762
    - Classe 3: 0.2345
    - Classe 4: 0.0064
    - Classe 5: 0.0322
    - Classe 6: 0.1576
    - Classe 7: 0.6749
    - Classe 8: 0.7705

--- Epoch 26/30 ---
Risultati Epoch 26:                                                                                                                                                               
  - Training Loss: 0.4223 | Validation Loss: 0.6946 | Validation mIoU: 0.3169
  - IoU per Classe (Validation):
    - Classe 1: 0.3356
    - Classe 2: 0.2959
    - Classe 3: 0.2400
    - Classe 4: 0.0064
    - Classe 5: 0.0299
    - Classe 6: 0.1584
    - Classe 7: 0.6949
    - Classe 8: 0.7739

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3169 <<<


--- Epoch 27/30 ---
Risultati Epoch 27:                                                                                                                                                               
  - Training Loss: 0.3941 | Validation Loss: 0.6946 | Validation mIoU: 0.3190
  - IoU per Classe (Validation):
    - Classe 1: 0.3355
    - Classe 2: 0.3018
    - Classe 3: 0.2361
    - Classe 4: 0.0060
    - Classe 5: 0.0312
    - Classe 6: 0.1605
    - Classe 7: 0.7031
    - Classe 8: 0.7779

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3190 <<<


--- Epoch 28/30 ---
Risultati Epoch 28:                                                                                                                                                               
  - Training Loss: 0.4128 | Validation Loss: 0.7144 | Validation mIoU: 0.3229
  - IoU per Classe (Validation):
    - Classe 1: 0.3336
    - Classe 2: 0.3172
    - Classe 3: 0.2432
    - Classe 4: 0.0056
    - Classe 5: 0.0315
    - Classe 6: 0.1574
    - Classe 7: 0.7149
    - Classe 8: 0.7799

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.3229 <<<


--- Epoch 29/30 ---
Risultati Epoch 29:                                                                                                                                                               
  - Training Loss: 0.4358 | Validation Loss: 0.7138 | Validation mIoU: 0.3171
  - IoU per Classe (Validation):
    - Classe 1: 0.3281
    - Classe 2: 0.3083
    - Classe 3: 0.2472
    - Classe 4: 0.0059
    - Classe 5: 0.0301
    - Classe 6: 0.1545
    - Classe 7: 0.6875
    - Classe 8: 0.7752

--- Epoch 30/30 ---
Risultati Epoch 30:                                                                                                                                                               
  - Training Loss: 0.4179 | Validation Loss: 0.7275 | Validation mIoU: 0.3203
  - IoU per Classe (Validation):
    - Classe 1: 0.3245
    - Classe 2: 0.3160
    - Classe 3: 0.2483
    - Classe 4: 0.0064
    - Classe 5: 0.0306
    - Classe 6: 0.1521
    - Classe 7: 0.7061
    - Classe 8: 0.7789

--- Addestramento Completato ---
Miglior mIoU di validazione ottenuto: 0.3229

'''