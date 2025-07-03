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
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from PIL import Image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# =============================================================================
# 2. CONFIGURAZIONE PER LO SPECIALISTA
# =============================================================================
def get_specialist_config() -> Dict[str, Any]:
    """Configurazione specifica per addestrare un modello specialista binario."""
    base_path = "./" 
    
    config = {
        "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        "DATASET_PATH": os.path.join(base_path, "dataset/training/train"),
        
        # --- CLASSE TARGET E PATH DEL MODELLO ---
        "TARGET_CLASS_ID": 4,  # Classe che vogliamo imparare (4 = traversable grass)
        "TARGET_CLASS_NAME": "traversable_grass",
        "MODEL_SAVE_PATH": os.path.join(base_path, f"specialist_{'traversable_grass'}.pth"),

        # Iperparametri di addestramento
        "NUM_EPOCHS": 50,         # Diamo più epoche, i dati sono pochi
        "BATCH_SIZE": 8,
        "LEARNING_RATE": 1e-3,    # LR più alto, addestriamo da zero
        
        # Parametri del dataset e augmentation
        "OUTPUT_SIZE": (544, 1024),
        "RANDOM_SEED": 42,
        "VALIDATION_SPLIT": 0.2,
        
        # Parametri di performance
        "USE_AMP": True,
        "NUM_WORKERS": 0,
    }
    return config

import albumentations as A
from albumentations.pytorch import ToTensorV2
from typing import Tuple

def get_transforms(is_train: bool, output_size: Tuple[int, int]) -> A.Compose:
    """
    Restituisce la pipeline di augmentation/trasformazione appropriata.
    Utilizza una normalizzazione standard di ImageNet, ideale per encoder pre-addestrati.
    
    Args:
        is_train (bool): Se True, restituisce le trasformazioni per il training set (con augmentation).
                         Se False, restituisce solo resize e normalizzazione per il validation set.
        output_size (Tuple[int, int]): La dimensione (H, W) a cui ridimensionare le immagini.
                         
    Returns:
        A.Compose: La pipeline di trasformazione di Albumentations.
    """
    # Usiamo media e deviazione standard di ImageNet, dato che l'encoder è pre-addestrato su di esso.
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    if is_train:
        # Pipeline di augmentation aggressiva per il training
        return A.Compose([
            A.Resize(height=output_size[0], width=output_size[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.15, rotate_limit=25, p=0.7, border_mode=0),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
            A.GaussNoise(p=0.3),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, fill_value=0, mask_fill_value=0, p=0.4),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else: # Per il Validation Set
        # Solo resize e normalizzazione, nessuna augmentation.
        return A.Compose([
            A.Resize(height=output_size[0], width=output_size[1]),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

def get_specialist_split(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    Filtra il dataset per le immagini contenenti la classe target e poi
    esegue uno split casuale semplice.
    
    Args:
        config (Dict): Il dizionario di configurazione.
        
    Returns:
        Tuple[List[str], List[str]]: Due liste contenenti i percorsi per il 
                                     training e la validazione dello specialista.
    """
    dataset_path = config["DATASET_PATH"]
    target_class_id = config["TARGET_CLASS_ID"]
    target_class_name = config["TARGET_CLASS_NAME"]
    
    print(f"Ricerca campioni per lo specialista della classe '{target_class_name}' (ID: {target_class_id})...")
    
    # 1. Trova tutti i percorsi validi
    all_dirs = sorted([
        os.path.join(dataset_path, d) for d in os.listdir(dataset_path) 
        if os.path.isdir(os.path.join(dataset_path, d)) and d.isdigit()
    ])
    
    # 2. Filtra per mantenere solo quelli con la classe target
    target_class_dirs = []
    for sample_dir in tqdm(all_dirs, desc="Filtraggio campioni"):
        mask_path = os.path.join(sample_dir, 'labels.png')
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path))
            if target_class_id in np.unique(mask):
                target_class_dirs.append(sample_dir)
                
    if not target_class_dirs:
        raise ValueError(f"Nessuna immagine trovata per la classe target {target_class_id}. Impossibile addestrare lo specialista.")
        
    print(f"Trovate {len(target_class_dirs)} immagini contenenti la classe '{target_class_name}'.")

    # 3. Esegui uno split casuale semplice su questo sottoinsieme
    train_dirs, val_dirs = train_test_split(
        target_class_dirs,
        test_size=config["VALIDATION_SPLIT"],
        random_state=config["RANDOM_SEED"]
    )
    
    print(f"Split per lo specialista creato: {len(train_dirs)} train, {len(val_dirs)} validation.")
    
    return train_dirs, val_dirs

# =============================================================================
# 3. ARCHITETTURA UNetMicro E BLOCCHI
# =============================================================================
class DoubleConv(nn.Module):
    """(Convolution => BatchNorm => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling con MaxPool2d seguito da DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling (con ConvTranspose2d) seguito da DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Usiamo ConvTranspose2d per l'upsampling, che è un layer addestrabile
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 è il tensore dal percorso di upscaling
        # x2 è la skip connection dal percorso di encoding
        x1 = self.up(x1)
        # Concatena la skip connection con il tensore upscalato
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """Convoluzione 1x1 finale per mappare ai canali delle classi"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNetMicro(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetMicro, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Canali di base molto più piccoli
        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        # Rimuoviamo anche il terzo stadio di downsampling per ora
        # self.down3 = Down(64, 128) 
        
        # Il punto più profondo ora ha 64 canali
        
        # Decoder
        # self.up1 = Up(128, 64)
        self.up2 = Up(64, 32)
        self.up3 = Up(32, 16)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        # Percorso dell'encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        # x4 = self.down3(x3)
        
        # Percorso del decoder
        # x = self.up1(x4, x3)
        x = self.up2(x3, x2) # Parte da x3, la skip è x2
        x = self.up3(x, x1)  # Continua con x1
        
        logits = self.outc(x)
        return logits

# =============================================================================
# 4. DATASET BINARIO
# =============================================================================
class BinarySegmentationDataset(Dataset):
    """Dataset che converte le maschere in formato binario (classe target vs resto)."""
    def __init__(self, sample_dirs: List[str], target_class: int, transform: A.Compose):
        self.sample_dirs = sample_dirs
        self.target_class = target_class
        self.transform = transform

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        img_path = os.path.join(sample_dir, 'rgb.jpg')
        mask_path = os.path.join(sample_dir, 'labels.png')
        
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))

        # Conversione in maschera binaria
        binary_mask = (mask == self.target_class).astype(np.uint8)
        
        augmented = self.transform(image=image, mask=binary_mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].unsqueeze(0) # Aggiunge dimensione canale
        
        return image_tensor.float(), mask_tensor.float() # La loss BCE vuole float

# =============================================================================
# 5. FUNZIONI DI TRAINING E VALIDAZIONE (semplificate per il binario)
# =============================================================================
def train_specialist_epoch(model, loader, optimizer, criterion, scaler, device, use_amp):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(loader, desc="Training Specialista")
    for images, masks in progress_bar:
        images, masks = images.to(device), masks.to(device)
        
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

def evaluate_specialist(model, loader, device):
    """Valuta lo specialista usando l'IoU sulla classe target."""
    model.eval()
    total_iou = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in tqdm(loader, desc="Validazione Specialista", leave=False):
            images, masks = images.to(device), masks.to(device)
            
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).int() # Applica sigmoide e soglia
            
            # Calcola IoU
            intersection = torch.logical_and(preds, masks).sum()
            union = torch.logical_or(preds, masks).sum()
            iou = (intersection + 1e-6) / (union + 1e-6) # Aggiungo epsilon per evitare divisione per zero
            
            total_iou += iou.item() * images.size(0)
            num_samples += images.size(0)
            
    return total_iou / num_samples if num_samples > 0 else 0

# =============================================================================
# 6. FUNZIONE PRINCIPALE (MAIN)
# =============================================================================
def main():
    config = get_specialist_config()
    device = config["DEVICE"]

    # Split dei soli dati pertinenti
    train_dirs, val_dirs = get_specialist_split(config)
    
    train_transforms = get_transforms(True, config["OUTPUT_SIZE"])
    val_transforms = get_transforms(False, config["OUTPUT_SIZE"])

    train_dataset = BinarySegmentationDataset(train_dirs, config["TARGET_CLASS_ID"], train_transforms)
    val_dataset = BinarySegmentationDataset(val_dirs, config["TARGET_CLASS_ID"], val_transforms)
    
    train_loader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True, num_workers=config["NUM_WORKERS"])
    val_loader = DataLoader(val_dataset, batch_size=config["BATCH_SIZE"], shuffle=False, num_workers=config["NUM_WORKERS"])

    # Modello specialista (1 canale di output)
    model = UNetMicro(n_channels=3, n_classes=1).to(device)
    
    # Loss per classificazione binaria
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["LEARNING_RATE"])
    scaler = torch.amp.GradScaler(enabled=config["USE_AMP"])

    best_iou = -1.0
    print(f"\n--- Inizio Addestramento Specialista per Classe {config['TARGET_CLASS_NAME']} ---")
    for epoch in range(config["NUM_EPOCHS"]):
        train_loss = train_specialist_epoch(model, train_loader, optimizer, criterion, scaler, device, config["USE_AMP"])
        val_iou = evaluate_specialist(model, val_loader, device)
        
        print(f"Epoch {epoch+1}/{config['NUM_EPOCHS']} -> Train Loss: {train_loss:.4f} | Validation IoU: {val_iou:.4f}")
        
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), config["MODEL_SAVE_PATH"])
            print(f"🎉 Modello specialista migliore salvato con IoU: {best_iou:.4f}")

    print(f"\n--- Addestramento Completato --- \nMiglior IoU di validazione: {best_iou:.4f}")

# =============================================================================
# 7. ESECUZIONE
# =============================================================================
if __name__ == "__main__":
    # Incolla qui le classi UNetMicro, DoubleConv, Down, Up, OutConv
    main()


'''

--- Inizio Addestramento Specialista per Classe traversable_grass ---
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00,  1.94it/s, loss=0.6147] 
Epoch 1/50 -> Train Loss: 0.5995 | Validation IoU: 0.0000                                                                                                                         
🎉 Modello specialista migliore salvato con IoU: 0.0000
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.74it/s, loss=0.5323] 
Epoch 2/50 -> Train Loss: 0.5460 | Validation IoU: 0.0000                                                                                                                         
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.98it/s, loss=0.5997] 
Epoch 3/50 -> Train Loss: 0.5214 | Validation IoU: 0.0000                                                                                                                         
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.93it/s, loss=0.4291] 
Epoch 4/50 -> Train Loss: 0.4683 | Validation IoU: 0.0000                                                                                                                         
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.95it/s, loss=0.4298] 
Epoch 5/50 -> Train Loss: 0.4476 | Validation IoU: 0.0000                                                                                                                         
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.83it/s, loss=0.4365] 
Epoch 6/50 -> Train Loss: 0.4361 | Validation IoU: 0.0000                                                                                                                         
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.91it/s, loss=0.3877] 
Epoch 7/50 -> Train Loss: 0.4152 | Validation IoU: 0.0000                                                                                                                         
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.95it/s, loss=0.4236] 
Epoch 8/50 -> Train Loss: 0.4142 | Validation IoU: 0.0000                                                                                                                         
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.85it/s, loss=0.3770] 
Epoch 9/50 -> Train Loss: 0.3863 | Validation IoU: 0.0000                                                                                                                         
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.92it/s, loss=0.4619] 
Epoch 10/50 -> Train Loss: 0.3898 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.07it/s, loss=0.3508] 
Epoch 11/50 -> Train Loss: 0.3650 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.90it/s, loss=0.3501] 
Epoch 12/50 -> Train Loss: 0.3528 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.94it/s, loss=0.3343] 
Epoch 13/50 -> Train Loss: 0.3488 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.00it/s, loss=0.3023] 
Epoch 14/50 -> Train Loss: 0.3344 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.79it/s, loss=0.4581] 
Epoch 15/50 -> Train Loss: 0.3560 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.98it/s, loss=0.2776] 
Epoch 16/50 -> Train Loss: 0.3152 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.00it/s, loss=0.2665] 
Epoch 17/50 -> Train Loss: 0.3078 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.82it/s, loss=0.2834] 
Epoch 18/50 -> Train Loss: 0.3032 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.71it/s, loss=0.4628] 
Epoch 19/50 -> Train Loss: 0.3280 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.03it/s, loss=0.3723] 
Epoch 20/50 -> Train Loss: 0.3025 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.92it/s, loss=0.2459] 
Epoch 21/50 -> Train Loss: 0.2770 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.83it/s, loss=0.2833] 
Epoch 22/50 -> Train Loss: 0.2820 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.89it/s, loss=0.2215] 
Epoch 23/50 -> Train Loss: 0.2628 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.90it/s, loss=0.2159] 
Epoch 24/50 -> Train Loss: 0.2512 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.08it/s, loss=0.3550] 
Epoch 25/50 -> Train Loss: 0.2792 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.75it/s, loss=0.2435] 
Epoch 26/50 -> Train Loss: 0.2549 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.13it/s, loss=0.2112] 
Epoch 27/50 -> Train Loss: 0.2475 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.99it/s, loss=0.2154] 
Epoch 28/50 -> Train Loss: 0.2405 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.84it/s, loss=0.1918] 
Epoch 29/50 -> Train Loss: 0.2293 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.84it/s, loss=0.3972] 
Epoch 30/50 -> Train Loss: 0.2702 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.69it/s, loss=0.3116] 
Epoch 31/50 -> Train Loss: 0.2562 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.66it/s, loss=0.1699] 
Epoch 32/50 -> Train Loss: 0.2220 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00,  2.48it/s, loss=0.2066] 
Epoch 33/50 -> Train Loss: 0.2222 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:02<00:00,  2.45it/s, loss=0.4486] 
Epoch 34/50 -> Train Loss: 0.2686 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.69it/s, loss=0.1548] 
Epoch 35/50 -> Train Loss: 0.2179 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.78it/s, loss=0.1860] 
Epoch 36/50 -> Train Loss: 0.2186 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.92it/s, loss=0.1538] 
Epoch 37/50 -> Train Loss: 0.2070 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.99it/s, loss=0.1400] 
Epoch 38/50 -> Train Loss: 0.2023 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.95it/s, loss=0.1792] 
Epoch 39/50 -> Train Loss: 0.2089 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.78it/s, loss=0.2866] 
Epoch 40/50 -> Train Loss: 0.2202 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.93it/s, loss=0.1453] 
Epoch 41/50 -> Train Loss: 0.1884 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.94it/s, loss=0.1264] 
Epoch 42/50 -> Train Loss: 0.1848 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.80it/s, loss=0.2301] 
Epoch 43/50 -> Train Loss: 0.2073 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.86it/s, loss=0.4022] 
Epoch 44/50 -> Train Loss: 0.2329 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.02it/s, loss=0.1720] 
Epoch 45/50 -> Train Loss: 0.1944 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.01it/s, loss=0.3724] 
Epoch 46/50 -> Train Loss: 0.2294 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.02it/s, loss=0.1189] 
Epoch 47/50 -> Train Loss: 0.1840 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  3.05it/s, loss=0.1195] 
Epoch 48/50 -> Train Loss: 0.1805 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.95it/s, loss=0.2982] 
Epoch 49/50 -> Train Loss: 0.2111 | Validation IoU: 0.0000                                                                                                                        
Training Specialista: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:01<00:00,  2.70it/s, loss=0.1495] 
Epoch 50/50 -> Train Loss: 0.1802 | Validation IoU: 0.0000                                                                                                                        

--- Addestramento Completato ---
Miglior IoU di validazione: 0.0000

'''