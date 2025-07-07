import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from PIL import Image

# Import Kornia e SMP
import kornia.augmentation as K
from kornia.utils import image_to_tensor
import segmentation_models_pytorch as smp

# =============================================================================
# 1. CONFIGURAZIONE MINIMA E INDISPENSABILE
# =============================================================================
config = {
    "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "OUTPUT_SIZE": (384, 768),
    "NUM_CLASSES": 9,
    # Usiamo le medie di ImageNet per questo esempio minimale
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "ENCODER": "efficientnet-b3",
    "LR": 1e-4,
}



from kornia.constants import DataKey, Resample   # <-- enum Import

def get_transforms_kornia() -> torch.nn.Sequential:
    return K.AugmentationSequential(
        # --- trasformazioni ---
        K.RandomHorizontalFlip(p=0.5),
        K.RandomRotation(degrees=15),
        K.ColorJitter(brightness=0.2, contrast=0.2, p=0.8),
        K.Resize(size=config["OUTPUT_SIZE"], align_corners=False),
        K.Normalize(mean=torch.tensor(config["MEAN"]),
                    std=torch.tensor(config["STD"])),

        # --- set comuni ---
        data_keys=["input", "mask"],          # <= vanno bene le stringhe
        extra_args={
            DataKey.MASK: {                   # <= QUI serve l’enum!
                "resample": Resample.NEAREST,
                "align_corners": False,
            }
        },
        same_on_batch=False,
    )


# =============================================================================
# 3. SCRIPT PRINCIPALE
# =============================================================================
if __name__ == "__main__":
    # --- PASSO 1: SETUP ESSENZIALE ---
    print(f"Device in uso: {config['DEVICE']}")
    
    # Modello
    model = smp.Unet(
        encoder_name=config["ENCODER"], 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=config["NUM_CLASSES"]
    ).to(config["DEVICE"])

    # Loss e Optimizer
    # Usiamo una loss semplice per questo test: Cross Entropy
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config["LR"])
    
    # Pipeline di Augmentation
    transforms = get_transforms_kornia().to(config["DEVICE"])
    print("Modello, loss, optimizer e pipeline Kornia creati.")

    # --- PASSO 2: CARICAMENTO DI UNA SOLA IMMAGINE E MASCHERA ---
    # !!! MODIFICA QUESTI PERCORSI CON UN TUO CAMPIONE VALIDO !!!
    try:
        image_path = "./dataset/training/train/0000/rgb.jpg"
        mask_path = "./dataset/training/train/0000/labels.png"
        
        image_pil = Image.open(image_path).convert("RGB")
        mask_pil = Image.open(mask_path)
    except FileNotFoundError:
        print("\n!!! ERRORE !!!")
        print(f"Immagine o maschera non trovata. Assicurati che i percorsi siano corretti.")
        print(f"Percorso immagine cercato: {image_path}")
        print(f"Percorso maschera cercato: {mask_path}")
        exit()

    # Conversione a tensori PyTorch (formato atteso da Kornia)
    image_np = np.array(image_pil)
    mask_np = np.array(mask_pil)
    
    # Immagine: (C, H, W), float, [0, 1]
    image_tensor = image_to_tensor(image_np).float() / 255.0
    # Maschera: (H, W), long
    mask_tensor = torch.from_numpy(mask_np.copy()).long()
    
    print(f"Immagine caricata, shape iniziale: {image_tensor.shape}")
    print(f"Maschera caricata, shape iniziale: {mask_tensor.shape}")
    
    # --- PASSO 3: SIMULAZIONE EPOCA DI TRAINING (1 STEP) ---
    print("\n--- INIZIO TRAINING (1 STEP) ---")
    model.train()

    # Kornia si aspetta un batch. Aggiungiamo una dimensione fittizia (B=1).
    img_batch = image_tensor.unsqueeze(0).to(config["DEVICE"])
    mask_batch = mask_tensor.unsqueeze(0).to(config["DEVICE"])

    # Applica le augmentations
    # L'output sarà un batch di 1 immagine e 1 maschera augmentate
    aug_img, aug_mask = transforms(img_batch, mask_batch)
    
    aug_mask = aug_mask.squeeze(1) 

    # Step di training
    optimizer.zero_grad()
    outputs = model(aug_img)
    loss = criterion(outputs, aug_mask.long()) # La maschera deve essere Long
    loss.backward()
    optimizer.step()
    
    print(f"Step di training completato. Loss: {loss.item():.4f}")
    
    # --- PASSO 4: SIMULAZIONE EPOCA DI VALIDAZIONE (1 STEP) ---
    print("\n--- INIZIO VALIDAZIONE (1 STEP) ---")
    model.eval()
    
    with torch.no_grad():
        # Per la validazione, usiamo la stessa pipeline per semplicità,
        # ma potremmo crearne una separata senza augmentations casuali.
        val_img, val_mask = transforms(img_batch, mask_batch)
        val_mask = val_mask.squeeze(1)
        
        outputs = model(val_img)
        val_loss = criterion(outputs, val_mask.long())
        
        preds = torch.argmax(outputs, dim=1)
        
        # Calcolo mIoU minimale
        intersection = torch.logical_and(val_mask, preds).sum()
        union = torch.logical_or(val_mask, preds).sum()
        iou = (intersection + 1e-6) / (union + 1e-6)
        
        print(f"Step di validazione completato. Loss: {val_loss.item():.4f}, IoU (semplificato): {iou.item():.4f}")

    print("\nSimulazione completata con successo.")
