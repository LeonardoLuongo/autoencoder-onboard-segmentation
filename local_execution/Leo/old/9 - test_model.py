import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import segmentation_models_pytorch as smp
import cv2
import matplotlib.pyplot as plt

# =============================================================================
# FUNZIONE DI PREDIZIONE - REQUISITO DEL PROGETTO
# =============================================================================
def predict(X: np.ndarray, model: torch.nn.Module, device: torch.device, output_size: tuple) -> np.ndarray:
    """
    Esegue la predizione di una maschera di segmentazione su una singola immagine.

    Args:
        X (np.ndarray): L'immagine di input come array NumPy (H, W, C) in formato RGB.
        model (torch.nn.Module): Il modello di segmentazione addestrato.
        device (torch.device): Il device su cui eseguire la predizione (es. 'cuda' o 'cpu').
        output_size (tuple): La dimensione (H, W) a cui l'immagine deve essere ridimensionata
                             prima di essere data in input al modello.

    Returns:
        np.ndarray: La maschera di predizione come array NumPy (H, W) con gli ID di classe.
    """
    model.eval()

    # 1. Definire le trasformazioni di preprocessing (devono essere IDENTICHE a quelle di validazione)
    transforms = A.Compose([
        A.Resize(height=output_size[0], width=output_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    # 2. Preprocessare l'immagine
    processed = transforms(image=X)
    image_tensor = processed['image'].unsqueeze(0).to(device) # Aggiunge la dimensione del batch

    # 3. Eseguire la predizione
    with torch.no_grad():
        logits = model(image_tensor)
        
    # 4. Post-processare l'output
    # Prende l'argmax sul canale delle classi per ottenere la maschera di indici
    pred_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy().astype(np.uint8)

    # 5. Riporta la maschera alla dimensione originale dell'immagine di input
    original_h, original_w = X.shape[:2]
    pred_mask_resized = cv2.resize(
        pred_mask, 
        (original_w, original_h), 
        interpolation=cv2.INTER_NEAREST # Fondamentale usare NEAREST per non creare ID spuri
    )
    
    return pred_mask_resized

# =============================================================================
# ESEMPIO D'USO
# =============================================================================
if __name__ == '__main__':
    # --- CONFIGURAZIONE ---
    MODEL_PATH = './best_model_b5_tversky_2stage.pth' # Assicurati che il percorso sia corretto
    IMAGE_PATH = './dataset/training/train/1347'     # Inserisci il percorso di un'immagine di test

    # --- CARICAMENTO MODELLO E CONFIGURAZIONE ---
    print(f"Caricamento del modello da {MODEL_PATH}...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Carica il payload salvato, che contiene sia i pesi che la configurazione
    payload = torch.load(MODEL_PATH, map_location=device)
    config = payload['config']
    
    # Ricrea il modello usando la configurazione salvata
    model = smp.Unet(
        encoder_name=config["ENCODER"],
        encoder_weights=None, # I pesi sono già nello state_dict
        in_channels=3,
        classes=config["NUM_CLASSES"]
    )
    model.load_state_dict(payload['model_state_dict'])
    model.to(device)
    print("Modello caricato con successo.")

    # --- ESECUZIONE DELLA PREDIZIONE ---
    try:
        print(f"Caricamento immagine di test da {IMAGE_PATH}...")
        input_image_rgb = np.array(Image.open(IMAGE_PATH).convert("RGB"))

        print("Esecuzione predizione...")
        predicted_mask = predict(input_image_rgb, model, device, config['OUTPUT_SIZE'])
        
        print(f"Predizione completata. La maschera risultante ha dimensioni {predicted_mask.shape} e tipo {predicted_mask.dtype}.")
        print(f"Valori unici nella maschera predetta: {np.unique(predicted_mask)}")

        # --- VISUALIZZAZIONE (OPZIONALE) ---
        # Per visualizzare, potremmo creare una funzione che mappa gli ID di classe ai colori
        def create_colored_mask(mask, num_classes):
            cmap = plt.get_cmap('nipy_spectral', num_classes)
            colored_mask = cmap(mask / (num_classes - 1))
            return (colored_mask[:, :, :3] * 255).astype(np.uint8)

        predicted_mask_colored = Image.fromarray(create_colored_mask(predicted_mask, config['NUM_CLASSES']))
        
        # Salva la maschera colorata
        output_mask_path = 'predicted_mask.png'
        predicted_mask_colored.save(output_mask_path)
        print(f"Maschera di predizione colorata salvata in: {output_mask_path}")

    except FileNotFoundError:
        print(f"ERRORE: Immagine di test non trovata in '{IMAGE_PATH}'. Modifica il percorso e riprova.")
    except Exception as e:
        print(f"Si è verificato un errore: {e}")