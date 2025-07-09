# =============================================================================
# SCRIPT DI VALUTAZIONE FINALE
# Questo script carica un modello addestrato e lo valuta su un set di test,
# calcolando il Mean Intersection-over-Union (mIoU) come da specifiche del progetto.
# =============================================================================

# 1. IMPORTS
import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
from collections import OrderedDict
import cv2 # <-- Assicurati che cv2 sia importato

warnings.filterwarnings("ignore", category=UserWarning) # Nasconde warning di SMP

# =============================================================================
# 2. CONFIGURAZIONE
# =============================================================================
MODEL_PATH = "./best_model_b3_08_07_21_00.pth"
TEST_DATA_PATH = "./dataset/testing/test"
VERBOSE = True
NUM_IMAGES_TO_SHOW = 3

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 3. FUNZIONI DI SUPPORTO (Caricamento e Predizione)
# =============================================================================
def load_model_and_transforms(model_path: str, device: torch.device) -> tuple:
    print(f"--- Caricamento del modello da: {model_path} ---")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    config = checkpoint['config']
    print("Configurazione recuperata dal checkpoint:")
    print(f"  - Encoder: {config['ENCODER']}")
    print(f"  - Dimensioni Input: {config['OUTPUT_SIZE']}")
    print(f"  - Numero Classi: {config['NUM_CLASSES']}")
    mean_val = config.get('MEAN', [0.485, 0.456, 0.406])
    std_val = config.get('STD', [0.229, 0.224, 0.225])
    print(f"  - Media: {mean_val}")
    print(f"  - Std Dev: {std_val}")

    model = smp.Unet(
        encoder_name=config["ENCODER"], encoder_weights=None,
        in_channels=3, classes=config["NUM_CLASSES"]
    )

    original_state_dict = checkpoint['model_state_dict']
    new_state_dict = OrderedDict()
    prefix_to_remove = 'unet.'
    for k, v in original_state_dict.items():
        if k.startswith(prefix_to_remove):
            name = k[len(prefix_to_remove):]
            new_state_dict[name] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print("Modello caricato e impostato in modalità valutazione.")

    transforms = A.Compose([
        A.Resize(height=config["OUTPUT_SIZE"][0], width=config["OUTPUT_SIZE"][1]),
        A.Normalize(mean=mean_val, std=std_val),
        ToTensorV2(),
    ])
    print("Pipeline di trasformazione per il test creata.")
    return model, transforms, config

def predict(model: torch.nn.Module, rgb_numpy: np.ndarray, transforms: A.Compose, device: torch.device, original_size: tuple) -> np.ndarray:
    """
    Esegue la predizione su una singola immagine e la riporta alla dimensione originale.
    """
    # 1. Pre-processing
    transformed = transforms(image=rgb_numpy)
    image_tensor = transformed['image'].unsqueeze(0).to(device)

    # 2. Inferenza
    with torch.no_grad():
        outputs = model(image_tensor)

    # 3. Post-processing
    preds = torch.argmax(outputs.squeeze(), dim=0)
    predicted_mask_resized = preds.cpu().numpy().astype(np.uint8)
    
    # 4. Riporta la maschera predetta alla dimensione originale
    original_height, original_width = original_size
    predicted_mask_original_size = cv2.resize(
        predicted_mask_resized, 
        (original_width, original_height), 
        interpolation=cv2.INTER_NEAREST
    )
    return predicted_mask_original_size

# =============================================================================
# 4. FUNZIONI PER LE METRICHE
# =============================================================================
def compute_iou(mask1, mask2, label):
    intersection = np.sum((mask1 == label) & (mask2 == label))
    union = np.sum((mask1 == label) | (mask2 == label))
    if union == 0:
        return np.nan
    return intersection / union

def compute_all_iou(mask1, mask2, num_labels=8):
    iou_scores = np.zeros(num_labels)
    for label_idx, label in enumerate(range(1, num_labels + 1)):
        iou = compute_iou(mask1, mask2, label)
        iou_scores[label_idx] = iou
    return iou_scores

# =============================================================================
# 5. SCRIPT PRINCIPALE DI VALUTAZIONE
# =============================================================================
if __name__ == "__main__":
    model, transforms, config = load_model_and_transforms(MODEL_PATH, DEVICE)

    try:
        samples = sorted([d for d in os.listdir(TEST_DATA_PATH) if os.path.isdir(os.path.join(TEST_DATA_PATH, d))])
        if not samples: raise FileNotFoundError
    except FileNotFoundError:
        print(f"\nERRORE: La directory di test '{TEST_DATA_PATH}' non è stata trovata o è vuota.")
        exit()

    print(f"\n--- Inizio valutazione su {len(samples)} campioni di test ---")

    all_ious = np.zeros((len(samples), config["NUM_CLASSES"] - 1))
    images_shown = 0

    for i, subdir in tqdm(enumerate(samples), total=len(samples), desc="Valutazione campioni"):
        subdir_path = os.path.join(TEST_DATA_PATH, subdir)
        rgb_path = os.path.join(subdir_path, 'rgb.jpg')
        labels_path = os.path.join(subdir_path, 'labels.png')

        if not os.path.exists(rgb_path) or not os.path.exists(labels_path):
            print(f"Attenzione: file mancanti in {subdir_path}. Campione saltato.")
            all_ious[i, :] = np.nan
            continue

        try:
            rgb_image = Image.open(rgb_path).convert("RGB")
            rgb_array = np.array(rgb_image)
            labels_array = np.array(Image.open(labels_path))

            # Passa le dimensioni originali a `predict`
            original_h, original_w = rgb_array.shape[:2]
            predicted_labels_array = predict(model, rgb_array, transforms, DEVICE, original_size=(original_h, original_w))
            
            iou_scores = compute_all_iou(labels_array, predicted_labels_array, num_labels=config["NUM_CLASSES"] - 1)
            all_ious[i, :] = iou_scores

            if VERBOSE and images_shown < NUM_IMAGES_TO_SHOW:
                fig, axes = plt.subplots(1, 3, figsize=(20, 6))
                axes[0].imshow(rgb_image); axes[0].set_title(f"Input: {subdir}"); axes[0].axis('off')
                axes[1].imshow(labels_array, cmap='jet', vmin=0, vmax=config['NUM_CLASSES'] - 1); axes[1].set_title("Ground Truth"); axes[1].axis('off')
                axes[2].imshow(predicted_labels_array, cmap='jet', vmin=0, vmax=config['NUM_CLASSES'] - 1); axes[2].set_title(f"Predizione (mIoU: {np.nanmean(iou_scores):.3f})"); axes[2].axis('off')
                plt.tight_layout(); plt.show()
                images_shown += 1

        except Exception as e:
            print(f"Errore durante l'elaborazione del campione {subdir}: {e}")
            all_ious[i, :] = np.nan

    print("\n\n--- RISULTATI DELLA VALUTAZIONE ---")
    
    # Gestisce il caso in cui tutti i campioni falliscono
    if np.all(np.isnan(all_ious)):
        print("\nATTENZIONE: Nessun campione è stato valutato con successo. Impossibile calcolare il punteggio.")
    else:
        final_score = np.nanmean(all_ious)
        print(f"\n Punteggio Finale (Mean IoU Globale): {final_score:.4f}")

        mean_iou_per_class = np.nanmean(all_ious, axis=0)
        print("\n Mean IoU per Classe:")
        class_names = config.get("CLASS_NAMES", [f"Classe {i}" for i in range(config['NUM_CLASSES'])])
        for i, iou in enumerate(mean_iou_per_class):
            print(f"  - {class_names[i+1]}: {iou:.4f}")

    print("\n--- Valutazione Completata ---")