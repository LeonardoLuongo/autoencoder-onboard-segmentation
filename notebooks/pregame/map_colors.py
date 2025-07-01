# File: create_color_map.py
import os
import glob
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

DATASET_PATH = "./dataset/training/train"
OUTPUT_FILE = "color_map.json"

def discover_and_save_mappings(dataset_path: str, output_file: str):
    """
    Scansiona le maschere, scopre i colori unici e salva le mappature
    in un file JSON.
    """
    print(f"Scansione del dataset in '{dataset_path}'...")
    unique_colors = set()
    mask_files = glob.glob(os.path.join(dataset_path, "*", "labels.png"))

    if not mask_files:
        print("ERRORE: Nessun file 'labels.png' trovato. Controlla il percorso del dataset.")
        return

    for mask_file in tqdm(mask_files, desc="Analisi Maschere"):
        mask_img = Image.open(mask_file).convert('RGB')
        mask_np = np.array(mask_img)
        pixels = mask_np.reshape(-1, 3)
        unique_colors.update(map(tuple, np.unique(pixels, axis=0)))

    sorted_colors = sorted(list(unique_colors))
    
    id_to_color = {i: color for i, color in enumerate(sorted_colors)}
    color_to_id = {str(color): i for i, color in enumerate(sorted_colors)} # Chiavi JSON devono essere stringhe

    # Trova l'indice per il background (nero o bianco)
    ignore_index = -100 # Valore di default
    if str((0, 0, 0)) in color_to_id:
        ignore_index = color_to_id[str((0, 0, 0))]
    elif str((255, 255, 255)) in color_to_id:
        ignore_index = color_to_id[str((255, 255, 255))]

    # Crea il pacchetto dati da salvare
    config_data = {
        "num_classes": len(sorted_colors),
        "ignore_index": ignore_index,
        "id_to_color": id_to_color,
        "color_to_id": color_to_id
    }

    with open(output_file, 'w') as f:
        json.dump(config_data, f, indent=4)
        
    print(f"\nScoperta completata. Trovate {config_data['num_classes']} classi.")
    print(f"Mappatura salvata con successo in '{output_file}'.")
    print("\nDettagli Mappa:")
    for class_id, color in id_to_color.items():
        print(f"  ID: {class_id} -> Colore: {color}")

if __name__ == "__main__":
    discover_and_save_mappings(DATASET_PATH, OUTPUT_FILE)