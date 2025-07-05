# src/config_parser.py
import yaml
import os
from typing import Dict, Any
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data_loader import SegmentationDataset, calculate_mean_std

def load_and_parse_config(config_path: str) -> Dict[str, Any]:
    """Carica il file di configurazione YAML e lo processa."""
    tqdm.write(f"Caricamento configurazione da: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # --- Processa i percorsi ---
    base_path = config.get('base_path', './')
    config['paths']['model_save'] = os.path.join(base_path, config['paths']['model_save'])
    config['paths']['plot_save'] = os.path.join(base_path, config['paths']['plot_save'])
    config['paths']['dataset_sources'] = [os.path.join(base_path, p) for p in config['paths']['dataset_sources']]

    # --- Imposta il device ---
    if config['device'] == 'auto':
        config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        config['device'] = torch.device(config['device'])
    
    # --- Calcola media e deviazione standard se necessario ---
    if config['data']['stats']['mean'] == 'auto' or config['data']['stats']['std'] == 'auto':
        tqdm.write("Rilevata configurazione 'auto' per le statistiche del dataset.")
        # Usa solo il primo (presumibilmente originale) set di dati per il calcolo
        original_data_path = config['paths']['dataset_sources'][0]
        stats_dirs = [d for d in glob.glob(os.path.join(original_data_path, "*")) if os.path.isdir(d)]
        
        stats_transforms = A.Compose([
            A.Resize(height=config['data']['output_size'][0], width=config['data']['output_size'][1]),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
            ToTensorV2(),
        ])
        
        stats_dataset = SegmentationDataset(stats_dirs, stats_transforms, config['data']['num_classes'])
        stats_loader = DataLoader(
            stats_dataset, 
            batch_size=config['data']['loader']['batch_size'] * 2,
            shuffle=False, 
            num_workers=config['data']['loader']['num_workers_train']
        )
        
        mean, std = calculate_mean_std(stats_loader)
        config['data']['stats']['mean'] = mean.tolist()
        config['data']['stats']['std'] = std.tolist()
        tqdm.write("Configurazione aggiornata con le statistiche calcolate.")
        tqdm.write("Per avvii futuri, puoi inserire questi valori nel file YAML per saltare questo passaggio:")
        tqdm.write(f"  mean: {config['data']['stats']['mean']}")
        tqdm.write(f"  std: {config['data']['stats']['std']}")
        
    return config