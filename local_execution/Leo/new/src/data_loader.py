# src/data_loader.py
import os
import glob
from typing import Tuple, List, Dict, Any
import cv2
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
from tqdm import tqdm
import imagehash
import networkx as nx
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    """Dataset per la segmentazione semantica."""
    def __init__(self, sample_dirs: List[str], transform: A.Compose, num_classes: int):
        self.sample_dirs = sample_dirs
        self.transform = transform
        self.num_classes = num_classes

    def __len__(self) -> int:
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample_dir = self.sample_dirs[idx]
        image_path = os.path.join(sample_dir, 'rgb.jpg')
        mask_path = os.path.join(sample_dir, 'labels.png')
        
        image = np.array(Image.open(image_path).convert("RGB"))
        mask = np.array(Image.open(mask_path))
        
        # Controlli di integrità
        assert mask.dtype == np.uint8, f"Tipo maschera non corretto: {mask.dtype} in {mask_path}"
        assert mask.max() < self.num_classes, f"ID classe {mask.max()} non valido in {mask_path} (Num classes: {self.num_classes})"
        
        augmented = self.transform(image=image, mask=mask)
        return augmented['image'], augmented['mask']

def get_transforms(cfg_aug: Dict, output_size: List[int], mean: List[float], std: List[float]) -> A.Compose:
    """Costruisce dinamicamente la pipeline di augmentation da una configurazione."""
    transforms_list = [A.Resize(height=output_size[0], width=output_size[1], interpolation=cv2.INTER_NEAREST)]
    
    for aug_item in cfg_aug:
        # Usa getattr per ottenere la classe di augmentation dal modulo albumentations
        aug_class = getattr(A, aug_item['name'])
        transforms_list.append(aug_class(**aug_item['params']))
        
    transforms_list.extend([
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
    return A.Compose(transforms_list)

def calculate_mean_std(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Calcola media e deviazione standard in modo numericamente stabile."""
    tqdm.write("Calcolo di media e deviazione standard in corso...")
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    pixel_count = 0

    for images, _ in tqdm(loader, desc="Analisi Statistiche Dataset"):
        batch_size, _, height, width = images.shape
        psum += images.sum(axis=[0, 2, 3])
        psum_sq += (images ** 2).sum(axis=[0, 2, 3])
        pixel_count += batch_size * height * width

    total_mean = psum / pixel_count
    total_var = (psum_sq / pixel_count) - (total_mean ** 2)
    total_std = torch.sqrt(torch.clamp(total_var, min=1e-6))
    
    mean_np, std_np = total_mean.numpy(), total_std.numpy()
    tqdm.write(f"Media calcolata (R, G, B): {mean_np}")
    tqdm.write(f"Std Dev calcolata (R, G, B): {std_np}")
    return mean_np, std_np

def get_grouped_stratified_split(config: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Esegue lo split dei dati raggruppando immagini simili e stratificando su classi rare."""
    all_dirs = []
    tqdm.write("Caricamento percorsi dei dataset...")
    for path in config['paths']['dataset_sources']:
        if not os.path.isdir(path):
            tqdm.write(f"Attenzione: il percorso '{path}' non esiste, verrà saltato.")
            continue
        dirs = [d for d in glob.glob(os.path.join(path, "*")) if os.path.isdir(d)]
        tqdm.write(f"Trovati {len(dirs)} campioni in '{path}'.")
        all_dirs.extend(dirs)
        
    if not all_dirs: raise FileNotFoundError("Nessuna directory di dati trovata nei percorsi specificati.")
    all_dirs = sorted(list(set(all_dirs)))

    hashes = {path: imagehash.phash(Image.open(os.path.join(path, 'rgb.jpg'))) for path in tqdm(all_dirs, desc="Calcolo Hash Immagini")}
    
    G = nx.Graph()
    G.add_nodes_from(hashes.keys())
    
    # Ottimizzazione del confronto (come nello script originale)
    buckets = defaultdict(list)
    for path, h in hashes.items(): buckets[int(str(h)[:2], 16)].append((path, h))
    tqdm.write("Confronto immagini simili per raggruppamento...")
    for bucket_items in tqdm(buckets.values(), desc="Analisi Bucket"):
        if len(bucket_items) < 2: continue
        for i in range(len(bucket_items)):
            for j in range(i + 1, len(bucket_items)):
                p1, h1 = bucket_items[i]
                p2, h2 = bucket_items[j]
                if h1 - h2 < config['data']['split']['hash_similarity_threshold']:
                    G.add_edge(p1, p2)

    image_groups = list(nx.connected_components(G))
    tqdm.write(f"Trovati {len(image_groups)} gruppi di immagini uniche.")

    # Stratificazione
    rare_class_indices = config['data']['split']['stratify_on_rare_classes']
    group_labels = []
    for group in tqdm(image_groups, desc="Etichettatura gruppi per lo split"):
        has_rare_class = any(
            rare_id in np.unique(np.array(Image.open(os.path.join(p, 'labels.png'))))
            for p in group
            for rare_id in rare_class_indices
        )
        group_labels.append(1 if has_rare_class else 0)

    train_groups, val_groups = train_test_split(
        image_groups,
        test_size=config['data']['split']['validation_fraction'],
        random_state=config['seed'],
        stratify=group_labels
    )
    train_dirs = [p for group in train_groups for p in group]
    val_dirs = [p for group in val_groups for p in group]
    tqdm.write(f"Split stratificato creato: {len(train_dirs)} train, {len(val_dirs)} validation.")
    return train_dirs, val_dirs