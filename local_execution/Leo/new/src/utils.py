# src/utils.py
import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def seed_everything(seed: int):
    """Imposta il seed per la riproducibilità."""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    tqdm.write(f"Global seed impostato a {seed}")

def plot_confusion_matrix(conf_matrix_np, class_names, ignore_index, save_path):
    """Salva e visualizza la matrice di confusione normalizzata."""
    tqdm.write("Visualizzazione e salvataggio Matrice di Confusione...")
    plot_matrix, plot_names = conf_matrix_np.copy(), class_names.copy()
    if ignore_index is not None and 0 <= ignore_index < len(class_names):
        plot_matrix = np.delete(np.delete(plot_matrix, ignore_index, axis=0), ignore_index, axis=1)
        plot_names = [name for i, name in enumerate(class_names) if i != ignore_index]
    
    row_sums = plot_matrix.sum(axis=1)[:, np.newaxis]
    norm_conf_matrix = np.divide(plot_matrix, row_sums, out=np.zeros_like(plot_matrix, dtype=float), where=row_sums != 0)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(norm_conf_matrix, annot=True, fmt='.2%', cmap='Blues', xticklabels=plot_names, yticklabels=plot_names, ax=ax)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    ax.set_title('Matrice di Confusione (Normalizzata per Riga)', fontsize=14)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    tqdm.write(f"Matrice di confusione salvata in '{save_path}'")