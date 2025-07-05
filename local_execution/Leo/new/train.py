# train.py
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from src.config_parser import load_and_parse_config
from src.utils import seed_everything, plot_confusion_matrix
from src.data_loader import get_grouped_stratified_split, get_transforms, SegmentationDataset
from src.model_builder import get_model, get_optimizer, get_scheduler, get_criterion
from src.engine import train_one_epoch, evaluate, GradScaler

def main(config_path: str):
    # 1. Carica e processa la configurazione
    config = load_and_parse_config(config_path)
    
    # 2. Imposta la riproducibilità e il device
    seed_everything(config['seed'])
    device = config['device']
    use_amp = config['use_amp'] and device.type == 'cuda'
    tqdm.write(f"Device in uso: {device} | Mixed Precision (AMP): {'Abilitato' if use_amp else 'Disabilitato'}")

    # 3. Preparazione dei dati
    train_dirs, val_dirs = get_grouped_stratified_split(config)
    
    train_transforms = get_transforms(
        config['augmentations']['train'],
        config['data']['output_size'],
        config['data']['stats']['mean'],
        config['data']['stats']['std']
    )
    val_transforms = get_transforms(
        config['augmentations']['validation'],
        config['data']['output_size'],
        config['data']['stats']['mean'],
        config['data']['stats']['std']
    )

    train_dataset = SegmentationDataset(train_dirs, train_transforms, config['data']['num_classes'])
    val_dataset = SegmentationDataset(val_dirs, val_transforms, config['data']['num_classes'])

    pin_mem = (device.type == 'cuda')
    train_loader = DataLoader(train_dataset, batch_size=config['data']['loader']['batch_size'], shuffle=True, 
                              num_workers=config['data']['loader']['num_workers_train'], pin_memory=pin_mem)
    val_loader = DataLoader(val_dataset, batch_size=config['data']['loader']['batch_size'], shuffle=False, 
                            num_workers=config['data']['loader']['num_workers_val'], pin_memory=pin_mem)

    # 4. Costruzione del modello e dei componenti di training
    model = get_model(config).to(device)
    criterion = get_criterion(config).to(device)
    scaler = GradScaler(enabled=use_amp)
    
    # 5. Ciclo di training multi-fase
    best_iou = -1.0
    best_conf_matrix = None
    global_epoch = 0

    for phase_idx, phase_config in enumerate(config['training_phases']):
        tqdm.write(f"\n--- INIZIO {phase_config['name']} (Fase {phase_idx + 1}/{len(config['training_phases'])}) ---")
        
        # Crea un nuovo optimizer e scheduler per ogni fase
        optimizer = get_optimizer(model, phase_config)
        scheduler = get_scheduler(optimizer, phase_config)
        
        for epoch in range(phase_config['epochs']):
            global_epoch += 1
            tqdm.write(f"\n--- Fase {phase_idx + 1} / Epoch {epoch + 1}/{phase_config['epochs']} (Globale: {global_epoch}) ---")
            
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device, use_amp)
            val_loss, val_miou, val_iou_per_class, val_conf_matrix = evaluate(model, val_loader, criterion, device, config)

            if scheduler and isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_miou)
            elif scheduler:
                scheduler.step() # Per altri scheduler come StepLR
            
            current_lr = optimizer.param_groups[0]['lr']
            tqdm.write(f"Risultati Epoch {global_epoch}: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val mIoU: {val_miou:.4f} | LR: {current_lr:.1e}")

            if val_miou > best_iou:
                best_iou = val_miou
                best_conf_matrix = val_conf_matrix
                torch.save({'model_state_dict': model.state_dict(), 'config': config}, config['paths']['model_save'])
                tqdm.write(f"\n>>> 🎉 Nuovo modello migliore salvato! mIoU: {best_iou:.4f} <<<")

    tqdm.write("\n--- Training Completato ---")
    tqdm.write(f"Miglior mIoU di validazione ottenuto: {best_iou:.4f}")
    if best_conf_matrix is not None:
        plot_confusion_matrix(
            best_conf_matrix, 
            config['data']['class_names'], 
            config['data']['ignore_index'], 
            config['paths']['plot_save']
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script di training per segmentazione semantica configurabile.")
    parser.add_argument('--config', type=str, default='configs/base_config.yaml',
                        help='Percorso al file di configurazione YAML.')
    args = parser.parse_args()
    main(args.config)


