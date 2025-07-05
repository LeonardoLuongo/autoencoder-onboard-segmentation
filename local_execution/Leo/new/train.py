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


'''
PS D:\MLPoject-Git\autoencoder-onboard-segmentation> d:\MLPoject-Git\autoencoder-onboard-segmentation\local_execution\Leo\new\run_train.bat

D:\MLPoject-Git\autoencoder-onboard-segmentation>cd ./local_execution/Leo/new/ 

D:\MLPoject-Git\autoencoder-onboard-segmentation\local_execution\Leo\new>python train.py --config configs/config.yaml 
Caricamento configurazione da: configs/config.yaml
Global seed impostato a 42
Device in uso: cuda | Mixed Precision (AMP): Abilitato
Caricamento percorsi dei dataset...
Trovati 931 campioni in '../../../dataset/training/train'.
Trovati 561 campioni in '../../../dataset/training/augmented'.
Trovati 1960 campioni in '../../../dataset/training/augmented_aggressive'.
Calcolo Hash Immagini: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3452/3452 [00:36<00:00, 93.83it/s] 
Confronto immagini simili per raggruppamento...
Analisi Bucket: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 86/86 [00:00<00:00, 165.89it/s] 
Trovati 2524 gruppi di immagini uniche.
Etichettatura gruppi per lo split: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 2524/2524 [00:41<00:00, 61.29it/s] 
Split stratificato creato: 2794 train, 658 validation.
C:\Users\leona\AppData\Roaming\Python\Python312\site-packages\albumentations\core\validation.py:114: UserWarning: ShiftScaleRotate is a special case of Affine transform. Please use Affine transform instead.
  original_init(self, **validated_kwargs)
D:\MLPoject-Git\autoencoder-onboard-segmentation\local_execution\Leo\new\src\data_loader.py:51: UserWarning: Argument(s) 'max_holes, max_height, max_width, min_holes, fill_value, mask_fill_value' are not valid for transform CoarseDropout
  transforms_list.append(aug_class(**aug_item['params']))
Creazione del modello: Unet con encoder 'efficientnet-b5'
Parametri addestrabili: 31,217,161

--- INIZIO Fase 1: Freeze Encoder & Train Decoder (Fase 1/2) ---
Encoder congelato. Addestramento solo del decoder.

--- Fase 1 / Epoch 1/7 (Globale: 1) ---
Risultati Epoch 1: Train Loss: 0.4953 | Val Loss: 0.3786 | Val mIoU: 0.1519 | LR: 1.0e-03                                                                                         

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1519 <<<

--- Fase 1 / Epoch 2/7 (Globale: 2) ---
Risultati Epoch 2: Train Loss: 0.4533 | Val Loss: 0.3568 | Val mIoU: 0.1637 | LR: 1.0e-03                                                                                         

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1637 <<<

--- Fase 1 / Epoch 3/7 (Globale: 3) ---
Risultati Epoch 3: Train Loss: 0.4353 | Val Loss: 0.3462 | Val mIoU: 0.1701 | LR: 1.0e-03                                                                                         

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1701 <<<

--- Fase 1 / Epoch 4/7 (Globale: 4) ---
Risultati Epoch 4: Train Loss: 0.4338 | Val Loss: 0.3368 | Val mIoU: 0.1767 | LR: 1.0e-03                                                                                         

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1767 <<<

--- Fase 1 / Epoch 5/7 (Globale: 5) ---
Risultati Epoch 5: Train Loss: 0.4242 | Val Loss: 0.3318 | Val mIoU: 0.1780 | LR: 1.0e-03                                                                                         

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1780 <<<

--- Fase 1 / Epoch 6/7 (Globale: 6) ---
Risultati Epoch 6: Train Loss: 0.4222 | Val Loss: 0.3304 | Val mIoU: 0.1791 | LR: 1.0e-03                                                                                         

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1791 <<<

--- Fase 1 / Epoch 7/7 (Globale: 7) ---
Risultati Epoch 7: Train Loss: 0.4143 | Val Loss: 0.3217 | Val mIoU: 0.1895 | LR: 1.0e-03                                                                                         

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1895 <<<

--- INIZIO Fase 2: Unfreeze All & Fine-tuning (Fase 2/2) ---
Tutti i parametri sono addestrabili (fine-tuning).

--- Fase 2 / Epoch 1/23 (Globale: 8) ---
Risultati Epoch 8: Train Loss: 0.3987 | Val Loss: 0.3053 | Val mIoU: 0.1987 | LR: 5.0e-05                                                                                         

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.1987 <<<

--- Fase 2 / Epoch 2/23 (Globale: 9) ---
Risultati Epoch 9: Train Loss: 0.3775 | Val Loss: 0.2924 | Val mIoU: 0.2014 | LR: 5.0e-05                                                                                         

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2014 <<<

--- Fase 2 / Epoch 3/23 (Globale: 10) ---
Risultati Epoch 10: Train Loss: 0.3676 | Val Loss: 0.2866 | Val mIoU: 0.2051 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2051 <<<

--- Fase 2 / Epoch 4/23 (Globale: 11) ---
Risultati Epoch 11: Train Loss: 0.3593 | Val Loss: 0.2740 | Val mIoU: 0.2138 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2138 <<<

--- Fase 2 / Epoch 5/23 (Globale: 12) ---
Risultati Epoch 12: Train Loss: 0.3436 | Val Loss: 0.2729 | Val mIoU: 0.2176 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2176 <<<

--- Fase 2 / Epoch 6/23 (Globale: 13) ---
Risultati Epoch 13: Train Loss: 0.3376 | Val Loss: 0.2786 | Val mIoU: 0.2152 | LR: 5.0e-05                                                                                        

--- Fase 2 / Epoch 7/23 (Globale: 14) ---
Risultati Epoch 14: Train Loss: 0.3288 | Val Loss: 0.2620 | Val mIoU: 0.2254 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2254 <<<

--- Fase 2 / Epoch 8/23 (Globale: 15) ---
Risultati Epoch 15: Train Loss: 0.3262 | Val Loss: 0.2581 | Val mIoU: 0.2320 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2320 <<<

--- Fase 2 / Epoch 9/23 (Globale: 16) ---
Risultati Epoch 16: Train Loss: 0.3143 | Val Loss: 0.2501 | Val mIoU: 0.2406 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2406 <<<

--- Fase 2 / Epoch 10/23 (Globale: 17) ---
Risultati Epoch 17: Train Loss: 0.3050 | Val Loss: 0.2444 | Val mIoU: 0.2467 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2467 <<<

--- Fase 2 / Epoch 11/23 (Globale: 18) ---
Risultati Epoch 18: Train Loss: 0.2945 | Val Loss: 0.2468 | Val mIoU: 0.2485 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2485 <<<

--- Fase 2 / Epoch 12/23 (Globale: 19) ---
Risultati Epoch 19: Train Loss: 0.2929 | Val Loss: 0.2408 | Val mIoU: 0.2570 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2570 <<<

--- Fase 2 / Epoch 13/23 (Globale: 20) ---
Risultati Epoch 20: Train Loss: 0.2889 | Val Loss: 0.2360 | Val mIoU: 0.2595 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2595 <<<

--- Fase 2 / Epoch 14/23 (Globale: 21) ---
Risultati Epoch 21: Train Loss: 0.2800 | Val Loss: 0.2344 | Val mIoU: 0.2600 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2600 <<<

--- Fase 2 / Epoch 15/23 (Globale: 22) ---
Risultati Epoch 22: Train Loss: 0.2782 | Val Loss: 0.2300 | Val mIoU: 0.2652 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2652 <<<

--- Fase 2 / Epoch 16/23 (Globale: 23) ---
Risultati Epoch 23: Train Loss: 0.2713 | Val Loss: 0.2287 | Val mIoU: 0.2602 | LR: 5.0e-05                                                                                        

--- Fase 2 / Epoch 17/23 (Globale: 24) ---
Risultati Epoch 24: Train Loss: 0.2651 | Val Loss: 0.2309 | Val mIoU: 0.2649 | LR: 5.0e-05                                                                                        

--- Fase 2 / Epoch 18/23 (Globale: 25) ---
Risultati Epoch 25: Train Loss: 0.2581 | Val Loss: 0.2257 | Val mIoU: 0.2677 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2677 <<<

--- Fase 2 / Epoch 19/23 (Globale: 26) ---
Risultati Epoch 26: Train Loss: 0.2577 | Val Loss: 0.2306 | Val mIoU: 0.2735 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2735 <<<

--- Fase 2 / Epoch 20/23 (Globale: 27) ---
Risultati Epoch 27: Train Loss: 0.2518 | Val Loss: 0.2250 | Val mIoU: 0.2726 | LR: 5.0e-05                                                                                        

--- Fase 2 / Epoch 21/23 (Globale: 28) ---
Risultati Epoch 28: Train Loss: 0.2582 | Val Loss: 0.2217 | Val mIoU: 0.2759 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2759 <<<

--- Fase 2 / Epoch 22/23 (Globale: 29) ---
Risultati Epoch 29: Train Loss: 0.2458 | Val Loss: 0.2181 | Val mIoU: 0.2786 | LR: 5.0e-05                                                                                        

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2786 <<<

--- Fase 2 / Epoch 23/23 (Globale: 30) ---
--- Fase 2 / Epoch 23/23 (Globale: 30) ---
Risultati Epoch 30: Train Loss: 0.2430 | Val Loss: 0.2141 | Val mIoU: 0.2845 | LR: 5.0e-05

>>> 🎉 Nuovo modello migliore salvato! mIoU: 0.2845 <<<

--- Training Completato ---
Miglior mIoU di validazione ottenuto: 0.2845
Visualizzazione e salvataggio Matrice di Confusione...
Matrice di confusione salvata in '../../../confusion_matrix_final.png'

'''