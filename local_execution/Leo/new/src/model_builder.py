# src/model_builder.py
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
from segmentation_models_pytorch import losses as smp_losses
from tqdm import tqdm

def get_model(config: Dict[str, Any]) -> nn.Module:
    """Costruisce il modello di segmentazione basato sulla configurazione."""
    model_cfg = config['model']
    
    # Usa getattr per selezionare dinamicamente l'architettura
    model_class = getattr(smp, model_cfg['architecture'])
    
    model = model_class(
        encoder_name=model_cfg['encoder'],
        encoder_weights=model_cfg['encoder_weights'],
        in_channels=3,
        classes=config['data']['num_classes']
    )
    tqdm.write(f"Creazione del modello: {model_cfg['architecture']} con encoder '{model_cfg['encoder']}'")
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tqdm.write(f"Parametri addestrabili: {params:,}")
    return model

def get_optimizer(model: nn.Module, phase_config: Dict[str, Any]) -> optim.Optimizer:
    """Crea l'optimizer per una specifica fase di training."""
    opt_cfg = phase_config['optimizer']
    opt_class = getattr(optim, opt_cfg['name'])
    
    # Seleziona i parametri da addestrare in base alla fase
    params_to_train = model.parameters()
    if phase_config.get('freeze_encoder', False):
        for param in model.encoder.parameters():
            param.requires_grad = False
        params_to_train = filter(lambda p: p.requires_grad, model.parameters())
        tqdm.write("Encoder congelato. Addestramento solo del decoder.")
    else:
        for param in model.parameters():
            param.requires_grad = True
        tqdm.write("Tutti i parametri sono addestrabili (fine-tuning).")
        
    return opt_class(params_to_train, **opt_cfg['params'])

def get_scheduler(optimizer: optim.Optimizer, phase_config: Dict[str, Any]):
    """Crea lo scheduler per l'optimizer."""
    sched_cfg = phase_config.get('scheduler')
    if not sched_cfg:
        return None
    
    sched_class = getattr(torch.optim.lr_scheduler, sched_cfg['name'])
    return sched_class(optimizer, **sched_cfg['params'])

def get_criterion(config: Dict[str, Any]) -> nn.Module:
    """Crea la funzione di loss."""
    loss_cfg = config['loss']
    # Le loss di SMP sono in un sottomodulo
    loss_class = getattr(smp_losses, loss_cfg['name'])
    return loss_class(**loss_cfg['params'])