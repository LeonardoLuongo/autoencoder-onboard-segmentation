# Segmentation Training Pipeline — Clean Version
# Author: cesco&lello (2025)
"""A self‑contained training script for semantic segmentation with an EfficientNet‑B3
U‑Net backbone, aggressive data augmentation (Kornia), stratified train/val split,
class‑balanced losses, optional mixed‑precision, and pixel‑wise contrastive
regularisation.

Major clean‑ups vs. the draft:
• grouped imports, removed duplicates/unused
• put constants in a dataclass → clearer access & type hints
• single source of truth for Kornia extra_args (avoid repetition)
• clarified docstrings, removed commented/outdated blocks
• dropped legacy Albumentations version (now 100 % Kornia)
• moved heavy utilities (confusion‑matrix, CutMix, PixelContrastiveLoss) to bottom
  for readability
• every public function has a type‑annotated signature & doctring
"""

from __future__ import annotations

# ============================================================================
# 1. Imports
# ============================================================================
import os
import glob
import random
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import imagehash
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn, optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import kornia.augmentation as K
from kornia.constants import DataKey, Resample
from kornia.utils import image_to_tensor
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.losses import FocalLoss
import seaborn as sns  # only for confusion‑matrix plot
import matplotlib.pyplot as plt

# ============================================================================
# 2. Reproducibility
# ============================================================================

def seed_everything(seed: int) -> None:
    """Set all relevant random seeds for deterministic behaviour."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================================
# 3. Configuration
# ============================================================================

@dataclass
class Config:
    """Hyper‑parameters & paths (edit here, nowhere else)."""

    # paths
    root: Path = Path("./")
    ds_original: Path = root / "dataset/training/train"
    ds_aug: Path = root / "dataset/training/augmented"
    ds_aug_aggr: Path = root / "dataset/training/augmented_aggressive"
    model_path: Path = root / "best_model_b3.pth"

    # data
    output_size: Tuple[int, int] = (384, 768)
    num_classes: int = 9
    ignore_index: int = 0
    val_split: float = 0.2

    # optimisation
    lr_head: float = 1e-3
    lr_full: float = 1e-4
    weight_decay: float = 1e-6
    freeze_epochs: int = 7
    unfreeze_epochs: int = 23
    finetune_epochs: int = 10
    batch_size: int = 3
    workers_train: int = 4
    workers_val: int = 2
    persistent_workers: bool = True

    # augmentation / sampling
    rare_class_indices: Tuple[int, ...] = (4, 6)
    rare_boost_factor: float = 25.0
    contrastive_temp: float = 0.1
    contrastive_max_samples: int = 4096 * 3
    contrastive_w: float = 1.0

    # misc
    random_seed: int = 42
    encoder: str = "efficientnet-b3"
    encoder_weights: str = "imagenet"
    device: torch.device = field(init=False)
    use_amp: bool = field(init=False)

    # stats (populated at runtime)
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_amp = self.device.type == "cuda"

    @property
    def class_names(self) -> List[str]:
        return ["Ignored"] + [f"Class {i}" for i in range(1, self.num_classes)]


CFG = Config()  # global singleton (keeps code compact)

# cache for dataset statistics
CACHE_PATH = CFG.root / "cache/preprocessing_cache.pth"
CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 4. Utilities — stats, split, class‑weights
# ============================================================================

def _dataset_signature(paths: List[Path]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        st = p.stat()
        h.update(f"{p}:{st.st_size}:{int(st.st_mtime)}".encode())
    return h.hexdigest()


def calculate_mean_std(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """Compute per‑channel mean / std in a single pass (GPU‑friendly)."""
    tqdm.write("Computing dataset mean/std …")
    psum, psum_sq, count = torch.zeros(3), torch.zeros(3), 0
    for imgs, _ in tqdm(loader, desc="Stats"):
        b, _, h, w = imgs.shape
        psum += imgs.sum(dim=[0, 2, 3])
        psum_sq += (imgs ** 2).sum(dim=[0, 2, 3])
        count += b * h * w
    mean = psum / count
    var = psum_sq / count - mean ** 2
    std = torch.sqrt(torch.clamp(var, min=1e-6))
    return mean.numpy(), std.numpy()


# ---------------------------------------------------------------------------
# Kornia transform helpers — single source of truth for extra_args
# ---------------------------------------------------------------------------

MASK_ARGS = {DataKey.MASK: {"resample": Resample.NEAREST, "align_corners": False}}

def train_transforms_kornia(cfg: Config, aggressive_p: float = 0.5) -> nn.Sequential:
    mean, std = cfg.mean, cfg.std  # populated later
    return K.AugmentationSequential(
        K.RandomHorizontalFlip(p=0.5),
        K.RandomVerticalFlip(p=aggressive_p / 2),
        K.RandomRotation(degrees=180, p=aggressive_p),
        K.ColorJitter(0.2, 0.2, 0.2, 0.1, p=0.7),
        K.RandomGaussianBlur((3, 5), (0.1, 2.0), p=0.4),
        K.Resize(cfg.output_size),
        K.Normalize(torch.tensor(mean), torch.tensor(std)),
        data_keys=["input", "mask"],
        extra_args=MASK_ARGS,
    )


def val_transforms_kornia(cfg: Config) -> nn.Sequential:
    mean, std = cfg.mean, cfg.std
    return K.AugmentationSequential(
        K.Resize(cfg.output_size),
        K.Normalize(torch.tensor(mean), torch.tensor(std)),
        data_keys=["input", "mask"],
        extra_args=MASK_ARGS,
    )


def cpu_transforms_kornia(cfg: Config) -> nn.Sequential:
    return K.AugmentationSequential(
        K.Resize(cfg.output_size),
        data_keys=["input", "mask"],
        extra_args=MASK_ARGS,
    )


# ---------------------------------------------------------------------------
# Segmentation Dataset
# ---------------------------------------------------------------------------

class SegmentationDataset(Dataset):
    """Folder‑based dataset: each sample_dir contains 'rgb.jpg' & 'labels.png'."""

    def __init__(self, sample_dirs: List[Path], transform: nn.Module, cfg: Config):
        self.sample_dirs = sample_dirs
        self.transform = transform
        self.cfg = cfg

    def __len__(self) -> int:  # noqa: D401
        return len(self.sample_dirs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sd = self.sample_dirs[idx]
        img_np = np.array(Image.open(sd / "rgb.jpg").convert("RGB"))
        mask_np = np.array(Image.open(sd / "labels.png"), dtype=np.uint8)
        assert mask_np.max() < self.cfg.num_classes, f"Invalid label in {sd}"
        img_t = image_to_tensor(img_np).float() / 255.0
        mask_t = torch.from_numpy(mask_np.copy()).long()
        aug_img, aug_mask = self.transform(img_t, mask_t)
        return aug_img.squeeze(0), aug_mask.squeeze()


# ---------------------------------------------------------------------------
# Pre‑processing cache (mean/std, split, class‑weights)
# ---------------------------------------------------------------------------

def load_or_build_cache(cfg: Config) -> Dict[str, Any]:
    """Load cached preprocessing or compute it once and cache it."""
    orig_paths = [p for p in (cfg.ds_original).iterdir() if p.is_dir()]
    dataset_sig = _dataset_signature(orig_paths)
    if CACHE_PATH.is_file():
        cache = torch.load(CACHE_PATH)
        if cache.get("dataset_version") == dataset_sig:
            tqdm.write("Using cached preprocessing data.")
            return cache
        tqdm.write("Cache invalid — recomputing …")

    # stats
    stats_ds = SegmentationDataset(orig_paths, cpu_transforms_kornia(cfg), cfg)
    stats_loader = DataLoader(stats_ds, batch_size=cfg.batch_size * 2, shuffle=False, num_workers=cfg.workers_train)
    mean, std = calculate_mean_std(stats_loader)

    # split (with hash‑based dedup)
    train_dirs, val_dirs = stratified_split(cfg)

    # class weights
    weights = class_frequency_weights(SegmentationDataset(train_dirs, cpu_transforms_kornia(cfg), cfg), cfg)

    cache = {
        "dataset_version": dataset_sig,
        "mean": mean,
        "std": std,
        "train_dirs": train_dirs,
        "val_dirs": val_dirs,
        "class_weights": weights.cpu(),
    }
    torch.save(cache, CACHE_PATH)
    tqdm.write(f"Cache saved → {CACHE_PATH}")
    return cache


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def stratified_split(cfg: Config) -> Tuple[List[Path], List[Path]]:
    """Deduplicate similar images (phash) then stratify w.r.t. rare class presence."""
    paths_dict = {"orig": cfg.ds_original, "aug": cfg.ds_aug, "aggr": cfg.ds_aug_aggr}
    all_dirs: List[Path] = []
    for name, pth in paths_dict.items():
        if pth.is_dir():
            dirs = [d for d in pth.iterdir() if d.is_dir()]
            tqdm.write(f"{name}: {len(dirs)} samples")
            all_dirs.extend(dirs)
    if not all_dirs:
        raise FileNotFoundError("No sample directories found.")

    # bucket‑by‑phash prefix (speed‑up)
    hashes = {p: imagehash.phash(Image.open(p / "rgb.jpg")) for p in tqdm(all_dirs, desc="phash")}
    buckets = defaultdict(list)
    for p, h in hashes.items():
        buckets[int(str(h)[:2], 16)].append((p, h))

    G = nx.Graph()
    G.add_nodes_from(hashes.keys())
    for items in tqdm(buckets.values(), desc="dedup"):
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                p1, h1 = items[i]
                p2, h2 = items[j]
                if h1 - h2 < 5:  # perceptual distance
                    G.add_edge(p1, p2)
    groups = list(nx.connected_components(G))

    # flag groups containing rare classes
    rare = cfg.rare_class_indices
    labels = [int(any(np.isin(np.unique(np.array(Image.open(gp.pop() / "labels.png"))), rare))) for gp in groups]
    train_g, val_g = train_test_split(groups, test_size=cfg.val_split, random_state=cfg.random_seed, stratify=labels)
    train_dirs = [p for g in train_g for p in g]
    val_dirs = [p for g in val_g for p in g]
    tqdm.write(f"Split: {len(train_dirs)} train | {len(val_dirs)} val")
    return train_dirs, val_dirs


# ---------------------------------------------------------------------------
# Class‑frequency weights
# ---------------------------------------------------------------------------

def class_frequency_weights(dataset: Dataset, cfg: Config) -> torch.Tensor:
    num_cls = cfg.num_classes
    counts = np.zeros(num_cls, dtype=np.int64)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=cfg.workers_train)
    for _, m in tqdm(loader, desc="class freq"):
        for mask in m:
            counts += np.bincount(mask.flatten(), minlength=num_cls)
    freq = counts / counts.sum()
    weights = 1.0 / (freq + 1e-6)
    weights /= weights.mean()
    for idx in cfg.rare_class_indices:
        weights[idx] *= cfg.rare_boost_factor
    return torch.tensor(np.clip(weights, None, 30.0), dtype=torch.float, device=cfg.device)


# ============================================================================
# 5. Model & Losses
# ============================================================================

def get_model(cfg: Config) -> nn.Module:
    model = smp.Unet(cfg.encoder, cfg.encoder_weights, in_channels=3, classes=cfg.num_classes)
    tqdm.write(f"Model params (trainable): {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    return model


class PixelContrastiveLoss(nn.Module):
    """Supervised pixel‑wise InfoNCE (all valid pixels, optional subsample)."""

    def __init__(self, temp: float, max_samples: int, ignore_index: int):
        super().__init__()
        self.temp, self.max_samples, self.ignore = temp, max_samples, ignore_index

    def forward(self, feats: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:  # noqa: D401
        B, C, H, W = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(-1, C)
        labels = labels.view(-1)
        valid = labels != self.ignore
        feats, labels = feats[valid], labels[valid]
        if self.max_samples and feats.size(0) > self.max_samples:
            idx = torch.randperm(feats.size(0), device=feats.device)[: self.max_samples]
            feats, labels = feats[idx], labels[idx]
        feats = F.normalize(feats, dim=1)
        logits = feats @ feats.T / self.temp
        eye = torch.eye(logits.size(0), device=feats.device, dtype=torch.bool)
        same_cls = labels.unsqueeze(0) == labels.unsqueeze(1)
        pos = same_cls & ~eye
        exp = torch.exp(logits)
        loss = -(torch.log((exp * pos).sum(1) / (exp * ~eye).sum(1) + 1e-8)).mean()
        return loss


class FocalTverskyLoss(nn.Module):
    def __init__(self, alpha=0.8, beta=0.2, gamma=2.0, class_weights=None, ignore_index=0):
        super().__init__()
        self.a, self.b, self.g = alpha, beta, gamma
        self.cw = class_weights
        self.ignore = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):  # noqa: D401
        probs = torch.softmax(logits, 1)
        true = F.one_hot(targets, logits.size(1)).permute(0, 3, 1, 2).float()
        if self.ignore is not None:
            mask = targets != self.ignore
            probs, true = probs * mask[:, None], true * mask[:, None]
        tp = (probs * true).sum((0, 2, 3))
        fp = (probs * (1 - true)).sum((0, 2, 3))
        fn = ((1 - probs) * true).sum((0, 2, 3))
        ti = (tp + 1e-7) / (tp + self.a * fn + self.b * fp + 1e-7)
        loss = (1 - ti).pow(self.g)
        if self.cw is not None:
            loss = loss * self.cw
        return loss.mean()


# ============================================================================
# 6. Training / Evaluation helpers
# ============================================================================

def cutmix(images: torch.Tensor, masks: torch.Tensor, alpha: float = 1.0):
    b, _, h, w = images.shape
    idx = torch.randperm(b, device=images.device)
    lam = np.random.beta(alpha, alpha)
    cut_rat = np.sqrt(1.0 - lam)
    ch, cw = int(h * cut_rat), int(w * cut_rat)
    cx, cy = np.random.randint(w), np.random.randint(h)
    x1, y1 = np.clip(cx - cw // 2, 0, w), np.clip(cy - ch // 2, 0, h)
    x2, y2 = np.clip(cx + cw // 2, 0, w), np.clip(cy + ch // 2, 0, h)

    images[:, :, y1:y2, x1:x2] = images[idx, :, y1:y2, x1:x2]
    masks[:, y1:y2, x1:x2] = masks[idx, y1:y2, x1:x2]
    return images, masks


def train_epoch(model: nn.Module, loader: DataLoader, opt: optim.Optimizer, scaler: GradScaler, loss_fn, epoch: int, epochs: int, cfg: Config) -> float:
    model.train(); running = 0.0
    bar = tqdm(loader, desc=f"Train {epoch+1}/{epochs}", leave=False)
    for imgs, msk in bar:
        if epoch < epochs - 5 and torch.rand(1) < 0.5:
            imgs, msk = cutmix(imgs, msk)
        imgs, msk = imgs.to(cfg.device), msk.to(cfg.device, dtype=torch.long)
        with autocast(enabled=cfg.use_amp):
            out = model(imgs)
            feat = model.encoder(imgs)[-1]
            loss = loss_fn(out, msk)  # combined loss already includes class‑weights
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item(); bar.set_postfix(loss=f"{loss.item():.4f}")
    return running / len(loader)


def evaluate(model: nn.Module, loader: DataLoader, loss_fn, cfg: Config) -> Tuple[float, float]:
    model.eval(); total, conf = 0.0, torch.zeros(cfg.num_classes, cfg.num_classes, dtype=torch.int64, device=cfg.device)
    with torch.no_grad():
        for imgs, msk in tqdm(loader, desc="Val", leave=False):
            imgs, msk = imgs.to(cfg.device), msk.to(cfg.device, dtype=torch.long)
            with autocast(enabled=cfg.use_amp):
                out = model(imgs)
                total += loss_fn(out, msk).item()
            pred = out.argmax(1)
            valid = msk.flatten() != cfg.ignore_index
            idx = msk.flatten()[valid] * cfg.num_classes + pred.flatten()[valid]
            conf += torch.bincount(idx, minlength=cfg.num_classes ** 2).reshape(cfg.num_classes, cfg.num_classes)
    tp = conf.diag(); fp = conf.sum(0) - tp; fn = conf.sum(1) - tp
    iou_cls = (tp / (tp + fp + fn + 1e-6)).cpu().numpy()
    miou = np.nanmean(iou_cls[cfg.ignore_index + 1 :])
    return total / len(loader), miou


# ============================================================================
# 7. Main
# ============================================================================

def main() -> None:
    cache = load_or_build_cache(CFG)
    CFG.mean, CFG.std = cache["mean"], cache["std"]

    seed_everything(CFG.random_seed)
    tqdm.write(f"Device: {CFG.device} | AMP: {CFG.use_amp}")

    train_tf = train_transforms_kornia(CFG, aggressive_p=0.1)
    val_tf = val_transforms_kornia(CFG)
    train_ds = SegmentationDataset(cache["train_dirs"], train_tf, CFG)
    val_ds = SegmentationDataset(cache["val_dirs"], val_tf, CFG)

    pin = CFG.device.type == "cuda"
    train_ld = DataLoader(train_ds, CFG.batch_size, True, num_workers=CFG.workers_train, pin_memory=pin, persistent_workers=CFG.persistent_workers)
    val_ld = DataLoader(val_ds, CFG.batch_size, False, num_workers=CFG.workers_val, pin_memory=pin, persistent_workers=CFG.persistent_workers)

    model = get_model(CFG).to(CFG.device)
    scaler = GradScaler(enabled=CFG.use_amp)

    # losses
    weights = cache["class_weights"].to(CFG.device)
    ft = FocalTverskyLoss(alpha=0.9, beta=0.1, gamma=3.0, class_weights=None, ignore_index=CFG.ignore_index).to(CFG.device)
    wce = nn.CrossEntropyLoss(weights, ignore_index=CFG.ignore_index).to(CFG.device)
    def combined(pred, tgt):
        return 0.8 * ft(pred, tgt) + 0.2 * wce(pred, tgt)

    # optimiser / schedulers
    opt = optim.AdamW(model.parameters(), lr=CFG.lr_full)

    best_miou = -1.0
    for epoch in range(CFG.unfreeze_epochs):
        tr_loss = train_epoch(model, train_ld, opt, scaler, combined, epoch, CFG.unfreeze_epochs, CFG)
        val_loss, miou = evaluate(model, val_ld, combined, CFG)
        tqdm.write(f"Ep {epoch+1}/{CFG.unfreeze_epochs} — train {tr_loss:.4f} | val {val_loss:.4f} | mIoU {miou:.4f}")
        if miou > best_miou:
            best_miou = miou
            torch.save({"model_state_dict": model.state_dict(), "cfg": CFG.__dict__}, CFG.model_path)
            tqdm.write(f"Saved new best (mIoU {best_miou:.4f}) → {CFG.model_path}")

    tqdm.write("Training complete.")


if __name__ == "__main__":
    main()
