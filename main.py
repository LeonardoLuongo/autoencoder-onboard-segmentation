import torch
import numpy as np
import cv2
import glob
import json
import ast
from pathlib import Path
import matplotlib.pyplot as plt
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


COLOR_TO_LABEL = json.load(open("./color_map.json"))["color_to_id"]
COLOR_TO_LABEL = {ast.literal_eval(k): v for k, v in COLOR_TO_LABEL.items()}
LABEL_TO_COLOR = json.load(open("./color_map.json"))["id_to_color"]
LABEL_TO_COLOR = {int(k): tuple(v) for k, v in LABEL_TO_COLOR.items()}
NUM_CLASSES = len(LABEL_TO_COLOR) - 1 
print(f"Number of classes (excluding 'void'): {NUM_CLASSES}")
IGNORE_INDEX = 8  # Index for 'void' class

CURRENT_BASE_PATH = Path(__file__).resolve().parent
TRAIN_DATASET_PATH = CURRENT_BASE_PATH / "dataset" / "training" / "train"
TEST_DATASET_PATH = CURRENT_BASE_PATH / "dataset" / "testing" / "test"

print(f"Current base path: {CURRENT_BASE_PATH}")
print(f"Training dataset path: {TRAIN_DATASET_PATH}")
print(f"Testing dataset path: {TEST_DATASET_PATH}")



class Sample:
    def __init__(self, img_path: str, seg_im_path: str) -> None:
        self.img_path = img_path
        self.label_path = seg_im_path

    def load_sample(self) -> tuple[np.ndarray, np.ndarray]:
        img = cv2.imread(self.img_path)
        label = cv2.imread(self.label_path)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)

        return img, label


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples
    
    def __len__(self) -> int:
        return len(self.samples)
         
    def __getitem__(self, idx) -> list[torch.Tensor, torch.Tensor]:
        H_target, W_target = 544, 1024

        img, target = self.samples[idx].load_sample()

        if img.shape[0] != H_target or img.shape[1] != W_target:
            # Immagine -> Interpolazione LINEARE (morbida)
            img = cv2.resize(img, (W_target, H_target), interpolation=cv2.INTER_LINEAR)
            # Maschera -> Interpolazione NEAREST (senza sfumature, mantiene i colori esatti)
            target = cv2.resize(target, (W_target, H_target), interpolation=cv2.INTER_NEAREST)
        
        # From image net normalization
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        img = (img - mean) / std

        # Convert to tensor
        img = np.transpose(img, (2, 0, 1))  # Cambia da HWC a CHW
        img = torch.from_numpy(img).float()

        # create mask
        mask = np.zeros((target.shape[0], target.shape[1]), dtype=np.int64) 
        for color, label in COLOR_TO_LABEL.items():
            equality = np.all(target == np.array(color), axis=-1)
            mask[equality] = label
        
        mask = torch.from_numpy(mask).long()
        return img, mask

def create_create_samples(imgs_wildcard: str | Path, seg_imgs_wildcard: str | Path) -> list[Sample]:
    imgs_path = glob.glob(str(imgs_wildcard))
    seg_imgs_path = glob.glob(str(seg_imgs_wildcard))
    samples = []
    for img_path, seg_im_path in zip(imgs_path, seg_imgs_path):
        samples.append(Sample(img_path, seg_im_path))
    return samples


def train_one_epoch(
        model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim, criterion: torch.nn, 
        device: torch.device
    ):
    model.train()
    for batch in dataloader:
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)['out']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()


def evaluate_mIoU(predictions: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    _, preds = torch.max(predictions, 1)
    preds = preds.cpu().numpy()
    labels = labels.cpu().numpy()

    intersection = np.zeros(num_classes)
    union = np.zeros(num_classes)

    for cls in range(num_classes):
        pred_inds = (preds == cls)
        label_inds = (labels == cls)

        intersection[cls] = np.sum(pred_inds & label_inds)
        union[cls] = np.sum(pred_inds | label_inds)

    iou = intersection / (union + 1e-10)
    mIoU = np.mean(iou)
    return mIoU

def predict_with_entropy(model: torch.nn.Module, image_tensor: torch.Tensor, entropy_threshold: float = 0.5, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        # Output del modello (Logits) -> [1, 5, H, W]
        logits = model(image_tensor)['out']
        
        # Applica Softmax per ottenere probabilità (da 0 a 1)
        probs = torch.nn.functional.softmax(logits, dim=1)
        
        # Calcolo dell'entropia per ogni pixel
        # H(x) = - sum(p(x) * log(p(x)))
        entropy = -torch.sum(probs * torch.log(probs + 1e-6), dim=1)
        
        _, preds = torch.max(probs, dim=1)
        
        # Se l'entropia è alta, la rete è confusa
        uncertain_mask = entropy > entropy_threshold
        
        preds[uncertain_mask] = ignore_index
        return preds

def from_logits_to_color_image(logits: torch.Tensor) -> np.ndarray:
    _, preds = torch.max(logits, 1)  # Shape: [B, H, W]
    preds = preds.cpu().numpy()
    batch_size, height, width = preds.shape
    color_images = np.zeros((batch_size, height, width, 3), dtype=np.uint8)

    for cls_id, color in LABEL_TO_COLOR.items():
        color_images[preds == cls_id] = color

    return color_images

def evaluate_model_one_epoch(
        model: torch.nn.Module, dataloader: torch.utils.data.DataLoader,
        criterion: torch.nn, device: torch.device, num_classes: int
    ) -> tuple[float, float]: # Restituisce una tupla (loss, mIoU)
    
    model.eval()
    total_loss = 0.0
    total_mIoU = 0.0
    num_batches = len(dataloader)

    best_mIoU = 0.0
    worst_mIoU = 1.0
    best_mIoU_pred = None
    best_mIoU_label = None
    worst_mIoU_pred = None
    worst_mIoU_label = None


    with torch.no_grad():
        for batch in dataloader:
            imgs, labels = batch
            imgs = imgs.to(device)
            labels = labels.to(device)

            print(f"Evaluating batch with images of shape: {imgs.shape}")
            logits = model(imgs)['out'] # Shape: [B, Num_Classes, H, W]

            loss = criterion(logits, labels)
            total_loss += loss.item()

            mIoU = evaluate_mIoU(logits, labels, num_classes)
            total_mIoU += mIoU

            if mIoU > best_mIoU:
                best_mIoU = mIoU
                best_mIoU_pred = logits
                best_mIoU_label = labels

            if mIoU < worst_mIoU:
                worst_mIoU = mIoU
                worst_mIoU_pred = logits
                worst_mIoU_label = labels

    avg_loss = total_loss / num_batches
    avg_mIoU = total_mIoU / num_batches
    
    return avg_loss, avg_mIoU, best_mIoU_pred, best_mIoU_label, worst_mIoU_pred, worst_mIoU_label



def show_evaluation_results(
        best_pred: torch.Tensor, best_label: torch.Tensor,
        worst_pred: torch.Tensor, worst_label: torch.Tensor
    ):
    best_color_image = from_logits_to_color_image(best_pred)
    worst_color_image = from_logits_to_color_image(worst_pred)

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    plt.title("Best Prediction")
    plt.imshow(best_color_image[0])
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title("Best Ground Truth")
    best_label_np = best_label[0].cpu().numpy()
    best_label_color = np.zeros((best_label_np.shape[0], best_label_np.shape[1], 3), dtype=np.uint8)
    for cls_id, color in LABEL_TO_COLOR.items():
        best_label_color[best_label_np == cls_id] = color
    plt.imshow(best_label_color)
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.title("Worst Prediction")
    plt.imshow(worst_color_image[0])
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title("Worst Ground Truth")
    worst_label_np = worst_label[0].cpu().numpy()
    worst_label_color = np.zeros((worst_label_np.shape[0], worst_label_np.shape[1], 3), dtype=np.uint8)
    for cls_id, color in LABEL_TO_COLOR.items():
        worst_label_color[worst_label_np == cls_id] = color
    plt.imshow(worst_label_color)
    plt.axis('off')

    plt.show()


def main():
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU.")

    # Load train samples
    train_samples = create_create_samples(TRAIN_DATASET_PATH / "*" / "*.jpg", TRAIN_DATASET_PATH / "*" / "*.png")
    train_dataset = MyDataset(train_samples)

    # Load test samples
    test_samples = create_create_samples(TEST_DATASET_PATH / "*" / "*.jpg", TEST_DATASET_PATH / "*" / "*.png")
    test_dataset = MyDataset(test_samples)

    # Create DataLoaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Load Pre-trained segmentation model
    model = torchvision.models.segmentation.deeplabv3_resnet50(pretrained=True)
    model.classifier = DeepLabHead(in_channels=2048, num_classes=NUM_CLASSES)
    model.aux_classifier = None
    model = model.to(device)

    # freeze backbone weigths
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Training loop
    num_epochs = 1
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mIoU, best_pred, best_label, worst_pred, worst_label = evaluate_model_one_epoch(model, test_loader, criterion, device, NUM_CLASSES)
        print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {val_mIoU:.4f}")

        show_evaluation_results(best_pred, best_label, worst_pred, worst_label)



if __name__ == "__main__":
    main()