import os
import math
import random
from dataclasses import dataclass
from typing import Tuple, Optional

import h5py
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =========================================================
# Config
# =========================================================
@dataclass
class CFG:
    data_path: str = "/home/yangz2/code/space_universe/data/A5/Galaxy10_DECals.h5"   # 修改成你的 h5 文件路径
    output_dir: str = "./outputs_unet"
    image_size: int = 128
    batch_size: int = 32
    num_workers: int = 4
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    val_ratio: float = 0.2
    use_amp: bool = True
    device: str = "cuda:1" if torch.cuda.is_available() else "cpu"

cfg = CFG()
CLASS_NAMES = [
    "Disturbed Galaxies",
    "Merging Galaxies",
    "Round Smooth Galaxies",
    "In-between Round Smooth Galaxies",
    "Cigar Shaped Smooth Galaxies",
    "Barred Spiral Galaxies",
    "Unbarred Tight Spiral Galaxies",
    "Unbarred Loose Spiral Galaxies",
    "Edge-on Galaxies without Bulge",
    "Edge-on Galaxies with Bulge",
]

# =========================================================
# Utils
# =========================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# =========================================================
# Data
# =========================================================
def load_galaxy10_h5(h5_path: str) -> Tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        # astroNN 常见字段: images / ans
        if "images" in f:
            images = f["images"][:]
        elif "X" in f:
            images = f["X"][:]
        else:
            raise KeyError("Cannot find image array. Expected key 'images' or 'X'.")

        if "ans" in f:
            labels = f["ans"][:]
        elif "y" in f:
            labels = f["y"][:]
        elif "labels" in f:
            labels = f["labels"][:]
        else:
            raise KeyError("Cannot find label array. Expected key 'ans', 'y', or 'labels'.")

    images = np.asarray(images)
    labels = np.asarray(labels).astype(np.int64)

    if images.ndim != 4:
        raise ValueError(f"Expected images with shape [N,H,W,C], got {images.shape}")
    if images.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel images, got shape {images.shape}")

    return images, labels


class Galaxy10Dataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        img = self.images[idx]
        lbl = int(self.labels[idx])
        img = Image.fromarray(img.astype(np.uint8))
        if self.transform is not None:
            img = self.transform(img)
        return img, lbl


class RandomRotate90:
    def __call__(self, img: Image.Image) -> Image.Image:
        k = random.randint(0, 3)
        if k == 0:
            return img
        return img.rotate(90 * k)


train_tfms = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    RandomRotate90(),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

val_tfms = transforms.Compose([
    transforms.Resize((cfg.image_size, cfg.image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# =========================================================
# Model: U-Net style classifier
# =========================================================
class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetClassifier(nn.Module):
    """
    U-Net 风格分类模型：
    - 编码器提取多尺度特征
    - 解码器保留 U-Net 的跳连结构
    - 最终把 bottleneck 和 decoder 的全局池化特征拼接后做分类
    """
    def __init__(self, num_classes: int = 10, base_ch: int = 32, dropout: float = 0.2):
        super().__init__()
        self.inc = DoubleConv(3, base_ch)
        self.down1 = Down(base_ch, base_ch * 2)
        self.down2 = Down(base_ch * 2, base_ch * 4)
        self.down3 = Down(base_ch * 4, base_ch * 8)
        self.down4 = Down(base_ch * 8, base_ch * 16)

        self.up1 = Up(base_ch * 16, base_ch * 8, base_ch * 8)
        self.up2 = Up(base_ch * 8, base_ch * 4, base_ch * 4)
        self.up3 = Up(base_ch * 4, base_ch * 2, base_ch * 2)
        self.up4 = Up(base_ch * 2, base_ch, base_ch)

        feat_dim = base_ch * 16 + base_ch * 8 + base_ch * 4 + base_ch * 2 + base_ch
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def gap(self, x):
        return F.adaptive_avg_pool2d(x, 1).flatten(1)

    def forward(self, x):
        x1 = self.inc(x)      # [B, C,   H,   W]
        x2 = self.down1(x1)   # [B, 2C,  H/2, W/2]
        x3 = self.down2(x2)   # [B, 4C,  H/4, W/4]
        x4 = self.down3(x3)   # [B, 8C,  H/8, W/8]
        x5 = self.down4(x4)   # [B,16C, H/16,W/16]

        d1 = self.up1(x5, x4) # [B, 8C,  H/8, W/8]
        d2 = self.up2(d1, x3) # [B, 4C,  H/4, W/4]
        d3 = self.up3(d2, x2) # [B, 2C,  H/2, W/2]
        d4 = self.up4(d3, x1) # [B,  C,   H,   W]

        feats = torch.cat([
            self.gap(x5),
            self.gap(d1),
            self.gap(d2),
            self.gap(d3),
            self.gap(d4),
        ], dim=1)
        return self.classifier(feats)


# =========================================================
# Train / Eval
# =========================================================
def train_one_epoch(model, loader, criterion, optimizer, scaler, device):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda") and cfg.use_amp)):
            logits = model(imgs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=(device.startswith("cuda") and cfg.use_amp)):
            logits = model(imgs)
            loss = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average="macro")
    report = classification_report(
        all_labels,
        all_preds,
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0,
    )
    return epoch_loss, epoch_acc, epoch_f1, report


# =========================================================
# Main
# =========================================================
def main():
    seed_everything(cfg.seed)
    ensure_dir(cfg.output_dir)

    print(f"Using device: {cfg.device}")
    images, labels = load_galaxy10_h5(cfg.data_path)
    print(f"Loaded images: {images.shape}, labels: {labels.shape}")

    X_train, X_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=cfg.val_ratio,
        random_state=cfg.seed,
        stratify=labels,
    )

    train_ds = Galaxy10Dataset(X_train, y_train, transform=train_tfms)
    val_ds = Galaxy10Dataset(X_val, y_val, transform=val_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = UNetClassifier(num_classes=10, base_ch=32, dropout=0.2).to(cfg.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.device.startswith("cuda") and cfg.use_amp))

    best_f1 = -1.0
    best_path = os.path.join(cfg.output_dir, "best_unet_classifier.pth")

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, cfg.device)
        va_loss, va_acc, va_f1, _ = validate(model, val_loader, criterion, cfg.device)
        scheduler.step()

        print(
            f"Epoch {epoch:02d}/{cfg.epochs} | "
            f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} F1 {tr_f1:.4f} | "
            f"Val Loss {va_loss:.4f} Acc {va_acc:.4f} F1 {va_f1:.4f}"
        )

        if va_f1 > best_f1:
            best_f1 = va_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": cfg.__dict__,
                "best_val_macro_f1": best_f1,
            }, best_path)
            print(f"Saved best model to: {best_path}")

    print("\nLoading best model for final evaluation...")
    ckpt = torch.load(best_path, map_location=cfg.device)
    model.load_state_dict(ckpt["model_state_dict"])

    va_loss, va_acc, va_f1, report = validate(model, val_loader, criterion, cfg.device)
    print(f"\nValidation Accuracy : {va_acc * 100:.2f}%")
    print(f"Validation Macro-F1 : {va_f1:.4f}")
    print("\nClassification Report:")
    print(report)


if __name__ == "__main__":
    main()
