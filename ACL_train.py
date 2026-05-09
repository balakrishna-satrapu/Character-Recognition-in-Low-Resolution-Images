import os
import sys
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
sys.path.append("/content/drive/MyDrive/yolov5")
from models.common import DetectMultiBackend
from utils.loss import ComputeLoss

# === Helper Functions ===
def verify_dimensions(lr_img, hr_img, scale_factor):
    lr_h, lr_w = lr_img.shape[-2:]
    hr_h, hr_w = hr_img.shape[-2:]
    return hr_h == lr_h * scale_factor and hr_w == lr_w * scale_factor

def create_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean

# === Custom Collate Function for YOLO-style target batching ===
def custom_collate_fn(batch):
    lr_imgs, targets = zip(*batch)
    lr_imgs = torch.stack(lr_imgs, 0)

    new_targets = []
    for i, t in enumerate(targets):
        if t.numel() > 0:
            t = t.clone()
            t[:, 0] = i  # update batch index for YOLO format
            new_targets.append(t)

    if len(new_targets) > 0:
        targets = torch.cat(new_targets, 0)
    else:
        targets = torch.empty((0, 6), device=lr_imgs.device)

    return lr_imgs, targets

# === Dataset ===
class SRDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, scale_factor=4, transform=None):
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, '*.jpg')))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, '*.jpg')))
        self.scale_factor = scale_factor
        self.transform = transform

        if len(self.lr_paths) != len(self.hr_paths):
            raise ValueError("Mismatch in LR and HR image count")

        for i in range(min(5, len(self.lr_paths))):
            lr = Image.open(self.lr_paths[i])
            hr = Image.open(self.hr_paths[i])
            if hr.size != (lr.width * scale_factor, lr.height * scale_factor):
                print(f"Warning: Image {i} size mismatch: LR={lr.size}, HR={hr.size}")

    def ensure_divisible(self, img, scale):
        w, h = img.size
        return img.resize(((w // scale) * scale, (h // scale) * scale), Image.BICUBIC)

    def __getitem__(self, idx):
        lr_img = Image.open(self.lr_paths[idx]).convert('RGB')
        hr_img = Image.open(self.hr_paths[idx]).convert('RGB')
        lr_img = self.ensure_divisible(lr_img, self.scale_factor)
        hr_img = hr_img.resize((lr_img.width * self.scale_factor, lr_img.height * self.scale_factor), Image.BICUBIC)

        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        # Load targets for LR image from labels folder, expecting YOLO txt format:
        txt_path = self.lr_paths[idx].replace('.jpg', '.txt').replace('LR', 'labels')
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                labels = [list(map(float, line.strip().split())) for line in f.readlines()]
            targets = torch.tensor(labels, dtype=torch.float32)
        else:
            targets = torch.zeros((0, 6), dtype=torch.float32)

        # If targets exist, ensure class index is valid and reset batch idx (set later in collate)
        if targets.numel() > 0:
            # Sometimes targets in YOLO have class in col 0, leave as is, batch idx assigned later
            pass

        return lr_img, hr_img, targets

    def __len__(self):
        return len(self.lr_paths)

# Dataset wrapper for Phase 2 (YOLO training) - returns lr_img and targets only
class Phase2Dataset(Dataset):
    def __init__(self, base_dataset):
        self.base_dataset = base_dataset

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        lr_img, _, targets = self.base_dataset[idx]
        return lr_img, targets

# === Phase 1: Train DBPN with YOLOv5 frozen ===
def train_dbpn_acl_phase1(dbpn_model, dataloader, yolo_model, compute_loss, num_epochs=100, lr=1e-5, checkpoint_dir='/content/drive/MyDrive/checkpoints_acl'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dbpn_model = dbpn_model.to(device).train()
    yolo_model.to(device).eval()
    for p in yolo_model.parameters():
        p.requires_grad = False

    os.makedirs(checkpoint_dir, exist_ok=True)
    optimizer = optim.Adam(dbpn_model.parameters(), lr=lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-7)

    l1_loss_fn = nn.L1Loss()
    mse_loss_fn = nn.MSELoss()
    best_loss = float('inf')

    for epoch in range(num_epochs):
        dbpn_model.train()
        total_loss = 0
        for lr_img, hr_img, targets in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            lr_img, hr_img = lr_img.to(device), hr_img.to(device)
            # targets already tensor on CPU; move to device
            targets = targets.to(device)

            sr_img = dbpn_model(lr_img)

            # SR losses
            l1_loss = l1_loss_fn(sr_img, hr_img)
            mse_loss = mse_loss_fn(sr_img, hr_img)
            edge_loss = F.l1_loss(sr_img[:, :, :, 1:] - sr_img[:, :, :, :-1], hr_img[:, :, :, 1:] - hr_img[:, :, :, :-1]) + \
                        F.l1_loss(sr_img[:, :, 1:, :] - sr_img[:, :, :-1, :], hr_img[:, :, 1:, :] - hr_img[:, :, :-1, :])
            loss_sr = 0.5 * l1_loss + 0.2 * mse_loss + 0.3 * edge_loss

            # Resize SR image to YOLO input size (must match training)
            sr_resized = F.interpolate(sr_img, size=(160, 160), mode='bilinear', align_corners=False)

            # Compute character recognition loss with frozen YOLO model
            # Make sure no gradients for yolo_model, but allow gradients for sr_img
            yolo_model.eval()  # enforce eval mode

            # Forward pass through yolo_model.model (YOLOv5's Detect head + backbone)
            # yolo_model.model returns list of predictions per YOLO output layer
            yolo_pred = yolo_model.model(sr_resized)

            if targets.numel() > 0:
                loss_parts, _ = compute_loss(yolo_pred, targets)
                loss_char = sum(loss_parts)
            else:
                loss_char = torch.tensor(0.0, device=device)

            # Total loss: super-resolution + weighted char recognition loss
            loss_total = loss_sr + 0.01 * loss_char

            optimizer.zero_grad()
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(dbpn_model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss_total.item()

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.6f}")

        # Save best model checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(dbpn_model.state_dict(), os.path.join(checkpoint_dir, "dbpn_acl_best.pth"))
            print("✅ Saved best DBPN model.")

# === Phase 2: Train YOLOv5 with DBPN frozen ===
def train_yolo_acl_phase2(dbpn_model, yolo_model, compute_loss, dataloader, num_epochs=50, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dbpn_model = dbpn_model.to(device).eval()
    yolo_model.model.to(device).train()

    # Ensure YOLO params require grad
    for p in yolo_model.model.parameters():
        p.requires_grad = True

    optimizer = optim.Adam(yolo_model.model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    for epoch in range(num_epochs):
        total_loss = 0
        for lr_img, targets in tqdm(dataloader, desc=f"Phase 2 Epoch {epoch+1}/{num_epochs}"):
            lr_img = lr_img.to(device)
            targets = targets.to(device).float()

            with torch.no_grad():
                sr_img = dbpn_model(lr_img)
            sr_resized = F.interpolate(sr_img, size=(160, 160), mode='bilinear', align_corners=False)

            preds = yolo_model.model(sr_resized)
            loss_parts, _ = compute_loss(preds, targets)

            # Debug prints (optional)
            # print(f"loss_parts requires_grad: {[lp.requires_grad for lp in loss_parts]}")

            loss = sum(loss_parts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Phase 2 Epoch {epoch+1} - Avg YOLO Loss: {avg_loss:.9f}")

    torch.save(yolo_model.model.state_dict(), "/content/drive/MyDrive/yolo_acl_best.pth")
    print("✅ Saved best YOLO model.")


# === Usage ===
if __name__ == '__main__':
    #from dbpn import DBPN  # Assuming your DBPN model is in dbpn.py

    map_location = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dbpn = DBPN(num_channels=3, scale_factor=4)
    checkpoint = torch.load("/content/drive/MyDrive/SRDBPN/checkpoints/dbpn_epoch_100.pth", map_location=map_location)
    dbpn.load_state_dict(checkpoint['model_state_dict'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    yolo_model = DetectMultiBackend("/content/drive/MyDrive/yolov5/runs/train/yolo_char_rec14/weights/best.pt", device=device)
    compute_loss = ComputeLoss(yolo_model.model)

    transform = create_transforms()
    dataset = SRDataset(
        lr_dir="/content/drive/MyDrive/SRDBPN/LR",
        hr_dir="/content/drive/MyDrive/yolov5/yolo_char_rec/data/images/train",
        scale_factor=4,
        transform=transform
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)

    # Phase 1: Train DBPN with frozen YOLO
    train_dbpn_acl_phase1(
        dbpn_model=dbpn,
        dataloader=dataloader,
        yolo_model=yolo_model,
        compute_loss=compute_loss,
        num_epochs=10,
        lr=1e-5,
        checkpoint_dir="/content/drive/MyDrive/SRDBPN/checkpoints_acl"
    )

    # Phase 2: Train YOLO with frozen DBPN
    phase2_dataset = Phase2Dataset(dataset)
    phase2_dataloader = DataLoader(
        phase2_dataset, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, collate_fn=custom_collate_fn
    )

    train_yolo_acl_phase2(
        dbpn_model=dbpn,
        yolo_model=yolo_model,
        compute_loss=compute_loss,
        dataloader=phase2_dataloader,
        num_epochs=10,
        lr=1e-4
    )
