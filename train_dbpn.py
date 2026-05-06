import os
import glob
from PIL import Image
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

#from dbpn import DBPN

def verify_dimensions(lr_img, hr_img, scale_factor):
    """Verify that the dimensions of LR and HR images are correct"""
    lr_h, lr_w = lr_img.shape[-2:]
    hr_h, hr_w = hr_img.shape[-2:]

    expected_hr_h = lr_h * scale_factor
    expected_hr_w = lr_w * scale_factor

    return hr_h == expected_hr_h and hr_w == expected_hr_w


class SRDataset(Dataset):
    """Dataset for super-resolution training"""
    def __init__(self, lr_dir, hr_dir, scale_factor=4, transform=None):
        """
        Args:
            lr_dir (str): Directory with low-resolution images
            hr_dir (str): Directory with high-resolution images
            scale_factor (int): Super-resolution scale factor
            transform (callable, optional): Optional transform to be applied on images
        """
        self.lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.jpg")))
        self.hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.jpg")))
        self.scale_factor = scale_factor

        if len(self.lr_paths) != len(self.hr_paths):
            raise ValueError(f"Number of LR images ({len(self.lr_paths)}) does not match number of HR images ({len(self.hr_paths)})")

        if len(self.lr_paths) == 0:
            raise ValueError(f"No images found in directories: {lr_dir}, {hr_dir}")

        self.transform = transform

        # Verify dimensions of first few images
        print("Verifying image dimensions...")
        for i in range(min(5, len(self.lr_paths))):
            lr_img = Image.open(self.lr_paths[i])
            hr_img = Image.open(self.hr_paths[i])
            lr_w, lr_h = lr_img.size
            hr_w, hr_h = hr_img.size
            if hr_w != lr_w * scale_factor or hr_h != lr_h * scale_factor:
                print(f"Warning: Dimension mismatch in image pair {i}:")
                print(f"LR size: {lr_w}x{lr_h}, HR size: {hr_w}x{hr_h}")
                print(f"Expected HR size: {lr_w * scale_factor}x{lr_h * scale_factor}")

    def __len__(self):
        """Return the total number of image pairs in the dataset"""
        return len(self.lr_paths)

    def ensure_divisible(self, img, scale):
        w, h = img.size
        new_w = w - (w % scale)
        new_h = h - (h % scale)
        if new_w != w or new_h != h:
            print(f"Resizing image from {w}x{h} to {new_w}x{new_h} to ensure divisibility by {scale}")
            img = img.resize((new_w, new_h), Image.BICUBIC)
        return img

    def __getitem__(self, idx):
        # Load images
        lr_img = Image.open(self.lr_paths[idx]).convert('RGB')
        hr_img = Image.open(self.hr_paths[idx]).convert('RGB')

        # Print dimensions before processing
        lr_size = lr_img.size
        hr_size = hr_img.size

        # Ensure dimensions are compatible with scale factor
        lr_img = self.ensure_divisible(lr_img, self.scale_factor)
        hr_size = tuple(s * self.scale_factor for s in lr_img.size)
        hr_img = hr_img.resize(hr_size, Image.BICUBIC)

        if self.transform:
            lr_img = self.transform(lr_img)
            hr_img = self.transform(hr_img)

        # Verify dimensions after transform
        if not verify_dimensions(lr_img, hr_img, self.scale_factor):
            print(f"Warning: Dimension mismatch after transform in image {self.lr_paths[idx]}")
            print(f"Original LR size: {lr_size}, HR size: {hr_size}")
            print(f"After transform - LR size: {lr_img.shape}, HR size: {hr_img.shape}")

        return lr_img, hr_img


def create_transforms():
    """Create transform pipeline for the images"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

# Add denormalization for visualization
def denormalize(tensor):
    """Denormalize the tensor back to image range"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def train_dbpn(model, dataloader, num_epochs=100, lr=1e-5, checkpoint_dir='checkpoints'):
    """Train the DBPN model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))

    # Loss functions
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                    factor=0.5, patience=5,
                                                    verbose=True, min_lr=1e-7)

    best_loss = float('inf')
    model.train()

    for epoch in range(num_epochs):
        total_loss = 0
        with tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}") as pbar:
            for batch_idx, (lr_img, hr_img) in enumerate(pbar):
                lr_img = lr_img.to(device)
                hr_img = hr_img.to(device)

                if not verify_dimensions(lr_img, hr_img, model.scale_factor):
                    print(f"\nWarning: Dimension mismatch in batch {batch_idx}")
                    print(f"LR shape: {lr_img.shape}, HR shape: {hr_img.shape}")
                    continue

                optimizer.zero_grad()
                sr_img = model(lr_img)

                if sr_img.size() != hr_img.size():
                    print(f"\nError: Output size mismatch in batch {batch_idx}")
                    print(f"SR shape: {sr_img.shape}, HR shape: {hr_img.shape}")
                    continue

                try:
                    # Calculate content loss
                    l1_loss = l1_criterion(sr_img, hr_img)
                    mse_loss = mse_criterion(sr_img, hr_img)

                    # Calculate edge loss using image gradients
                    sr_dx = sr_img[:, :, :, 1:] - sr_img[:, :, :, :-1]
                    sr_dy = sr_img[:, :, 1:, :] - sr_img[:, :, :-1, :]
                    hr_dx = hr_img[:, :, :, 1:] - hr_img[:, :, :, :-1]
                    hr_dy = hr_img[:, :, 1:, :] - hr_img[:, :, :-1, :]

                    edge_loss = F.l1_loss(sr_dx, hr_dx) + F.l1_loss(sr_dy, hr_dy)

                    # Combine losses


                    loss = l1_loss * 0.5 + mse_loss * 0.2 + edge_loss * 0.3

                    loss.backward()

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()

                    total_loss += loss.item()
                    pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'l1': f'{l1_loss.item():.6f}',
                        'edge': f'{edge_loss.item():.6f}'
                    })

                except RuntimeError as e:
                    print(f"\nError in batch {batch_idx}: {str(e)}")
                    print(f"SR shape: {sr_img.shape}, HR shape: {hr_img.shape}")
                    if batch_idx == 0:
                        raise e
                    continue

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")

        scheduler.step(avg_loss)

        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"dbpn_epoch_{epoch + 1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_model_path = os.path.join(checkpoint_dir, "dbpn_best.pth")
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved: {best_model_path}")


def main():
    # Configuration
    lr_dir = "/content/drive/MyDrive/SRDBPN/LR"  # Low-resolution images directory
    hr_dir = "/content/drive/MyDrive/SRDBPN/images"  # High-resolution images directory
    batch_size = 8      # Reduced batch size for stability
    num_epochs = 100
    learning_rate = 1e-5  # Using the lower learning rate
    checkpoint_dir = "/content/drive/MyDrive/SRDBPN/checkpoints"
    scale_factor = 4

    # Create transforms
    transform = create_transforms()

    try:
        # Create dataset and dataloader
        dataset = SRDataset(lr_dir, hr_dir, scale_factor=scale_factor, transform=transform)
        dataloader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2,
                              pin_memory=True)

        print(f"Dataset size: {len(dataset)} images")
        print(f"Number of batches per epoch: {len(dataloader)}")

        # Create model
        model = DBPN(num_channels=3, scale_factor=scale_factor)

        # Train model
        train_dbpn(model, dataloader,
                   num_epochs=num_epochs,
                   lr=learning_rate,
                   checkpoint_dir=checkpoint_dir)

    except Exception as e:
        print(f"Error occurred during training: {str(e)}")


if __name__ == "__main__":
    main()