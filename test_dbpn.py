import os
import glob
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim

#from dbpn import DBPN


def load_model(model_path, scale_factor=4):
    """Load a trained DBPN model

    Args:
        model_path (str): Path to the model checkpoint
        scale_factor (int): Super-resolution scale factor

    Returns:
        model (nn.Module): Loaded model
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DBPN(scale_factor=scale_factor)

    # Handle different checkpoint formats
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def create_transforms():
    """Create transform pipeline for testing"""
    return {
        'to_tensor': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])
        ]),
        'to_pil': transforms.ToPILImage()
    }


def denormalize(tensor):
    """Denormalize the tensor back to image range"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean


def test_dbpn(model, lr_dir, save_dir, hr_dir=None, batch_size=1):
    """Test DBPN model on images

    Args:
        model (nn.Module): Trained DBPN model
        lr_dir (str): Directory containing low-resolution test images
        save_dir (str): Directory to save super-resolved images
        hr_dir (str, optional): Directory containing high-resolution ground truth images
        batch_size (int): Batch size for testing
    """
    os.makedirs(save_dir, exist_ok=True)
    device = next(model.parameters()).device

    # Get image paths
    lr_paths = sorted(glob.glob(os.path.join(lr_dir, "*.jpg")))
    if not lr_paths:
        raise ValueError(f"No images found in directory: {lr_dir}")

    if hr_dir:
        hr_paths = sorted(glob.glob(os.path.join(hr_dir, "*.jpg")))
        if len(hr_paths) != len(lr_paths):
            raise ValueError("Number of LR and HR images don't match")

    # Create transforms
    transforms_dict = create_transforms()

    # Initialize metrics
    metrics = {'psnr': [], 'ssim': []}

    with torch.no_grad():
        for i in tqdm(range(0, len(lr_paths), batch_size), desc="Processing images"):
            # Process batch
            batch_lr_paths = lr_paths[i:i + batch_size]
            batch_tensors = []

            for lr_path in batch_lr_paths:
                # Load and preprocess image
                lr_img = Image.open(lr_path).convert("RGB")
                lr_tensor = transforms_dict['to_tensor'](lr_img)

                # Ensure dimensions are divisible by scale factor
                _, h, w = lr_tensor.size()
                new_h = h - (h % model.scale_factor)
                new_w = w - (w % model.scale_factor)
                if new_h != h or new_w != w:
                    lr_tensor = transforms.functional.center_crop(lr_tensor, (new_h, new_w))

                batch_tensors.append(lr_tensor.unsqueeze(0))

            # Concatenate batch and move to device
            lr_batch = torch.cat(batch_tensors, dim=0).to(device)

            # Generate SR images
            sr_batch = model(lr_batch)

            # Denormalize before saving
            sr_batch = denormalize(sr_batch)
            sr_batch = torch.clamp(sr_batch, 0, 1)

            # Process each image in batch
            for j, sr_tensor in enumerate(sr_batch):
                # Convert to PIL
                sr_img = transforms_dict['to_pil'](sr_tensor.cpu())

                # Save SR image
                filename = os.path.basename(batch_lr_paths[j])
                save_path = os.path.join(save_dir, filename)
                sr_img.save(save_path, quality=95)

                # Calculate metrics if HR available
                if hr_dir:
                    hr_img = Image.open(hr_paths[i + j]).convert("RGB")
                    hr_img = hr_img.resize(sr_img.size, Image.BICUBIC)

                    # Convert to numpy arrays
                    hr_array = np.array(hr_img).astype(np.float32) / 255.
                    sr_array = np.array(sr_img).astype(np.float32) / 255.

                    # Calculate metrics
                    psnr = compare_psnr(hr_array, sr_array, data_range=1.0)
                    ssim = compare_ssim(hr_array, sr_array,
                                      channel_axis=-1,
                                      data_range=1.0)

                    metrics['psnr'].append(psnr)
                    metrics['ssim'].append(ssim)

    # Print results
    if hr_dir and metrics['psnr']:
        avg_psnr = np.mean(metrics['psnr'])
        avg_ssim = np.mean(metrics['ssim'])
        print(f"\nResults:")
        print(f"Average PSNR: {avg_psnr:.2f} dB")
        print(f"Average SSIM: {avg_ssim:.4f}")
        print(f"Number of images: {len(metrics['psnr'])}")

    print(f"\nSaved super-resolved images to: {save_dir}")


def main():
    # Configuration
    model_path = "/content/drive/MyDrive/SRDBPN/checkpoints/dbpn_epoch_100.pth"  # Path to model checkpoint
    lr_dir = "/content/drive/MyDrive/SRDBPN/test/LR"                   # Test LR images directory
    hr_dir = "/content/drive/MyDrive/SRDBPN/test/HR"                   # Test HR images directory (optional)
    save_dir = "/content/drive/MyDrive/SRDBPN/test/SR"                   # Directory to save results
    batch_size = 4                            # Batch size for testing
    scale_factor = 4                          # Super-resolution scale factor

    try:
        # Create directories
        os.makedirs(save_dir, exist_ok=True)

        # Load model
        model = load_model(model_path, scale_factor)
        print(f"Model loaded from: {model_path}")
        print(f"Testing on device: {next(model.parameters()).device}")

        # Test model
        test_dbpn(
            model=model,
            lr_dir=lr_dir,
            hr_dir=hr_dir,
            save_dir=save_dir,
            batch_size=batch_size
        )

    except Exception as e:
        print(f"Error occurred during testing: {str(e)}")


if __name__ == "__main__":
    main()