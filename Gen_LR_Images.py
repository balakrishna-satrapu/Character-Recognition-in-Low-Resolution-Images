from PIL import Image
import os

hr_dir = "/content/drive/MyDrive/SRDBPN/images"
lr_dir = "/content/drive/MyDrive/SRDBPN/LR"
scale_factor = 4
os.makedirs(lr_dir, exist_ok=True)

# Resize and save
for filename in os.listdir(hr_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(hr_dir, filename)
        img = Image.open(img_path)

        # Downscale the image
        lr_img = img.resize((img.width // scale_factor, img.height // scale_factor), Image.BICUBIC)

        # Save LR image
        lr_img.save(os.path.join(lr_dir, filename))
