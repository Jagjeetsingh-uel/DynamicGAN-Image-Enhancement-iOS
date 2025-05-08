import os
import glob
import random
from PIL import Image, ImageFilter, ImageOps
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.transforms.functional import crop
from tqdm import tqdm

import torch.nn.functional as F
from torchvision.models import vgg19

from concurrent.futures import ThreadPoolExecutor

import torch.multiprocessing as mp  # Import multiprocessing
from concurrent.futures import ThreadPoolExecutor  # Import ThreadPoolExecutor

# --- Set device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True
executor = ThreadPoolExecutor(max_workers=6)
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------
# Residual Block and Generator
# ---------------------------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, num_residual_blocks=16):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(inplace=True)
        )
        self.residual_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_residual_blocks)])
        self.middle_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.ReLU(inplace=True)
        )
        self.output_conv = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.initial(x)
        res = self.residual_blocks(x)
        res = self.middle_conv(res)
        x = self.upsample(x + res)
        return self.output_conv(x)

# ---------------------------------------------
# Discriminator
# ---------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)


# ---------------------------------------------
# Losses
# ---------------------------------------------
class AdversarialLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCELoss()

    def forward(self, pred, target):
        return self.criterion(pred, target)


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        return self.criterion(pred, target)


class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(pretrained=True).features[:36]
        self.feature_extractor = nn.Sequential(*list(vgg)).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        return self.criterion(self.feature_extractor(pred), self.feature_extractor(target))

# ---------------------------------------------
# Custom Dataset for Dynamic LR-HR Pairs
# ---------------------------------------------
class SuperResolutionDataset(Dataset):
    """
    Dataset that dynamically creates low-res images from high-res images using random degradation.
    Supports .png and .jpg files.
    """
    def __init__(self, root_dir, patch_size=256):
        self.patch_size = patch_size - (patch_size % 4)  # Ensure divisible by 4
        self.hr_images = glob.glob(os.path.join(root_dir, "*.png")) + glob.glob(os.path.join(root_dir, "*.jpg"))

        self.hr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.lr_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])

    def degrade(self, hr_patch):
        # Gaussian blur
        if random.random() > 0.5:
            hr_patch = hr_patch.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 2.5)))

        # Downscale and upscale (Bilinear)
        w, h = hr_patch.size
        scale = random.choice([2, 4])
        lr_patch = hr_patch.resize((w // scale, h // scale), Image.BILINEAR)
        lr_patch = lr_patch.resize((w, h), Image.BILINEAR)

        # Pixelation
        if random.random() > 0.5:
            block = random.randint(4, 8)
            lr_patch = lr_patch.resize((w // block, h // block), Image.NEAREST)
            lr_patch = lr_patch.resize((w, h), Image.NEAREST)

        # Blackout patch
        if random.random() > 0.7:
            box_x = random.randint(0, w - 50)
            box_y = random.randint(0, h - 50)
            blackout = Image.new("RGB", (50, 50), (0, 0, 0))
            lr_patch.paste(blackout, (box_x, box_y))

        return lr_patch

    def __getitem__(self, idx):
        hr_image = Image.open(self.hr_images[idx]).convert("RGB")
        w, h = hr_image.size

        upscale_factor = 4
        hr_crop_size = self.patch_size * upscale_factor  # e.g., 256 × 4 = 1024

        # Ensure HR crop is large enough
        if w < hr_crop_size or h < hr_crop_size:
            hr_image = hr_image.resize((hr_crop_size, hr_crop_size), Image.BICUBIC)
        else:
            x = random.randint(0, w - hr_crop_size)
            y = random.randint(0, h - hr_crop_size)
            hr_image = crop(hr_image, top=y, left=x, height=hr_crop_size, width=hr_crop_size)

        # Generate LR by downsampling
        lr_image = hr_image.resize((self.patch_size, self.patch_size), Image.BICUBIC)

        # Add degradations (optional)
        # lr_image = self.degrade(lr_image)
        lr_image = executor.submit(self.degrade, lr_image).result()


        return self.lr_transform(lr_image), self.hr_transform(hr_image)

    def __len__(self):
        return len(self.hr_images)

# ----------------------------------------------
# Data Pipeline
# ----------------------------------------------

class ImageDataset(Dataset):
    """Dataset to load low-res and high-res image pairs"""
    def __init__(self, root_dir, transform=None, upscale_factor=4):  # Added upscale_factor
        self.root_dir = root_dir
        self.transform = transform
        self.upscale_factor = upscale_factor  # Store the upscale factor
        self.image_pairs = [(os.path.join(root_dir, 'low_res', img),
                             os.path.join(root_dir, 'high_res', img))
                            for img in os.listdir(os.path.join(root_dir, 'low_res'))]
        if not self.image_pairs:
            raise ValueError("No image pairs found in the specified directories.")

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        low_res_path, high_res_path = self.image_pairs[idx]
        low_res_image = Image.open(low_res_path).convert('RGB')
        high_res_image = Image.open(high_res_path).convert('RGB')

        if self.transform:
            ow, oh = high_res_image.size
            # Resize both LR and HR images to 256x256
            high_res_image = high_res_image.resize((256, 256), Image.BICUBIC)
            low_res_image = low_res_image.resize((256, 256), Image.BICUBIC)


            low_res_image = self.transform(low_res_image)
            high_res_image = self.transform(high_res_image)
        return low_res_image, high_res_image



def get_dataloader(root_dir, batch_size=16, shuffle=True, num_workers=4): # Added num_workers
    """Utility function to create DataLoader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers) # Pass num_workers
    return dataloader


# ---------------------------------------------
# Training Function
# ---------------------------------------------
def train(generator, discriminator, dataloader, epochs=50, lr=1e-4):
    generator.to(DEVICE)
    discriminator.to(DEVICE)

    opt_g = torch.optim.Adam(generator.parameters(), lr=lr)
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr)

    adv_loss = AdversarialLoss().to(DEVICE)
    content_loss = ContentLoss().to(DEVICE)
    perceptual_loss = PerceptualLoss().to(DEVICE)

    for epoch in range(epochs):
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]")
        for lr_img, hr_img in loop:
            lr_img = lr_img.to(DEVICE)
            hr_img = hr_img.to(DEVICE)

            # --- Train Discriminator ---
            fake_hr = generator(lr_img).detach()
            real_preds = discriminator(hr_img)
            fake_preds = discriminator(fake_hr)
            d_loss = 0.5 * (adv_loss(real_preds, torch.ones_like(real_preds)) +
                            adv_loss(fake_preds, torch.zeros_like(fake_preds)))
            opt_d.zero_grad()
            d_loss.backward()
            opt_d.step()

            # --- Train Generator ---
            fake_hr = generator(lr_img)
            fake_preds = discriminator(fake_hr)
            g_adv = adv_loss(fake_preds, torch.ones_like(fake_preds))
            g_content = content_loss(fake_hr, hr_img)
            g_percep = perceptual_loss(fake_hr, hr_img)
            g_loss = g_adv + g_content + g_percep

            opt_g.zero_grad()
            g_loss.backward()
            opt_g.step()

            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

        torch.save(generator.state_dict(), f'generator_epoch_v4_{epoch+1}.pth')



# ----------------------------------------------
# Model Saving & Inference
# ----------------------------------------------

def save_onnx(generator, onnx_file_path):
    """Export Generator model to ONNX format"""
    dummy_input = torch.randn(1, 3, 256, 256).to(DEVICE) # changed dummy input size
    generator.eval().to(DEVICE)
    torch.onnx.export(generator, dummy_input, onnx_file_path, export_params=True, opset_version=11,
                      input_names=['input'], output_names=['output']) # Added input and output names
    print(f"ONNX model saved to {onnx_file_path}")

def load_model(model, model_path, device):
    """
    Loads the trained model.

    Args:
        model (nn.Module): The model to load the weights into.
        model_path (str): The path to the saved model.
        device (torch.device): The device to load the model on.
    Returns:
        nn.Module: The loaded model.
    """
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode for inference
    return model


def enhance_image(generator, image_path, output_path):
    """Enhance a low-quality image using the trained Generator"""
    generator.eval().to(DEVICE)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    image = Image.open(image_path).convert('RGB')
    image = image.resize((256,256), Image.BICUBIC) # changed input image size
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        enhanced_tensor = generator(input_tensor)
    enhanced_image = transforms.ToPILImage()(enhanced_tensor.squeeze().cpu())
    enhanced_image.save(output_path)
    print(f"Enhanced image saved to {output_path}")



# ----------------------------------------------
# Main Function
# ----------------------------------------------

if __name__ == "__main__":
    # Create dummy directories and files if they don't exist
    HR_IMAGE_DIR = "./data/high_res_images"


    # dataset = SuperResolutionDataset(HR_IMAGE_DIR, patch_size=256)
    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=6, pin_memory=True)

    # generator = Generator()
    # discriminator = Discriminator()

    # Train the model
    # train(generator, discriminator, dataloader, epochs=2, lr=1e-4)

    # Save the generator model
    # torch.save(generator.state_dict(), 'generator_model.pth')
    # print("Generator model saved to generator_model.pth")

    # Export the generator to ONNX
    # save_onnx(generator, 'generator_model.onnx')

    # Load Generator
    my_generator = Generator()
    my_generator = load_model(my_generator, 'generator_epoch_v4_2.pth', DEVICE)

    # Example of enhancing an image
    enhance_image(my_generator, './data/before.png', 'enhanced.png')
    # enhance_image(loaded_generator,'./data/before.png','enhanced.png')
