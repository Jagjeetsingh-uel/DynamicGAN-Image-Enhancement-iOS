import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Check for device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------
# Model Architecture
# ----------------------------------------------

class ResidualBlock(nn.Module):
    """Residual Block for the Generator"""
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """Generator using Residual Blocks"""
    def __init__(self, in_channels=3, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.initial = nn.Conv2d(in_channels, 64, kernel_size=9, stride=1, padding=4)
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)]
        )
        self.middle_conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.output_conv = nn.Conv2d(64, in_channels, kernel_size=9, stride=1, padding=4)
        self.relu = nn.ReLU(inplace=True)  # Add ReLU here

    def forward(self, x):
        initial = self.relu(self.initial(x))  # Apply ReLU after initial conv
        residual = self.residual_blocks(initial)
        middle = self.middle_conv(residual)
        output = self.output_conv(initial + middle)
        return output # Removed final activation


class Discriminator(nn.Module):
    """Discriminator to classify real vs. fake high-res images"""
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 1),
            nn.Sigmoid()  # Add Sigmoid here
        )

    def forward(self, x):
        return self.model(x)


# ----------------------------------------------
# Loss Functions
# ----------------------------------------------

class AdversarialLoss(nn.Module):
    """Adversarial Loss using Binary Cross-Entropy"""
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, pred, target):
        return self.criterion(pred, target)


class ContentLoss(nn.Module):
    """Content Loss (L1 Loss)"""
    def __init__(self):
        super(ContentLoss, self).__init__()
        self.criterion = nn.L1Loss()

    def forward(self, pred, target):
        return self.criterion(pred, target)


class PerceptualLoss(nn.Module):
    """Perceptual Loss using pretrained VGG19"""
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(vgg.features[:36])).eval()
        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.criterion = nn.MSELoss()

    def forward(self, pred, target):
        pred_features = self.feature_extractor(pred)
        target_features = self.feature_extractor(target)
        return self.criterion(pred_features, target_features)


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



def get_dataloader(root_dir, batch_size=16, shuffle=True):
    """Utility function to create DataLoader"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    dataset = ImageDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


# ----------------------------------------------
# Training Script
# ----------------------------------------------

def train(generator, discriminator, dataloader, epochs=100, lr=1e-4, checkpoint_dir='./checkpoints'):
    """Training loop for GAN"""
    generator.to(DEVICE)
    discriminator.to(DEVICE)

    # Define optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    # Define loss functions
    adversarial_loss = AdversarialLoss().to(DEVICE)
    content_loss = ContentLoss().to(DEVICE)
    perceptual_loss = PerceptualLoss().to(DEVICE)

    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(epochs):
        generator.train()
        discriminator.train()
        loop = tqdm(dataloader, leave=True)

        for low_res, high_res in loop:
            low_res = low_res.to(DEVICE)
            high_res = high_res.to(DEVICE)

            # Train Discriminator
            optimizer_d.zero_grad()
            fake_high_res = generator(low_res)
            real_preds = discriminator(high_res)
            fake_preds = discriminator(fake_high_res.detach())
            real_loss = adversarial_loss(real_preds, torch.ones_like(real_preds))
            fake_loss = adversarial_loss(fake_preds, torch.zeros_like(fake_preds))
            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            optimizer_d.step()

            # Train Generator
            optimizer_g.zero_grad()
            fake_high_res = generator(low_res) #removed redundant generator call
            fake_preds = discriminator(fake_high_res)
            g_adv_loss = adversarial_loss(fake_preds, torch.ones_like(fake_preds))
            g_content_loss = content_loss(fake_high_res, high_res)
            g_perceptual_loss = perceptual_loss(fake_high_res, high_res)
            g_loss = g_adv_loss + g_content_loss + g_perceptual_loss
            g_loss.backward()
            optimizer_g.step()

            # Log losses
            loop.set_description(f"Epoch [{epoch}/{epochs}]")
            loop.set_postfix(d_loss=d_loss.item(), g_loss=g_loss.item())

        # Save model checkpoints
        torch.save(generator.state_dict(), os.path.join(checkpoint_dir, f'generator_epoch_{epoch}.pth'))
        torch.save(discriminator.state_dict(), os.path.join(checkpoint_dir, f'discriminator_epoch_{epoch}.pth'))



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
    root_dir = './data/Training Images'
    hr_dir = os.path.join(root_dir, 'high_res')
    lr_dir = os.path.join(root_dir, 'low_res')

    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
        for i in range(3):
            img = Image.new('RGB', (1080, 1024), color=(255, 0, 0))
            img.save(os.path.join(hr_dir, f'hr_{i+1}.jpg'))
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
        for i in range(3):
            img = Image.new('RGB', (1080, 1024), color=(0, 0, 255)) # changed dummy image size
            img.save(os.path.join(lr_dir, f'lr_{i+1}.jpg'))

    # Instantiate generator and discriminator
    generator = Generator().to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    # Get data loader
    dataloader = get_dataloader(root_dir, batch_size=2)  # Adjust batch_size as needed

    # Train the model
    train(generator, discriminator, dataloader, epochs=10, lr=1e-4)  # Adjust epochs and lr as needed

    # Save the generator model
    torch.save(generator.state_dict(), 'generator_model.pth')
    print("Generator model saved to generator_model.pth")

    # Export the generator to ONNX
    save_onnx(generator, 'generator_model.onnx')

    # Load Generator
   # loaded_generator = Generator()
   # loaded_generator = load_model(loaded_generator, 'generator_model.pth', DEVICE)

    # Example of enhancing an image
    enhance_image(generator, os.path.join(lr_dir, '033_d891acbd.jpg'), 'enhanced_image.jpg')
