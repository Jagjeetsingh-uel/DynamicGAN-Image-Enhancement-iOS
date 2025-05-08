import torch
import torch.nn as nn
import torch.optim as optim
from torch import onnx
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageFilter
import os
import random
import numpy as np
import math
# Optional: For Core ML conversion
import coremltools as ct
import onnx
import onnxruntime
from torch.cuda.amp import autocast, GradScaler


# =========================================
# 1. Model Architecture
# =========================================

class ResidualBlock(nn.Module):
    """
    A basic residual block with two convolutional layers and a skip connection.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class SelfAttention(nn.Module):
    """
    Self-attention module to capture long-range dependencies in the feature maps.
    """
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))  # Learnable scaling factor

    def forward(self, x):
        batch_size, _, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width).permute(0, 2, 1)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)
        attention = torch.matmul(query, key)
        attention = torch.nn.functional.softmax(attention, dim=-1)
        out = torch.matmul(value, attention.permute(0, 2, 1))
        out = out.view(batch_size, -1, height, width)
        out = self.gamma * out + x
        return out

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class Generator(nn.Module):
    """
    Generator network for image enhancement.  Uses residual blocks and self-attention.
    """
    def __init__(self, in_channels=3, out_channels=3, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.residual_blocks = nn.Sequential(*[
            ResidualBlock(64, 64) for _ in range(num_residual_blocks)
        ])
        self.attention = SelfAttention(64) # Add Self-Attention
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor=2) # Upscale
        self.conv3 = nn.Conv2d(128 // 4, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.residual_blocks(out)
        out = self.attention(out) # Apply Self-Attention
        out = self.conv2(out)
        out = self.pixel_shuffle(out)
        out = self.conv3(out)
        return out

class Discriminator(nn.Module):
    """
    Discriminator network to classify real vs. enhanced images.
    """
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(512)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(512, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv4(out)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        out = self.sigmoid(out)
        return out.view(-1, 1)

class PerceptualLoss(nn.Module):
    """
    Perceptual loss based on a pre-trained VGG19 network.
    """
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        # vgg19 = models.vgg19(pretrained=True).features
        vgg19 = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        self.feature_extractor = nn.Sequential(*vgg19[:36]).eval()  # Use features up to relu4_4
        for param in self.feature_extractor.parameters():
            param.requires_grad = False  # Freeze VGG weights
        self.mse_loss = nn.MSELoss()

    def forward(self, enhanced_images, high_res_images):
        enhanced_features = self.feature_extractor(enhanced_images)
        high_res_features = self.feature_extractor(high_res_images)
        return self.mse_loss(enhanced_features, high_res_features)

# =========================================
# 2. Data Pipeline
# =========================================

class ImageDataset(Dataset):
    """
    Dataset class for loading low-res and high-res image pairs.
    """
    def __init__(self, low_res_dir, high_res_dir, transform=None):
        self.low_res_dir = low_res_dir
        self.high_res_dir = high_res_dir
        self.transform = transform
        self.low_res_images = sorted(os.listdir(low_res_dir))
        self.high_res_images = sorted(os.listdir(high_res_dir))
        assert len(self.low_res_images) == len(self.high_res_images), "Number of low-res and high-res images must match."

    def __len__(self):
        return len(self.low_res_images)

    def __getitem__(self, idx):
        low_res_path = os.path.join(self.low_res_dir, self.low_res_images[idx])
        high_res_path = os.path.join(self.high_res_dir, self.high_res_images[idx])
        low_res_image = Image.open(low_res_path).convert("RGB")
        high_res_image = Image.open(high_res_path).convert("RGB")
        if self.transform:
            low_res_image_transforms = transforms.Compose([
                transforms.Resize(128),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            high_res_image_transforms = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            low_res_image = low_res_image_transforms(low_res_image)
            high_res_image = high_res_image_transforms(high_res_image)

        return low_res_image, high_res_image

def create_blurry_image(image, blur_type='gaussian', kernel_size=3, sigma=1.0):
    """
    Simulates a blurry image from a high-resolution image.

    Args:
        image (PIL.Image.Image): The input high-resolution image.
        blur_type (str): The type of blur to apply ('gaussian', 'motion').
        kernel_size (int): The size of the blur kernel.
        sigma (float): Standard deviation for Gaussian blur.

    Returns:
        PIL.Image.Image: The blurry image.
    """
    img = image.copy()  # Create a copy to avoid modifying the original image

    if blur_type == 'gaussian':
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
    elif blur_type == 'motion':
        # Create a motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[kernel_size // 2, :] = 1 / kernel_size  # Horizontal motion blur
        img = img.filter(ImageFilter.Kernel(size=(kernel_size, kernel_size), kernel=kernel.flatten()))
    else:
        raise ValueError(f"Unknown blur type: {blur_type}")
    return img

class BlurryImageDataset(Dataset):
    """
    Dataset class that generates blurry images from high-res images on the fly.
    This is useful for training without a separate set of low-res images.
    """

    def __init__(self, high_res_dir, blur_type='gaussian', kernel_size=3, sigma=1.0):
        self.high_res_dir = high_res_dir
        self.high_res_images = sorted(os.listdir(high_res_dir))
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.sigma = sigma

        self.low_res_transform = transforms.Compose([
            transforms.Resize(128),
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.high_res_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.high_res_images)

    def __getitem__(self, idx):
        high_res_path = os.path.join(self.high_res_dir, self.high_res_images[idx])
        high_res_image = Image.open(high_res_path).convert("RGB")
        low_res_image = create_blurry_image(high_res_image, self.blur_type, self.kernel_size, self.sigma)

        low_res_image = self.low_res_transform(low_res_image)
        high_res_image = self.high_res_transform(high_res_image)

        return low_res_image, high_res_image
def get_data_loader(low_res_dir, high_res_dir, batch_size, shuffle=True, num_workers=4, use_blurry_dataset=False, blur_type='gaussian', kernel_size=3, sigma=1.0):
    """
    Creates a PyTorch DataLoader for loading image pairs.

    Args:
        low_res_dir (str): Path to the directory containing low-resolution images.
        high_res_dir (str): Path to the directory containing high-resolution images.
        batch_size (int): The batch size.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker threads for data loading.
        use_blurry_dataset (bool): If True, uses BlurryImageDataset to generate blurry images on the fly.
        blur_type (str): Type of blur ('gaussian', 'motion') if use_blurry_dataset is True.
        kernel_size (int): Blur kernel size if use_blurry_dataset is True.
        sigma(float): Sigma for gaussian blur

    Returns:
        torch.utils.data.DataLoader: The DataLoader.
    """

    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(128),       # Example augmentation
        transforms.RandomHorizontalFlip(), # Example augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained VGG
    ])
    """
    if use_blurry_dataset:
        dataset = BlurryImageDataset(high_res_dir, transform, blur_type, kernel_size, sigma)
    else:
        dataset = ImageDataset(low_res_dir, high_res_dir, transform)
    """
    if use_blurry_dataset:
        dataset = BlurryImageDataset(high_res_dir, blur_type, kernel_size, sigma)
    else:
        dataset = ImageDataset(low_res_dir, high_res_dir)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

# =========================================
# 3. Training Script
# =========================================

def train_gan(generator, discriminator, train_loader, optimizer_g, optimizer_d, perceptual_loss_fn, adversarial_loss_fn, num_epochs, device, checkpoint_dir='checkpoints'):

    '''Trains the GAN model.'''
    os.makedirs(checkpoint_dir, exist_ok=True)
    generator.to(device)
    discriminator.to(device)
    perceptual_loss_fn.to(device)
    for epoch in range(num_epochs):
        for i, (low_res_images, high_res_images) in enumerate(train_loader):
            low_res_images = low_res_images.to(device)
            high_res_images = high_res_images.to(device)
            # ---------------------
            # Train Discriminator
            # ---------------------
            discriminator.zero_grad()
            # Real images
            real_labels = torch.ones(high_res_images.size(0), 1).to(device)
            real_outputs = discriminator(high_res_images)
            discriminator_real_loss = adversarial_loss_fn(real_outputs, real_labels)
            # Fake images
            # scaler = GradScaler()
            with autocast():
                generated_images = generator(low_res_images)


            fake_labels = torch.zeros(high_res_images.size(0), 1).to(device)
            fake_outputs = discriminator(generated_images.detach())  # Detach to avoid backprop through generator
            discriminator_fake_loss = adversarial_loss_fn(fake_outputs, fake_labels)
            # Total discriminator loss
            discriminator_loss = discriminator_real_loss + discriminator_fake_loss
            discriminator_loss.backward()
            optimizer_d.step()

            # ---------------------
            # Train Generator
            # ---------------------
            generator.zero_grad()
            # Adversarial loss for generator
            generated_outputs = discriminator(generated_images)
            generator_adversarial_loss = adversarial_loss_fn(generated_outputs, real_labels)  # Try to fool discriminator
            # Content loss (L1 or L2) -  L1 Loss gives less blurry results
            generator_content_loss = nn.SmoothL1Loss()(generated_images, high_res_images)
            # Perceptual loss
            generator_perceptual_loss = perceptual_loss_fn(generated_images, high_res_images)
            # Total generator loss
            generator_loss = generator_adversarial_loss + 1e-3 * generator_content_loss + 1e-3 * generator_perceptual_loss # scaling the content and perceptual loss
            generator_loss.backward()
            optimizer_g.step()

            # ---------------------
            # Logging
            # ---------------------
            if (i + 1) % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(train_loader)}], "
                      f"D_loss: {discriminator_loss.item():.4f}, G_loss: {generator_loss.item():.4f}, "
                      f"G_adv: {generator_adversarial_loss.item():.4f}, G_content: {generator_content_loss.item():.4f}, G_percep: {generator_perceptual_loss.item():.4f}")

        # Save model checkpoints
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'optimizer_g_state_dict': optimizer_g.state_dict(),
                'optimizer_d_state_dict': optimizer_d.state_dict(),
            }, os.path.join(checkpoint_dir, f'checkpoint_{epoch+1}.pth'))
            print(f"Saved checkpoint at epoch {epoch+1}")

def calculate_psnr(img1, img2):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming images are normalized to [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """
    Calculates the Structural Similarity Index (SSIM) between two images.
    """
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    mu1 = torch.mean(img1)
    mu2 = torch.mean(img2)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = torch.mean((img1 - mu1) ** 2)
    sigma2_sq = torch.mean((img2 - mu2) ** 2)
    sigma12 = torch.mean((img1 - mu1) * (img2 - mu2))
    ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim

def validate_gan(generator, val_loader, device, perceptual_loss_fn, epoch):
    print("Start Validating")
    """
    Validates the GAN model and calculates PSNR and SSIM.

    Args:
        generator (nn.Module): The generator network.
        val_loader (torch.utils.data.DataLoader): The validation data loader.
        device (torch.device): The device to validate on (CPU or GPU).
        perceptual_loss_fn (nn.Module): perceptual loss
        epoch (int): the epoch number
    """
    generator.to(device)
    generator.eval()  # Set to evaluation mode
    total_psnr = 0.0
    total_ssim = 0.0
    total_perceptual_loss = 0.0
    num_batches = 0

    with torch.no_grad():  # Disable gradient calculation during validation
        for low_res_images, high_res_images in val_loader:
            low_res_images = low_res_images.to(device)
            high_res_images = high_res_images.to(device)
            generated_images = generator(low_res_images)

            total_perceptual_loss += perceptual_loss_fn(generated_images, high_res_images).item()

            for i in range(generated_images.size(0)):  # Iterate over images in the batch
                psnr = calculate_psnr(generated_images[i], high_res_images[i])
                ssim = calculate_ssim(generated_images[i], high_res_images[i])
                total_psnr += psnr
                total_ssim += ssim
            num_batches += 1

    avg_psnr = total_psnr / num_batches
    avg_ssim = total_ssim / num_batches
    avg_perceptual_loss = total_perceptual_loss/num_batches
    print(f"Validation - Epoch [{epoch}], Avg. PSNR: {avg_psnr:.4f}, Avg. SSIM: {avg_ssim:.4f}, Avg. Perceptual Loss: {avg_perceptual_loss:.4f}")
    generator.train()  # Set back to training mode

# =========================================
# 4. Model Saving & Inference
# =========================================

def save_model(model, model_path):
    """
    Saves the trained model.

    Args:
        model (nn.Module): The model to save.
        model_path (str): The path to save the model.
    """
    torch.save(model.state_dict(), model_path)
    print(f"Saved model to {model_path}")

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


def enhance_image(model, low_res_image, device):
    """
    Enhances a single low-resolution image using the trained model.

    Args:
        model (nn.Module): The trained generator model.
        low_res_image (PIL.Image.Image): The low-resolution image to enhance.
        device (torch.device): The device to perform inference on (CPU or GPU).

    Returns:
        PIL.Image.Image: The enhanced high-resolution image.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    low_res_tensor = transform(low_res_image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        enhanced_image_tensor = model(low_res_tensor)
    # Convert the tensor back to a PIL image
    enhanced_image_tensor = enhanced_image_tensor.squeeze(0).cpu()  # Remove batch dimension and move to CPU
    enhanced_image_tensor = enhanced_image_tensor.clamp(0, 1)  # Ensure pixel values are in the valid range [0, 1]
    enhanced_image = transforms.ToPILImage()(enhanced_image_tensor)
    return enhanced_image

# =========================================
# 5. Exporting to Core ML (Optional)
# =========================================

def convert_to_coreml(model_path, input_shape, output_shape, model_name='image_enhancement'):
    """
    Converts a PyTorch model to Core ML format.

    Args:
        model_path (str): Path to the saved PyTorch model (.pth file).
        input_shape (tuple): Shape of the input tensor (e.g., (1, 3, 256, 256)).
        output_shape (tuple): Shape of the output tensor (e.g., (1, 3, 1024, 1024)).
        model_name (str): Name of the Core ML model.

    Returns:
        None
    "" "
    # 1. Load the PyTorch model
    device = torch.device('cpu')  # Conversion should be done on CPU
    generator = Generator() # Or your specific generator
    generator = load_model(generator, model_path, device)
    generator.eval()

    # 2. Create a dummy input tensor
    dummy_input = torch.randn(input_shape)

    # 3. Export to ONNX format
    onnx_file_path = model_name + '.onnx'
    torch.onnx.export(generator,
                      dummy_input,
                      onnx_file_path,
                      export_params=True,
                      opset_version=11,  # Choose an appropriate opset version
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])

    # 4. Load the ONNX model
    onnx_model = onnx.load(onnx_file_path)

    # 5. Convert to Core ML format
    coreml_model = ct.convert(onnx_model,
                              inputs=[ct.TensorType(name='input', shape=input_shape)],
                              outputs=[ct.TensorType(name='output', shape=output_shape)],
                              )

    # 6. Save the Core ML model
    coreml_model_path = model_name + '.mlmodel'
    coreml_model.save(coreml_model_path)
    print(f"Converted model to Core ML and saved at {coreml_model_path}")
    """

    model = Generator()
    model.load_state_dict(torch.load("enhanced_image_generator.pth"))
    model.eval()

    example_input = torch.rand(1, 3, 128, 128)
    traced_model = torch.jit.trace(model, example_input)

    mlmodel = ct.convert(
        traced_model,
        inputs=[ct.TensorType(name="input", shape=example_input.shape)],
        convert_to="mlprogram",  # recommended in v6+
    )
    mlmodel.save("generator.mlpackage")

# =========================================
# Main Function (Example Usage)
# =========================================

def main():
    """
    Main function to train and evaluate the model.
    """

    input_shape = (1, 3, 128, 128)  # Example input shape
    output_shape = (1, 3, 1024, 1024)  # Example output shape
    convert_to_coreml('enhanced_image_generator.pth', input_shape, output_shape)
    exit(0)

    # Set random seed for reproducibility
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Data loading
    low_res_dir = '/Users/jagjeetsinghlabana/Documents/Projects/Python/GAN/data/low_res1000'
    high_res_dir = '/Users/jagjeetsinghlabana/Documents/Projects/Python/GAN/data/high_res1000'
    print("Data Path ",low_res_dir, high_res_dir)
    # If you want to generate blurry images from high-res on the fly, use this:
    use_blurry_dataset = True # Set this to True
    blur_type = 'gaussian'  # or 'motion'
    kernel_size = 5       # Adjust as needed
    sigma = 1.5

    train_loader = get_data_loader(low_res_dir, high_res_dir, batch_size=4, shuffle=True, num_workers=4, use_blurry_dataset=use_blurry_dataset, blur_type=blur_type, kernel_size=kernel_size, sigma=sigma)
    print("1")
    val_loader = get_data_loader(low_res_dir, high_res_dir, batch_size=2, shuffle=False, num_workers=4, use_blurry_dataset=use_blurry_dataset, blur_type=blur_type, kernel_size=kernel_size, sigma=sigma) #separate loader for validation
    print("2")
    # Model instantiation
    generator = Generator()
    print("3")
    discriminator = Discriminator()
    print("4")
    perceptual_loss_fn = PerceptualLoss()
    print("5")
    adversarial_loss_fn = nn.BCEWithLogitsLoss()  # Use BCEWithLogitsLoss for numerical stability
    print("6")
    # Optimizers
    optimizer_g = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0.9, 0.999))
    print("7")
    # Training
    num_epochs = 2
    checkpoint_dir = 'checkpoints'
    train_gan(generator, discriminator, train_loader, optimizer_g, optimizer_d, perceptual_loss_fn, adversarial_loss_fn, num_epochs, device, checkpoint_dir)
    print("8")
    # Validation
    validate_gan(generator, val_loader, device, perceptual_loss_fn, num_epochs)
    print("9")
    # Save the trained generator model
    save_model(generator, 'enhanced_image_generator.pth')
    print("10")
    # Example of enhancing a single image
    # loaded_generator = Generator()
    # loaded_generator = load_model(loaded_generator, 'enhanced_image_generator.pth', device)
    # low_res_image_path = 'test_image.jpg'  # Replace with your test image path
    # low_res_image = Image.open(low_res_image_path).convert('RGB')
    # enhanced_image = enhance_image(loaded_generator, low_res_image, device)
    # enhanced_image.save('enhanced_image.jpg')
    # print("Enhanced image saved to enhanced_image.jpg")

    # Optional: Convert the model to Core ML
    input_shape = (1, 3, 128, 128)  # Example input shape
    output_shape = (1, 3, 1024, 1024) # Example output shape
    convert_to_coreml('enhanced_image_generator.pth', input_shape, output_shape)

if __name__ == "__main__":
    main()