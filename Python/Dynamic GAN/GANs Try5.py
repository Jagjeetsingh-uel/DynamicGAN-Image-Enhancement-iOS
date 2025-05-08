#Try 3
#Dict Path  /Users/jagjeetsinghlabana/Documents/Projects/Python/GAN/images/003.jpg

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm
import torchvision.models as models  # For ONNX export

# --- 1. Define the Generator (SRCNN) ---
class SRCNN(nn.Module):
    def __init__(self, num_channels=3, num_filters=64):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(num_filters, num_filters // 2, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(num_filters // 2, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    def forward_with_relu(self, x): #added relu
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x)) # added final ReLU
        return x


# --- 2. Define the Discriminator (Simplified) ---
class Discriminator(nn.Module):
    def __init__(self, num_channels=3, num_filters=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(num_filters, num_filters * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(num_filters * 2, num_filters * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(num_filters * 4, 1, kernel_size=4, stride=2, padding=0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.conv4(x)
        return torch.sigmoid(x)  # Output probability


# --- 3. Define the Dataset ---
class SRDataset(Dataset):
    def __init__(self, hr_dir, lr_dir, scale_factor=2):
        super(SRDataset, self).__init__()
        self.hr_image_paths = sorted([os.path.join(hr_dir, filename) for filename in os.listdir(hr_dir)])
        self.lr_image_paths = sorted([os.path.join(lr_dir, filename) for filename in os.listdir(lr_dir)])
        self.scale_factor = scale_factor
        self.to_tensor = transforms.ToTensor()
        self.downsample = transforms.Resize(scale_factor, interpolation=transforms.InterpolationMode.BICUBIC)


    def __len__(self):
        return len(self.hr_image_paths)

    def __getitem__(self, index):
        hr_image = Image.open(self.hr_image_paths[index]).convert('RGB')  # Ensure RGB format
        hr_image = self.to_tensor(hr_image)

        lr_image = Image.open(self.lr_image_paths[index]).convert('RGB')
        lr_image = self.to_tensor(lr_image)
        return lr_image, hr_image



# --- 4. Training Function ---
def train_gan(hr_dir, lr_dir, epochs=100, batch_size=64, scale_factor=2, device='cuda'):
    """
    Trains the SRGAN model using PyTorch.

    Args:
        hr_dir (str): Path to the directory containing high-resolution images.
        lr_dir (str): Path to the directory containing low-resolution images.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        scale_factor (int): The scale factor between HR and LR images.
        device (str): 'cuda' if GPU is available, 'cpu' otherwise.
    """
    # 1. Check for empty directories
    if not os.listdir(hr_dir) or not os.listdir(lr_dir):
        raise ValueError("Both HR and LR directories must contain images.")

    # 2. Prepare Data
    train_dataset = SRDataset(hr_dir, lr_dir, scale_factor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, drop_last=True)

    # 3. Build Model
    generator = SRCNN(num_channels=3).to(device)  # RGB images
    discriminator = Discriminator(num_channels=3).to(device)  # RGB images
    #generator.forward = generator.forward_with_relu # change forward pass

    # 4. Define Loss Functions and Optimizers
    criterion_gan = nn.BCELoss()  # Binary Cross-Entropy Loss for GAN
    criterion_pixel = nn.MSELoss()  # Pixel-wise MSE loss
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # 5. Train the Models
    for epoch in range(epochs):
        total_loss_g = 0.0
        total_loss_d = 0.0
        with tqdm(total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{epochs}', unit='batch') as pbar:
            for lr_batch, hr_batch in train_dataloader:
                lr_batch = lr_batch.to(device)
                hr_batch = hr_batch.to(device)

                # --- Train Discriminator ---
                optimizer_d.zero_grad()
                # Real images:
                real_labels = torch.ones((hr_batch.size(0), 1, 1, 1)).to(device)  # Correct shape for判别器
                outputs_real = discriminator(hr_batch)
                loss_d_real = criterion_gan(outputs_real, real_labels)
                # Fake images:
                sr_batch = generator(lr_batch) # changed from generator(lr_batch)
                fake_labels = torch.zeros((sr_batch.size(0), 1, 1, 1)).to(device)  # Correct shape for 判别器
                outputs_fake = discriminator(sr_batch.detach())  # Detach to avoid training generator
                loss_d_fake = criterion_gan(outputs_fake, fake_labels)
                # Total discriminator loss
                loss_d = loss_d_real + loss_d_fake
                loss_d.backward()
                optimizer_d.step()
                total_loss_d += loss_d.item()

                # --- Train Generator ---
                optimizer_g.zero_grad()
                valid_labels = torch.ones((sr_batch.size(0), 1, 1, 1)).to(device)  # Generator wants discriminator to think they are real
                outputs_g = discriminator(sr_batch)
                loss_g_gan = criterion_gan(outputs_g, valid_labels)
                loss_g_pixel = criterion_pixel(sr_batch, hr_batch)
                loss_g = loss_g_gan + 1e-2 * loss_g_pixel  # Combine GAN loss and pixel-wise loss.  Adjust the weight (1e-2) as needed.
                loss_g.backward()
                optimizer_g.step()
                total_loss_g += loss_g.item()

                pbar.set_postfix(loss_g=loss_g.item(), loss_d=loss_d.item())
                pbar.update(1)
        print(f"Epoch {epoch + 1}/{epochs}, Avg Loss G: {total_loss_g / len(train_dataloader)}, Avg Loss D: {total_loss_d / len(train_dataloader)}")
    return generator



# --- 5. Main Execution ---
if __name__ == '__main__':
    # Set the device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Set Data Directories
    hr_dir = '/Users/jagjeetsinghlabana/Documents/Projects/Python/GAN/data/high_res'  # Replace with your high-resolution image directory
    lr_dir = '/Users/jagjeetsinghlabana/Documents/Projects/Python/GAN/data/low_res'  # Replace with your low-resolution image directory

    # Create dummy directories and files if they don't exist
    if not os.path.exists(hr_dir):
        os.makedirs(hr_dir)
        for i in range(3):
            img = Image.new('RGB', (100, 100), color=(255, 0, 0))
            img.save(os.path.join(hr_dir, f'hr_{i+1}.jpg'))
    if not os.path.exists(lr_dir):
        os.makedirs(lr_dir)
        for i in range(3):
            img = Image.new('RGB', (50, 50), color=(0, 0, 255))  # Smaller size for LR
            img.save(os.path.join(lr_dir, f'lr_{i+1}.jpg'))

    # 2. Train SRGAN Model
    generator_model = train_gan(hr_dir, lr_dir, epochs=10, batch_size=64, scale_factor=2, device=device) # Reduced epochs

    # 3. Save PyTorch Model (.pth)
    torch.save(generator_model.state_dict(), 'SRGAN_Generator.pth')
    print("PyTorch Generator model saved as SRGAN_Generator.pth")

    # 4. Export to ONNX
    # Define the input shape for the ONNX model
    input_shape = (1, 3, 256, 256)  # Example: (batch_size, channels, height, width)
    generator_model.eval()  # Set to evaluation mode before exporting
    dummy_input = torch.randn(input_shape).to(device)
    torch.onnx.export(generator_model,
                      dummy_input,
                      "SRGAN_Generator.onnx",
                      export_params=True,
                      opset_version=10,  # Choose an appropriate ONNX opset version
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'])
    print("ONNX Generator model exported to SRGAN_Generator.onnx")
