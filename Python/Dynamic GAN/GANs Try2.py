# models/attention.py
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    """Self-Attention Module."""
    def __init__(self, in_dim):
        super(SelfAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.key_conv = nn.Conv2d(in_dim, in_dim // 8, 1)
        self.value_conv = nn.Conv2d(in_dim, in_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, C, W, H = x.size()
        proj_query = self.query_conv(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(B, -1, W * H)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(B, -1, W * H)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(B, C, W, H)

        out = self.gamma * out + x
        return out


# models/generator.py
import torch
import torch.nn as nn
import torchvision.models as models

from models.attention import SelfAttention

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(Generator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU()
        )

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_residual_blocks)],
            SelfAttention(64)  # Attention after Residual Blocks
        )

        self.upsample = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),

            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU()
        )

        self.final = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        x = self.initial(x)
        residual = x
        x = self.residual_blocks(x)
        x = x + residual
        x = self.upsample(x)
        x = self.final(x)
        return torch.tanh(x)


# models/discriminator.py
import torch.nn as nn

def discriminator_block(in_filters, out_filters, stride):
    block = nn.Sequential(
        nn.Conv2d(in_filters, out_filters, 3, stride, 1),
        nn.BatchNorm2d(out_filters),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2, inplace=True),

            discriminator_block(64, 64, 2),
            discriminator_block(64, 128, 1),
            discriminator_block(128, 128, 2),
            discriminator_block(128, 256, 1),
            discriminator_block(256, 256, 2),
            discriminator_block(256, 512, 1),
            discriminator_block(512, 512, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 1)
        )

    def forward(self, img):
        return self.model(img).view(img.size(0))
