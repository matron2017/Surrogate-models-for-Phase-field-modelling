import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def add_frame(input_tensor, num_pad=1):
    output_tensor = F.pad(input_tensor, (num_pad, num_pad, num_pad, num_pad))
    output_tensor[:, :, num_pad:-num_pad, :num_pad] = input_tensor[:, :, :, -num_pad:]
    output_tensor[:, :, num_pad:-num_pad, -num_pad:] = input_tensor[:, :, :, :num_pad]
    output_tensor[:, :, :num_pad, num_pad:-num_pad] = input_tensor[:, :, -num_pad:, :]
    output_tensor[:, :, -num_pad:, num_pad:-num_pad] = input_tensor[:, :, :num_pad, :]
    output_tensor[:, :, :num_pad, :num_pad] = input_tensor[:, :, -num_pad:, -num_pad:]
    output_tensor[:, :, :num_pad, -num_pad:] = input_tensor[:, :, -num_pad:, :num_pad]
    output_tensor[:, :, -num_pad:, :num_pad] = input_tensor[:, :, :num_pad, -num_pad:]
    output_tensor[:, :, -num_pad:, -num_pad:] = input_tensor[:, :, :num_pad, :num_pad]
    return output_tensor

# ✅ Double Convolution Block with GeLU (no BatchNorm)
class DoubleConv(nn.Module):
    """(convolution => GeLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.double_conv(x)

# ✅ Downsampling Block
class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# ✅ Upsampling Block
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# ✅ Output Convolution Layer
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
