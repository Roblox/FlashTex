import torch
import torch.nn as nn

from diffusers.models.resnet import ResnetBlock2D


class SmallEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, stride=2, padding=2)
        self.norm1 = nn.GroupNorm(8, 64)
        self.actv1 = nn.SiLU()

        self.conv2a = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.norm2a = nn.GroupNorm(16, 128)
        self.actv2a = nn.SiLU()
        self.conv2b = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.norm2b = nn.GroupNorm(16, 128)
        self.actv2b = nn.SiLU()
        self.conv2c = nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1)
        self.norm2c = nn.GroupNorm(16, 128)
        self.actv2c = nn.SiLU()

        self.conv3a = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.norm3a = nn.GroupNorm(32, 256)
        self.actv3a = nn.SiLU()
        self.conv3b = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.norm3b = nn.GroupNorm(32, 256)
        self.actv3b = nn.SiLU()
        self.conv3c = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.norm3c = nn.GroupNorm(32, 256)
        self.actv3c = nn.SiLU()

        self.conv_final = nn.Conv2d(256, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.actv1(self.norm1(self.conv1(x)))
        x = self.actv2a(self.norm2a(self.conv2a(x)))
        x = self.actv2b(self.norm2b(self.conv2b(x)))
        x = self.actv2c(self.norm2c(self.conv2c(x)))
        x = self.actv3a(self.norm3a(self.conv3a(x)))
        x = self.actv3b(self.norm3b(self.conv3b(x)))
        x = self.actv3c(self.norm3c(self.conv3c(x)))
        x = self.conv_final(x)
        return x


class SmallResnetEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=5, stride=2, padding=2)
        self.norm1 = nn.GroupNorm(16, 128)
        self.actv1 = nn.SiLU()

        self.conv2a = ResnetBlock2D(in_channels=128, out_channels=256, groups=32, temb_channels=None)
        self.conv2b = ResnetBlock2D(in_channels=256, out_channels=256, groups=32, temb_channels=None)
        self.conv2c = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.norm2c = nn.GroupNorm(32, 256)
        self.actv2c = nn.SiLU()

        self.conv3a = ResnetBlock2D(in_channels=256, out_channels=512, groups=32, temb_channels=None)
        self.conv3b = ResnetBlock2D(in_channels=512, out_channels=512, groups=32, temb_channels=None)
        self.conv3c = nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1)
        self.norm3c = nn.GroupNorm(32, 512)
        self.actv3c = nn.SiLU()

        self.conv4a = ResnetBlock2D(in_channels=512, out_channels=512, groups=32, temb_channels=None)
        self.conv4b = ResnetBlock2D(in_channels=512, out_channels=512, groups=32, temb_channels=None)

        self.conv_final = nn.Conv2d(512, out_channels, kernel_size=1, stride=1, padding=0)
        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_uniform_(m.weight)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        x = self.actv1(self.norm1(self.conv1(x)))
        x = self.conv2a(x, None)
        x = self.conv2b(x, None)
        x = self.actv2c(self.norm2c(self.conv2c(x)))
        x = self.conv3a(x, None)
        x = self.conv3b(x, None)
        x = self.actv3c(self.norm3c(self.conv3c(x)))
        x = self.conv4a(x, None)
        x = self.conv4b(x, None)
        x = self.conv_final(x)
        return x
