import torch
import torch.nn as nn


class RGBEncoder(nn.Module):
    def __init__(self, in_ch=3, out_ch=64, kernel_size=5, stride=1, padding=2):
        super(RGBEncoder, self).__init__()

        self.initial_layer = self.conv_relu(in_ch, out_ch, kernel_size, stride, padding)
        self.conv_layer = self.conv_relu(out_ch, out_ch, kernel_size, stride, padding)
        self.downsample_layer = self.conv_relu(out_ch, out_ch, kernel_size, stride=2, padding=padding)

    def conv_relu(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.initial_layer(x)
        conv1 = self.conv_layer(x)
        conv2 = self.downsample_layer(conv1)
        conv3 = self.downsample_layer(conv2)
        conv4 = self.downsample_layer(conv3)
        return conv1, conv2, conv3, conv4


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.shape
        y = self.global_avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class HSIEncoder(nn.Module):
    def __init__(self, in_ch=31, out_ch=64, kernel_size=5, stride=1, padding=2, reduction=16):
        super(HSIEncoder, self).__init__()
        self.initial_layer = self.conv_relu_se(in_ch, out_ch, kernel_size, stride, padding, reduction)
        self.conv_layer = self.conv_relu_se(out_ch, out_ch, kernel_size, stride, padding, reduction)
        self.downsample_layer = self.conv_relu_se(out_ch, out_ch, kernel_size, stride=2, padding=padding, reduction=reduction)

    def conv_relu_se(self, in_channels, out_channels, kernel_size, stride, padding, reduction):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.ReLU(inplace=True),
            SEBlock(out_channels, reduction)
        )

    def forward(self, x):
        x = self.initial_layer(x)
        conv1 = self.conv_layer(x)
        conv2 = self.downsample_layer(conv1)
        conv3 = self.downsample_layer(conv2)
        conv4 = self.downsample_layer(conv3)
        return [conv1, conv2, conv3, conv4]

