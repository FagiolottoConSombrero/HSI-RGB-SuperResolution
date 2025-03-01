import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.first_conv = self.make_up_conv(128, 64)
        self.second_conv = self.make_up_conv(192, 64)
        self.third_conv = self.make_up_conv(192, 64)
        self.last_conv = self.make_conv(192, 31)

    def make_up_conv(self, in_channels, out_channels, kernel_size=2, stride=2):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def make_conv(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

    def forward(self, hsi_feats, warp_feats):
        x = torch.cat((hsi_feats[3], warp_feats[3]), dim=1)
        x = self.first_conv(x)
        conc = torch.cat((x, hsi_feats[2], warp_feats[2]), dim=1)
        x = self.second_conv(conc)
        conc = torch.cat((x, hsi_feats[1], warp_feats[1]), dim=1)
        x = self.third_conv(conc)
        conc = torch.cat((x, hsi_feats[0], warp_feats[0]), dim=1)
        x = self.last_conv(conc)

        return x
