import torch.nn as nn
import torch.nn.functional as F
import torch
from correlation import ModuleCorrelation


def backwarp(input, flow):
    B, _, H, W = input.shape

    hor = torch.linspace(-1.0 + (1.0 / W), 1.0 - (1.0 / W), W)
    hor = hor.view(1, 1, 1, -1).expand(-1, -1, H, -1)
    ver = torch.linspace(-1.0 + (1.0 / H), 1.0 - (1.0 / H), H)
    ver = ver.view(1, 1, -1, 1).expand(-1, -1, -1, W)
    grid = torch.cat([hor, ver], 1)

    if input.is_cuda:
        grid = grid.cuda()

    flow = torch.cat([flow[:, 0:1, :, :] / ((W - 1.0) / 2.0),
                      flow[:, 1:2, :, :] / ((H - 1.0) / 2.0)], 1)

    vgrid = grid + flow
    vgrid = vgrid.permute(0, 2, 3, 1)

    input = torch.cat([input, flow.new_ones([B, 1, H, W])], 1)

    output = F.grid_sample(input=input, grid=vgrid, mode='bilinear', padding_mode='border', align_corners=False)

    mask = output[:, -1:, :, :]
    mask[mask > 0.999] = 1.0
    mask[mask < 1.0] = 0.0

    return output[:, :-1, :, :] * mask


class Extractor(nn.Module):
    def __init__(self, in_ch=3, out_ch=16, kernel_size=3, stride=2, padding=1):
        super(Extractor, self).__init__()

        self.layer_one = self.conv_relu(in_ch, out_ch, kernel_size, stride, padding)
        self.layer_two = self.conv_relu(out_ch, out_ch * 2, kernel_size, stride, padding)
        self.layer_three = self.conv_relu(out_ch * 2, out_ch * 4, kernel_size, stride, padding)
        self.layer_four = self.conv_relu(out_ch * 4, out_ch * 6, kernel_size, stride, padding)
        self.layer_five = self.conv_relu(out_ch * 6, out_ch * 8, kernel_size, stride, padding)
        self.layer_siz = self.conv_relu(out_ch * 8, out_ch * 12, kernel_size, stride, padding)

    def conv_relu(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.LeakyReLU(negative_slope=0.1))

    def forward(self, x):
        feat1 = self.layer_one(x)
        feat2 = self.layer_two(feat1)
        feat3 = self.layer_three(feat2)
        feat4 = self.layer_four(feat3)
        feat5 = self.layer_five(feat4)
        feat6 = self.layer_siz(feat5)

        return [feat1, feat2, feat3, feat4, feat5, feat6]


class Decoder(nn.Module):
    def __init__(self, intLevel):
        super(Decoder, self).__init__()

        # Definizione dei canali in ingresso per i diversi livelli
        self.intPrevious = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel + 1]
        self.intCurrent = [None, None, 81 + 32 + 2 + 2, 81 + 64 + 2 + 2, 81 + 96 + 2 + 2, 81 + 128 + 2 + 2, 81, None][intLevel]

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)  # Attivazione globale
        self.correlation = ModuleCorrelation()  # Usa il modulo predefinito per la correlazione

        if intLevel < 6:
            self.netUpflow = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
            self.netUpfeat = nn.ConvTranspose2d(self.intPrevious + 128 + 128 + 96 + 64 + 32, 2, kernel_size=4, stride=2, padding=1)
            self.fltBackwarp = [None, None, None, 5.0, 2.5, 1.25, 0.625, None][intLevel + 1]

        # Creazione sequenziale dei blocchi convoluzionali
        self.conv_blocks = nn.ModuleList([
            self._conv_block(self.intCurrent, 128),
            self._conv_block(self.intCurrent + 128, 128),
            self._conv_block(self.intCurrent + 128 + 128, 96),
            self._conv_block(self.intCurrent + 128 + 128 + 96, 64),
            self._conv_block(self.intCurrent + 128 + 128 + 96 + 64, 32),
            nn.Conv2d(self.intCurrent + 128 + 128 + 96 + 64 + 32, 2, kernel_size=3, stride=1, padding=1)
        ])

    def _conv_block(self, in_channels, out_channels):
        """Helper function to create a Conv2D + LeakyReLU block"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, tenOne, tenTwo, objPrevious=None):
        if objPrevious is None:
            tenVolume = self.correlation(tenOne, tenTwo)  # Usa il modulo di correlazione
            tenFeat = self.leaky_relu(tenVolume)  # Applico LeakyReLU direttamente
            tenFlow = None
        else:
            tenFlow = self.netUpflow(objPrevious['tenFlow'])
            tenFeat = self.netUpfeat(objPrevious['tenFeat'])
            tenVolume = self.correlation(tenOne, backwarp(tenTwo, tenFlow * self.fltBackwarp))
            tenFeat = torch.cat([self.leaky_relu(tenVolume), tenOne, tenFlow, tenFeat], dim=1)

        # Passaggio attraverso i livelli convoluzionali
        for conv_block in self.conv_blocks[:-1]:  # Tutti tranne l'ultimo layer
            tenFeat = torch.cat([conv_block(tenFeat), tenFeat], dim=1)

        tenFlow = self.conv_blocks[-1](tenFeat)  # Ultimo livello per ottenere il flow

        return {'tenFlow': tenFlow, 'tenFeat': tenFeat}


im1 = torch.rand(3, 512, 512)
im2 = torch.rand(3, 512, 512)
encoder = Extractor()
decoder = Decoder(3)
features1 = encoder(im1)
features2 = encoder(im2)
estimate = decoder(features1[5], features2[5], None)
estimate_2 = decoder(features1[4], features2[4], estimate)
print(estimate.shape)
print(estimate_2.shape)



