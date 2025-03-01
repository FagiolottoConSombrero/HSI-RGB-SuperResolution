from models.encoder import *
from models.optical_net import *
from models.decoder import *
from models.attention_module import MaskPredictor


class AlignmentNet(nn.Module):
    """
    Modello fatto da Matteo Kolyszko
    """
    def __init__(self):
        super(AlignmentNet, self).__init__()
        self.encoder_rgb = RGBEncoder()
        self.encoder_hsi = HSIEncoder()
        self.flow_net_1 = FlowNet()
        self.flow_net_2 = FlowNet()
        self.warp = Warp()
        self.attention_module = MaskPredictor()
        self.decoder = Decoder()

    def coarse_alignment(self, lr_rgb, hr_rgb):
        first_flow = self.flow_net_1(lr_rgb, hr_rgb) # lista di 4 tensori
        warped_im = self.warp(hr_rgb, first_flow[0])
        return warped_im, first_flow[0]

    def feature_alignment(self, lr_rgb, warped_im):
        warp_features = self.encoder_rgb(warped_im) # lista di 4 tensori
        rgb_features = self.encoder_rgb(lr_rgb) # lista di 4 tensori
        second_flow = self.flow_net_2(lr_rgb, warped_im) # lista di 4 tensori
        aligned_features = []
        for i in range(4):
            warped_im = self.warp(warp_features[i], second_flow[i])
            corrected_im = self.attention_module(rgb_features[i], warp_features[i], second_flow[i])
            aligned_feature = warped_im * corrected_im
            aligned_features.append(aligned_feature)
        return aligned_features

    def forward(self, lr_rgb, hr_rgb, lr_hsi):
        warped_im, first_flow = self.coarse_alignment(lr_rgb, hr_rgb)
        aligned_features = self.feature_alignment(lr_rgb, warped_im)
        hsi_features = self.encoder_hsi(lr_hsi)
        hr_hsi = self.decoder(hsi_features, aligned_features)
        return hr_hsi
