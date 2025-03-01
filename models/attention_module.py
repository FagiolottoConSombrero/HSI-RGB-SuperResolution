import torch
from torch import nn


def get_activation(activation, activation_params=None, num_channels=None):
    if activation_params is None:
        activation_params = {}

    if activation == 'relu':
        return nn.ReLU(inplace=True)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    elif activation == 'lrelu':
        return nn.LeakyReLU(negative_slope=activation_params.get('negative_slope', 0.1), inplace=True)
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'prelu':
        return nn.PReLU(num_parameters=num_channels)
    elif activation == 'none':
        return None
    else:
        raise Exception('Unknown activation {}'.format(activation))


def conv_block(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, bias=True,
               batch_norm=False, activation='relu', padding_mode='zeros', activation_params=None):
    layers = []

    layers.append(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=bias, padding_mode=padding_mode))

    if batch_norm:
        layers.append(nn.BatchNorm2d(out_planes))

    activation_layer = get_activation(activation, activation_params, num_channels=out_planes)
    if activation_layer is not None:
        layers.append(activation_layer)

    return nn.Sequential(*layers)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1, batch_norm=False, activation='relu',
                 padding_mode='zeros', attention='none'):
        super(ResBlock, self).__init__()
        self.conv1 = conv_block(inplanes, planes, kernel_size=3, padding=1, stride=stride, dilation=dilation,
                                batch_norm=batch_norm, activation=activation, padding_mode=padding_mode)

        self.conv2 = conv_block(planes, planes, kernel_size=3, padding=1, dilation=dilation, batch_norm=batch_norm,
                                activation='none', padding_mode=padding_mode)

        self.downsample = downsample
        self.stride = stride

        self.activation = get_activation(activation, num_channels=planes)

    def forward(self, x):
        residual = x

        out = self.conv2(self.conv1(x))

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.activation(out)

        return out


class MaskPredictor(nn.Module):
    def __init__(self, input_dim=64, project_dim=32, offset_feat_dim=32,
                 num_offset_feat_extractor_res=1, num_weight_predictor_res=1, use_offset=True, offset_modulo=1.0,
                 use_bn=False, activation='relu'):
        super().__init__()
        self.use_offset = use_offset
        self.offset_modulo = offset_modulo

        self.feat_project_layer = conv_block(input_dim, project_dim, 1, stride=1, padding=0,
                                                    batch_norm=use_bn,
                                                    activation=activation)

        offset_feat_extractor = []
        offset_feat_extractor.append(conv_block(2, offset_feat_dim, 3, stride=1, padding=1, batch_norm=use_bn,
                                                       activation=activation))

        for _ in range(num_offset_feat_extractor_res):
            offset_feat_extractor.append(ResBlock(offset_feat_dim, offset_feat_dim, stride=1,
                                                         batch_norm=use_bn, activation=activation))
        self.offset_feat_extractor = nn.Sequential(*offset_feat_extractor)

        weight_predictor = []
        weight_predictor.append(conv_block(project_dim * 2 + offset_feat_dim * use_offset, 2 * project_dim, 3,
                                                  stride=1, padding=1, batch_norm=use_bn, activation=activation))

        for _ in range(num_weight_predictor_res):
            weight_predictor.append(ResBlock(2 * project_dim, 2 * project_dim, stride=1,
                                                    batch_norm=use_bn, activation=activation))

        weight_predictor.append(conv_block(2 * project_dim, 1, 3, stride=1, padding=1,
                                                  batch_norm=use_bn,
                                                  activation='none'))

        self.weight_predictor = nn.Sequential(*weight_predictor)

    def forward(self, x, ref, flow):
        x_proj = self.feat_project_layer(x)
        ref_proj = self.feat_project_layer(ref)
        # flow_feat = self.offset_feat_extractor(flow)
        flow_feat = self.offset_feat_extractor(flow % self.offset_modulo)
        weight_pred_in = [x_proj, ref_proj, flow_feat]
        weight_pred_in = torch.cat(weight_pred_in, dim=1)
        weight = self.weight_predictor(weight_pred_in)
        weight_norm = torch.sigmoid(weight)
        return weight_norm


