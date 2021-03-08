import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import List, Tuple, Dict


def _make_linear_block(
        in_dim: int,
        out_dim: int,
        act: nn.Module = lambda: nn.ReLU(inplace=True),
        batchnorm: bool = True) -> nn.Sequential:

    if batchnorm:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            act())
    else:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            act())

class InfoGANGenerator(nn.Module):

    def __init__(self,
                 noise_dim: int,
                 code_dim: int,
                 feature_map: int,
                 img_channels: int,
                 conv_start_shape: List,
                 fc_layers: List = [],
                 fc_act: nn.Module = lambda: nn.ReLU(inplace=True),
                 act: nn.Module = lambda: nn.ReLU(inplace=True)) -> None:
        """
        Args:
            noise_dim: Dimension of noise vector z
            code_dim: Dimension of latent code c
            feature_maps: Number of feature maps to use
            img_channels: Number of channels in the images
            fc_layers: Fully connected block dimensions prior to conv blocks
            fc_act: Activation fn for fully connected layers
            act: Activations fn for conv layers
        """
        super().__init__()

        self._noise_dim = noise_dim
        self._code_dim = code_dim
        self._img_channels = img_channels
        self._conv_start_shape = conv_start_shape

        fc_layers = fc_layers + [math.prod(conv_start_shape)]
        fc_modules = nn.ModuleList()
        inp = noise_dim + code_dim
        for layer_dim in fc_layers:
            fc_modules.append(_make_linear_block(inp, layer_dim, act=fc_act))
            inp = layer_dim
        self._fc = nn.Sequential(*fc_modules)

        self._deconv = nn.Sequential(
            self._make_deconv_block(feature_map*2, feature_map, act=act),
            self._make_deconv_block(feature_map, 1,
                                    batchnorm=False, act=nn.Tanh)
        )

    @staticmethod
    def _make_deconv_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            act: nn.Module = lambda: nn.ReLU(inplace=True),
            batchnorm: bool = True) -> nn.Sequential:

        if batchnorm:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                act(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels,
                                   kernel_size, stride, padding, bias=True),
                act()
            )

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = self._fc(torch.cat([z, c], dim=-1))
        x = x.reshape(-1, *self._conv_start_shape)
        x = self._deconv(x)
        return x


class InfoGANDiscriminator(nn.Module):
    def __init__(self,
                 categ_code_dim: int,
                 cont_code_dim: int,
                 feature_map: int,
                 img_channels: int,
                 conv_end_shape: List,
                 fc_layers: List = [],
                 act: nn.Module = lambda: nn.LeakyReLU(0.1, inplace=True),
                 fc_act: nn.Module = lambda: nn.LeakyReLU(0.1, inplace=True)) -> None:
        """
        Args:
            categ_code_dim: Number of categories in the categorical distribution
            cont_code_dim: Number of gaussion variables
            feature_maps: Max number of feature maps
            img_channels: Number of channels in the images
            fc_layers: Fully connected block dimensions prior to conv blocks
            fc_act: Activation fn for fully connected layers
            act: Activations fn for conv layers
        """
        super().__init__()
        self._cont_code_dim = cont_code_dim
        self._categ_code_dim = categ_code_dim
        self._img_channels = img_channels

        conv_modules = nn.ModuleList([
            self._make_conv_block(img_channels, feature_map,
                                  batchnorm=False, act=act),
            self._make_conv_block(feature_map, 2*feature_map, act=act)
        ])

        fc_modules = nn.ModuleList()
        inp = math.prod(conv_end_shape)
        for layer_dim in fc_layers:
            fc_modules.append(_make_linear_block(inp, layer_dim, act=fc_act))
            inp = layer_dim

        self._base = nn.Sequential(*conv_modules, nn.Flatten(), *fc_modules)

        self._disc = nn.Linear(inp, 1)
        self._q_network = nn.Sequential(
            _make_linear_block(inp, 128, act=fc_act),
            nn.Linear(128, categ_code_dim + 2*cont_code_dim)
        )

    @staticmethod
    def _make_conv_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 0,
            act: nn.Module = lambda: nn.LeakyReLU(0.1, inplace=True),
            batchnorm: bool = True) -> nn.Sequential:

        if batchnorm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                act()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride, padding, bias=True),
                act()
            )

    def forward(self,
                x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        x = self._base(x)
        disc = self._disc(x)
        q_hid = self._q_network(x)
        q_categ = q_hid[:, :self._categ_code_dim]
        q_gauss = q_hid[:, self._categ_code_dim:]
        q_gauss_mean, q_gauss_std = q_gauss.split(2, dim=-1)
        q_gauss_std = F.softplus(q_gauss_std)
        q_out = {'categ': q_categ, 'gauss_mean': q_gauss_mean,
                 'gauss_std': q_gauss_std}
        return (disc, q_out)
