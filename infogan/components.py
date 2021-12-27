import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import List, Tuple, Dict


def _make_linear_block(
        in_dim: int,
        out_dim: int,
        act: nn.Module = lambda: nn.ReLU(inplace=True),
        batchnorm: bool = True,
        spectralnorm: bool = False) -> nn.Sequential:
    assert not(
        batchnorm and spectralnorm), "Can't use batchnorm and spectralnorm together"

    if batchnorm:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
            act())
    elif spectralnorm:
        return nn.Sequential(
            nn.utils.spectral_norm(nn.Linear(in_dim, out_dim)),
            act())
    else:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            act())


class InfoGANGenerator(nn.Module):

    def __init__(self,
                 noise_dim: int,
                 code_dim: int,
                 feature_maps: List,
                 img_channels: int,
                 conv_start_shape: List,
                 fc_layers: List = [],
                 fc_act: nn.Module = lambda: nn.ReLU(inplace=True),
                 act: nn.Module = lambda: nn.ReLU(inplace=True)) -> None:
        """
        Args:
            noise_dim        - Dimension of noise vector z
            code_dim         - Dimension of latent code c
            feature_maps     - Feature map size for each deconv layer
            img_channels     - Number of channels in the ouput image
            conv_start_shape - Shape of the input to the deconv block
            fc_layers        - Fully connected layer dimensions prior to deconv blocks
            fc_act           - Activation fn for fully connected layers
            act              - Activations fn for deconv layers
        """
        super().__init__()

        self._noise_dim = noise_dim
        self._code_dim = code_dim
        self._img_channels = img_channels
        self._conv_start_shape = conv_start_shape

        fc_layers = fc_layers + [math.prod(conv_start_shape)]
        fc_modules = []
        inp = noise_dim + code_dim
        for layer_dim in fc_layers:
            fc_modules.append(_make_linear_block(inp, layer_dim, act=fc_act))
            inp = layer_dim
        self._fc = nn.Sequential(*fc_modules)

        deconv_modules = []
        deconv_modules.append(self._make_deconv_block(
            conv_start_shape[0], feature_maps[0], act=act))
        for i in range(1, len(feature_maps)):
            deconv_modules.append(self._make_deconv_block(feature_maps[i-1], feature_maps[i],
                                                          act=act, batchnorm=False))
        deconv_modules.append(self._make_deconv_block(feature_maps[-1], img_channels,
                                                      act=nn.Tanh, batchnorm=False))
        self._deconv = nn.Sequential(*deconv_modules)

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
                                   kernel_size, stride, padding, bias=False),
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
                 feature_maps: List,
                 img_channels: int,
                 conv_end_shape: List,
                 fc_layers: List = [],
                 act: nn.Module = lambda: nn.LeakyReLU(0.1, inplace=True),
                 fc_act: nn.Module = lambda: nn.LeakyReLU(0.1, inplace=True),
                 sn: bool = True) -> None:
        """
        Args:
            categ_code_dim - Total dimension of all categorical distribution variables
            cont_code_dim  - Number of gaussion variables
            feature_maps   - Feature map size for each conv layer
            img_channels   - Number of channels in the input image
            conv_end_shape - Shape of the output form the conv_layers for batch size=1
            fc_layers      - Fully connected layer dimensions after the conv blocks
            act            - Activations fn for conv layers
            fc_act         - Activation fn for fully connected layers
            sn             - Use Spectral Normalization
        """
        super().__init__()
        self._cont_code_dim = cont_code_dim
        self._categ_code_dim = categ_code_dim
        self._img_channels = img_channels

        conv_modules = []
        conv_modules.append(self._make_conv_block(img_channels, feature_maps[0], batchnorm=False,
                                                  spectralnorm=sn, act=act))
        for i in range(1, len(feature_maps)):
            conv_modules.append(self._make_conv_block(feature_maps[i-1], feature_maps[i], batchnorm=not sn,
                                                      spectralnorm=sn, act=act))

        fc_modules = []
        inp = math.prod(conv_end_shape)
        for layer_dim in fc_layers:
            fc_modules.append(_make_linear_block(inp, layer_dim, act=fc_act,
                                                 batchnorm=not sn, spectralnorm=sn))
            inp = layer_dim

        self.base = nn.Sequential(*conv_modules, nn.Flatten(), *fc_modules)
        self.base_feat_dim = inp

        self.disc = nn.Linear(inp, 1)

    @staticmethod
    def _make_conv_block(
            in_channels: int,
            out_channels: int,
            kernel_size: int = 4,
            stride: int = 2,
            padding: int = 1,
            act: nn.Module = lambda: nn.LeakyReLU(0.1, inplace=True),
            batchnorm: bool = True,
            spectralnorm: bool = False) -> nn.Sequential:

        assert not(
            batchnorm and spectralnorm), "Can't use batchnorm and spectralnorm together"

        if batchnorm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_channels),
                act()
            )
        elif spectralnorm:
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Conv2d(in_channels, out_channels,
                                                 kernel_size, stride, padding)),
                act()
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size,
                          stride, padding, bias=True),
                act()
            )

    def forward(self, x: torch.Tensor,
                need_base_feat: bool = True) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        base_feat = self.base(x)
        disc = self.disc(base_feat)
        if need_base_feat:
            return (disc, base_feat)
        else:
            return disc


class QNetwork(nn.Module):

    def __init__(self,
                 input_dim: int,
                 categ_code_dim: int,
                 cont_code_dim: int,
                 act: nn.Module = lambda: nn.LeakyReLU(0.1, inplace=True)):

        super().__init__()
        self.q_network = nn.Sequential(
            _make_linear_block(input_dim, 128, act=act),
            nn.Linear(128, categ_code_dim + 2*cont_code_dim)
        )
        self._categ_code_dim = categ_code_dim
        self._cont_code_dim = cont_code_dim

    def forward(self, x):
        q_hid = self.q_network(x)
        q_categ = q_hid[:, :self._categ_code_dim]
        if self._cont_code_dim > 0:
            q_gauss = q_hid[:, self._categ_code_dim:]
            q_gauss_mean, q_gauss_std = q_gauss.chunk(2, dim=-1)
            q_gauss_std = torch.sqrt(q_gauss_std.exp())
            q_out = {'categ': q_categ, 'gauss_mean': q_gauss_mean,
                     'gauss_std': q_gauss_std}
        else:
            q_out = {'categ': q_categ}
        return q_out
