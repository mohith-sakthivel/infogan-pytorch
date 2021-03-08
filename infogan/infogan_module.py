from argparse import ArgumentParser
from typing import Any, List, Tuple

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as dist
from torch.utils.data import DataLoader

from infogan.components import InfoGANGenerator, InfoGANDiscriminator

from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST


class InfoGAN(pl.LightningModule):
    """
    InfoGAN Implementation
    """

    def __init__(self,
                 lambda_coeff: float = 1,
                 betas_opt: Tuple[float, float] = (0.5, 0.999),
                 feature_maps_gen: int = 64,
                 feature_maps_disc: int = 64,
                 fc_layers_gen: List = [1024, ],
                 fc_layers_disc: List = [1024, ],
                 img_channels: int = 1,
                 noise_dim: int = 62,
                 categ_code_dim: int = 10,
                 cont_code_dim: int = 2,
                 conv_start_shape_gen: List = [128, 7, 7],
                 conv_end_shape_disc: List = [128, 5, 5],
                 learning_rate_gen: float = 1e-3,
                 learning_rate_disc: float = 2e-4,
                 **kwargs: Any,
                 ) -> None:
        """
        Args:

        """

        super().__init__()
        self.save_hyperparameters()

        self.generator = InfoGANGenerator(noise_dim,
                                          categ_code_dim+cont_code_dim,
                                          feature_maps_gen,
                                          img_channels,
                                          conv_start_shape=conv_start_shape_gen,
                                          fc_layers=fc_layers_gen)
        self.generator.apply(self._initialize_weights)

        self.discriminator = InfoGANDiscriminator(categ_code_dim,
                                                  cont_code_dim,
                                                  feature_maps_disc,
                                                  img_channels,
                                                  conv_end_shape_disc,
                                                  fc_layers=fc_layers_disc)
        self.discriminator.apply(self._initialize_weights)

        self._initialize_samplers()
        self.adverserial_loss = nn.BCEWithLogitsLoss()
        self.categorical_loss = nn.CrossEntropyLoss()
        self.gaussian_loss = nn.MSELoss()

    def _initialize_samplers(self) -> None:
        if self.hparams.categ_code_dim > 0:
            self._categorical_dist = dist.OneHotCategorical(
                logits=torch.ones(self.hparams.categ_code_dim))
        if self.hparams.cont_code_dim > 0:
            self._normal_dist = dist.MultivariateNormal(torch.zeros(self.hparams.cont_code_dim),
                                                        torch.eye(self.hparams.cont_code_dim))

    @staticmethod
    def _initialize_weights(module) -> None:
        classname = module.__class__.__name__
        if classname.find("Conv") != -1:
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(module.weight, 1.0, 0.02)
            torch.nn.init.zeros_(module.bias)

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List]:
        generator_opt = optim.Adam(self.generator.parameters(),
                                   lr=self.hparams.learning_rate_gen,
                                   betas=self.hparams.betas_opt)
        discriminator_opt = optim.Adam(self.discriminator.parameters(),
                                       lr=self.hparams.learning_rate_disc,
                                       betas=self.hparams.betas_opt)
        return ([generator_opt, discriminator_opt], [])

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Generates an image
        Args:
            z: Noise vector
            c: Latent code
        """
        return self.generator(z, c)

    def training_step(self,
                      batch: Tuple,
                      batch_idx: int,
                      optimizer_idx: int) -> torch.Tensor:
        real_img, _ = batch
        result = None
        # Train Generator/Discriminator based on optimizer id
        if optimizer_idx == 0:
            result = self._gen_step(real_img)
        elif optimizer_idx == 1:
            result = self._disc_step(real_img)
        return result

    def _gen_step(self, real_img: torch.Tensor) -> torch.Tensor:
        gen_loss = self._get_gen_loss(len(real_img))
        self.log("loss/gen", gen_loss, on_epoch=True)
        return gen_loss

    def _disc_step(self, real_img: torch.Tensor) -> torch.Tensor:
        disc_loss = self._get_disc_loss(real_img)
        self.log("loss/disc", disc_loss, on_epoch=True)
        return disc_loss

    def _sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn((batch_size, self.hparams.noise_dim))

    def _sample_code(self, batch_size: int) -> torch.Tensor:
        code = None
        if self.hparams.categ_code_dim > 0:
            code = self._categorical_dist.sample([batch_size])
        if self.hparams.cont_code_dim > 0:
            cont_code = self._normal_dist.sample([batch_size])
            code = torch.cat([code, cont_code],
                             dim=-1) if code is not None else cont_code
        return code

    def _get_latents(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self._sample_noise(batch_size)
        c = self._sample_code(batch_size)
        return z, c

    def _get_fake_pred(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        fake_img = self.generator(z, c)
        return self.discriminator(fake_img)

    def _get_gen_loss(self, batch_size: int) -> torch.Tensor:
        # Calculate adverserial loss
        z, c = self._get_latents(batch_size)
        fake_pred, q_pred = self._get_fake_pred(z, c)
        target = torch.ones_like(fake_pred)
        adv_loss = self.adverserial_loss(fake_pred, target)

        q_categ_loss, q_gauss_loss = (torch.zeros([]), torch.zeros([]))
        # Calculate loss from categorical latent code prediction
        if self.hparams.categ_code_dim > 0:
            q_categ_loss = self.categorical_loss(
                q_pred['categ'], c[:, :self.hparams.categ_code_dim].argmax(dim=-1))
        # Calculate loss from gaussian latent code prediction
        if self.hparams.cont_code_dim > 0:
            q_gauss = dist.Independent(dist.Normal(
                q_pred['gauss_mean'], q_pred['gauss_std']), reinterpreted_batch_ndims=1)
            q_gauss_loss = self.gaussian_loss(
                q_gauss.rsample(), c[:, self.hparams.categ_code_dim:])
        mi_loss = q_categ_loss + q_gauss_loss

        return adv_loss + self.hparams.lambda_coeff * mi_loss

    def _get_disc_loss(self, real_img: torch.Tensor) -> torch.Tensor:
        # Calculate adverserial loss from real images
        real_pred, _ = self.discriminator(real_img)
        real_target = torch.ones_like(real_pred)
        real_loss = self.adverserial_loss(real_pred, real_target)
        # Calculate adverserial loss from fake images
        z, c = self._get_latents(len(real_img))
        fake_pred, q_pred = self._get_fake_pred(z, c)
        fake_target = torch.zeros_like(fake_pred)
        fake_loss = self.adverserial_loss(fake_pred, fake_target)

        adv_loss = (real_loss + fake_loss)/2

        # Calculate loss from categorical latent code prediction
        if self.hparams.categ_code_dim > 0:
            q_categ_loss = self.categorical_loss(
                q_pred['categ'], c[:, :self.hparams.categ_code_dim].argmax(dim=-1))
        # Calculate loss from gaussian latent code prediction
        if self.hparams.cont_code_dim > 0:
            q_gauss = dist.Independent(dist.Normal(
                q_pred['gauss_mean'], q_pred['gauss_std']), reinterpreted_batch_ndims=1)
            q_gauss_loss = self.gaussian_loss(
                q_gauss.rsample(), c[:, self.hparams.categ_code_dim:])
        mi_loss = q_categ_loss + q_gauss_loss

        return adv_loss + self.hparams.lambda_coeff*mi_loss

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--betas_opt", default=(0.5, 0.999), type=tuple)
        parser.add_argument("--feature_maps_gen", default=64, type=int)
        parser.add_argument("--feature_maps_disc",default=64, type=int)
        parser.add_argument("--fc_layers_gen", default=[1024,], type=list)
        parser.add_argument("--fc_layers_disc", default=[1024,], type=list)
        parser.add_argument("--img_channels", default=1, type=int)
        parser.add_argument("--noise_dim", default=62, type=int)
        parser.add_argument("--categ_code_dim", default=10, type=int)
        parser.add_argument("--cont_code_dim", default=2, type=int)
        parser.add_argument("--conv_start_shape_gen", default=[128, 7, 7], type=list)
        parser.add_argument("--conv_end_shape_disc", default=[128, 5, 5], type=list)
        parser.add_argument("--learning_rate_gen", default=1e-3, type=float)
        parser.add_argument("--learning_rate_disc", default=2e-4, type=float)
        return parser


def cli_main(args=None):
    pl.seed_everything(1234)

    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--data_dir", default="./", type=str)
    parser.add_argument("--num_workers", default=1, type=int)

    script_args, _ = parser.parse_known_args(args)

    transforms = transform_lib.Compose([
        transform_lib.Resize(28),
        transform_lib.ToTensor(),
        transform_lib.Normalize((0.5, ), (0.5, )),
    ])
    dataset = MNIST(root=script_args.data_dir,
                    download=True, transform=transforms)

    dataloader = DataLoader(dataset, batch_size=script_args.batch_size, shuffle=True, num_workers=script_args.num_workers)

    parser = InfoGAN.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args(args)

    model = InfoGAN(**vars(args))
    callbacks = []
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    cli_main()
