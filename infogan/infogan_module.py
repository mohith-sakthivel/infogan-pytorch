from torch.utils import data
import yaml
from argparse import ArgumentParser
from typing import Any, List, Tuple

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as dist
from torch.utils.data import DataLoader, RandomSampler

from torchvision import transforms as transform_lib
from torchvision.datasets import MNIST, ImageFolder, SVHN

from infogan.components import InfoGANGenerator, InfoGANDiscriminator, QNetwork
from infogan.utils import TensorboardGenerativeModelImageSampler, CodeLatentDimInterpolator


class InfoGAN(pl.LightningModule):
    """
    InfoGAN Implementation
    """

    def __init__(self,
                 lambda_coeff: float = 1,
                 betas_opt: List[float] = [0.5, 0.999],
                 feature_maps_gen: List = [64, ],
                 feature_maps_disc: List = [64, 128],
                 fc_layers_gen: List = [1024, ],
                 fc_layers_disc: List = [1024, ],
                 img_channels: int = 1,
                 noise_dim: int = 62,
                 categ_code_dims: list = [10, ],
                 cont_code_dim: int = 2,
                 conv_start_shape_gen: List = [128, 7, 7],
                 conv_end_shape_disc: List = [128, 5, 5],
                 hinge_loss: bool = False,
                 learning_rate_gen: float = 1e-3,
                 learning_rate_disc: float = 2e-4,
                 num_gen_opts: int = 1,
                 **kwargs: Any,
                 ) -> None:
        """
        Args:
            lambda_coeff         - Weight of the MI term in the objective fn
            beats_opt            - Beta values for Adam optimizer
            feature_maps_gen     - Feature map size for each deconv layer in the generator
            feature_maps_disc    - Feature map size for each conv layer in the discriminator
            fc_layers_gen        - Fully connected layer dimensions prior to deconv blocks in the generator
            fc_layers_disc       - Fully connected layer dimensions after the conv blocks in the discriminator
            img_channels         - Number of channels in the image
            noise_dim            - Dimension of random noise variables
            categ_code_dims      - A list with the number of categories in each categorical distribution of the 
                                   latent code variable
            cont_code_dim        - Number of variables in the gaussina code
            conv_start_shape_gen - Shape of the input to the deconv block in the generator (for batch size=1)
            conv_end_shape_disc  - Shape of the output form the conv_layers in the discriminator (for batch size=1)
            hinge_loss           - Use Hinge loss instead of the standard GAN loss
            learning_rate_gen    - learning rate for the generator
            learning_rate_disc   - learning rate for the discriminator
        """

        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.generator = InfoGANGenerator(noise_dim,
                                          sum(categ_code_dims)+cont_code_dim,
                                          feature_maps_gen,
                                          img_channels,
                                          conv_start_shape=conv_start_shape_gen,
                                          fc_layers=fc_layers_gen)
        self.generator.apply(self._initialize_weights)

        self.discriminator = InfoGANDiscriminator(sum(categ_code_dims),
                                                  cont_code_dim,
                                                  feature_maps_disc,
                                                  img_channels,
                                                  conv_end_shape_disc,
                                                  fc_layers=fc_layers_disc,
                                                  sn=self.hparams.hinge_loss)

        self.q_network = QNetwork(self.discriminator.base_feat_dim,
                                  sum(categ_code_dims),
                                  cont_code_dim)

        self.q_network.apply(self._initialize_weights)
        if not hinge_loss:
            self.discriminator.apply(self._initialize_weights)

        self._initialize_samplers()

        if not hinge_loss:
            self.adverserial_loss = nn.BCEWithLogitsLoss()
        if len(categ_code_dims) > 0:
            self.categorical_loss = nn.CrossEntropyLoss()
        if cont_code_dim > 0:
            self.gaussian_loss = nn.MSELoss()

    def _initialize_samplers(self) -> None:
        if len(self.hparams.categ_code_dims) > 0:
            self.categ_dists = [dist.OneHotCategorical(logits=torch.ones(c_dim))
                                for c_dim in self.hparams.categ_code_dims]
        if self.hparams.cont_code_dim > 0:
            self._normal_dist = dist.MultivariateNormal(torch.zeros(self.hparams.cont_code_dim),
                                                        torch.eye(self.hparams.cont_code_dim))

    @staticmethod
    def _initialize_weights(module) -> None:
        classname = module.__class__.__name__
        if classname.find("Conv") != -1 or classname.find("Linear") != -1:
            torch.nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif classname.find("BatchNorm") != -1:
            torch.nn.init.normal_(module.weight, 1.0, 0.02)
            torch.nn.init.zeros_(module.bias)

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List]:
        gen_opt = optim.Adam(self.generator.parameters(),
                             lr=self.hparams.learning_rate_gen,
                             betas=self.hparams.betas_opt)
        disc_opt = optim.Adam([*self.discriminator.parameters(), *self.q_network.parameters()],
                              lr=self.hparams.learning_rate_disc,
                              betas=self.hparams.betas_opt)
        return ([gen_opt, disc_opt], [])

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Generates an image
        Args:
            z - Noise vector
            c - Latent code
        """
        return self.generator(z, c)

    def training_step(self,
                      batch: Tuple,
                      batch_idx: int) -> torch.Tensor:
        gen_opt, disc_opt = self.optimizers(use_pl_optimizer=True)
        real_img, _ = batch

        gen_loss = torch.zeros([], device=self.device)
        for i in range(self.hparams.num_gen_opts):
            gen_loss_iter = self._get_gen_loss(len(real_img))
            gen_opt.zero_grad()
            self.manual_backward(gen_loss_iter)
            gen_opt.step()
            gen_loss += gen_loss_iter
        gen_loss /= self.hparams.num_gen_opts
        self.log("gen_train/loss", gen_loss, on_epoch=True)

        disc_loss = self._get_disc_loss(real_img)
        disc_opt.zero_grad()
        self.manual_backward(disc_loss)
        disc_opt.step()
        self.log("disc_train/loss", disc_loss, on_epoch=True)
        return {"gen_loss": gen_loss.detach(), "disc_loss": disc_loss.detach()}

    def sample_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn((batch_size, self.hparams.noise_dim), device=self.device)

    def sample_code(self, batch_size: int) -> torch.Tensor:
        cat_code = cont_code = None
        if len(self.hparams.categ_code_dims) > 0:
            cat_codes = [categ_dist.sample([batch_size])
                         for categ_dist in self.categ_dists]
            cat_code = torch.cat(cat_codes, dim=-1)
        if self.hparams.cont_code_dim > 0:
            cont_code = self._normal_dist.sample([batch_size])
        return torch.cat([code for code in [cat_code, cont_code]
                          if code is not None], dim=-1).to(self.device)

    def _get_latents(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.sample_noise(batch_size)
        c = self.sample_code(batch_size)
        return z, c

    def _get_gen_loss(self, batch_size: int) -> torch.Tensor:
        # Calculate adverserial loss
        z, c = self._get_latents(batch_size)
        fake_img = self.generator(z, c)
        fake_pred, disc_latents = self.discriminator(fake_img)
        q_pred = self.q_network(disc_latents)
        if self.hparams.hinge_loss:
            adv_loss = -fake_pred.mean()
        else:
            target = torch.ones_like(fake_pred)
            adv_loss = self.adverserial_loss(fake_pred, target)

        q_categ_loss, q_gauss_loss = (torch.zeros([], device=self.device),
                                      torch.zeros([], device=self.device))
        # Calculate loss from categorical latent code prediction
        start_dim = 0
        if len(self.hparams.categ_code_dims) > 0:
            for c_dim in self.hparams.categ_code_dims:
                end_dim = start_dim + c_dim
                categ_posterior = self.categorical_loss(q_pred['categ'][:, start_dim:end_dim],
                                                        c[:, start_dim:end_dim].argmax(dim=-1))
                categ_prior = - \
                    torch.log(torch.ones_like(categ_posterior) / c_dim)
                q_categ_loss -= (categ_prior - categ_posterior)
                start_dim += c_dim
            q_categ_loss = q_categ_loss/len(self.hparams.categ_code_dims)
        # Calculate loss from gaussian latent code prediction
        if self.hparams.cont_code_dim > 0:
            q_gauss = dist.Independent(dist.Normal(
                q_pred['gauss_mean'], q_pred['gauss_std']), reinterpreted_batch_ndims=1)
            q_gauss_loss = self.gaussian_loss(
                q_gauss.rsample(), c[:, start_dim:])
        mi_loss = q_categ_loss + q_gauss_loss

        self.log("gen_train/adv_loss", adv_loss, on_epoch=False)
        self.log("gen_train/categ_info_loss", q_categ_loss, on_epoch=False)
        self.log("gen_train/gauss_info_loss", q_gauss_loss, on_epoch=False)

        return adv_loss + self.hparams.lambda_coeff * mi_loss

    def _get_disc_loss(self, real_img: torch.Tensor) -> torch.Tensor:
        # Calculate adverserial loss from real images
        real_pred = self.discriminator(real_img, need_base_feat=False)
        if self.hparams.hinge_loss:
            real_loss = F.relu(1-real_pred).mean()
        else:
            real_target = torch.ones_like(real_pred)
            real_loss = self.adverserial_loss(real_pred, real_target)
        # Calculate adverserial loss from fake images
        z, c = self._get_latents(len(real_img))
        fake_img = self.generator(z, c)
        fake_pred, disc_latents = self.discriminator(fake_img)
        q_pred = self.q_network(disc_latents.detach())
        if self.hparams.hinge_loss:
            fake_loss = F.relu(1+fake_pred).mean()
        else:
            fake_target = torch.zeros_like(fake_pred)
            fake_loss = self.adverserial_loss(fake_pred, fake_target)

        adv_loss = real_loss + fake_loss

        q_categ_loss, q_gauss_loss = (torch.zeros([], device=self.device),
                                      torch.zeros([], device=self.device))
        # Calculate loss from categorical latent code prediction
        start_dim = 0
        if len(self.hparams.categ_code_dims) > 0:
            for c_dim in self.hparams.categ_code_dims:
                end_dim = start_dim + c_dim
                categ_posterior = self.categorical_loss(q_pred['categ'][:, start_dim:end_dim],
                                                        c[:, start_dim:end_dim].argmax(dim=-1))
                categ_prior = - \
                    torch.log(torch.ones_like(categ_posterior) / c_dim)
                q_categ_loss -= (categ_prior - categ_posterior)
                start_dim += c_dim
            q_categ_loss = q_categ_loss/len(self.hparams.categ_code_dims)
        # Calculate loss from gaussian latent code prediction
        if self.hparams.cont_code_dim > 0:
            q_gauss = dist.Independent(dist.Normal(
                q_pred['gauss_mean'], q_pred['gauss_std']), reinterpreted_batch_ndims=1)
            q_gauss_loss = self.gaussian_loss(
                q_gauss.rsample(), c[:, start_dim:])
        mi_loss = q_categ_loss + q_gauss_loss

        self.log("disc_train/adv_loss", adv_loss, on_epoch=False)
        self.log("disc_train/categ_info_loss", q_categ_loss, on_epoch=False)
        self.log("disc_train/gauss_info_loss", q_gauss_loss, on_epoch=False)

        return adv_loss + self.hparams.lambda_coeff * mi_loss


def cli_main(args=None):
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1234)

    script_args, _ = parser.parse_known_args(args)
    pl.seed_everything(script_args.seed)

    if script_args.dataset == 'mnist':
        transforms = transform_lib.Compose([
            transform_lib.Resize((28, 28)),
            transform_lib.ToTensor(),
            transform_lib.Normalize((0.5,), (0.5,)),
        ])
        dataset = MNIST(root=script_args.data_dir,
                        download=True, transform=transforms)
    elif script_args.dataset in ['celeba', 'svhn']:
        transforms = transform_lib.Compose([
            transform_lib.Resize((32, 32)),
            transform_lib.ToTensor(),
            transform_lib.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        if script_args.dataset == 'celeba':
            dataset = ImageFolder(root=script_args.data_dir+script_args.dataset,
                                  transform=transforms)
        elif script_args.dataset == 'svhn':
            dataset = SVHN(root=script_args.data_dir,
                           download=True, transform=transforms)

    args = parser.parse_args(args)

    with open('configs/%s.yml' % (script_args.dataset), 'r') as cfg:
        config_args = yaml.safe_load(cfg)
    for k, v in config_args.items():
        args.__setattr__(k, v)

    num_batches = args.num_batches if args.num_batches is not None else len(
        dataset) // args.batch_size
    sampler = RandomSampler(dataset, True, num_batches * args.batch_size)

    dataloader = DataLoader(dataset, batch_size=script_args.batch_size,
                            sampler=sampler, num_workers=script_args.num_workers)

    model = InfoGAN(**vars(args))
    callbacks = [
        TensorboardGenerativeModelImageSampler(
            num_samples=5, log_epoch_interval=5),
        CodeLatentDimInterpolator(epoch_interval=10)
    ]
    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks)
    trainer.fit(model, dataloader)


if __name__ == "__main__":
    cli_main()
