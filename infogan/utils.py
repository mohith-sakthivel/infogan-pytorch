from typing import Optional, Tuple, List
import numpy as np

import torch
import torch.nn.functional as F

from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import _module_available


_TORCHVISION_AVAILABLE: bool = _module_available("torchvision")

if _TORCHVISION_AVAILABLE:
    import torchvision
else:
    raise "torch vision package is required."


class TensorboardGenerativeModelImageSampler(Callback):
    """
        Generates images and logs to tensorboard
        Credits: Modified from pytorch bolts
    """

    def __init__(
        self,
        num_samples: int = 3,
        nrow: int = 8,
        padding: int = 2,
        log_epoch_interval: int = 10,
        normalize: bool = True,
        norm_range: Optional[Tuple[int, int]] = (0, 1),
        scale_each: bool = False,
        pad_value: int = 0,
    ) -> None:
        """
        Args:
            num_samples - Number of images displayed in the grid
            nrow        - Number of images displayed in each row of the grid
            padding     - Amount of padding
            normalize   - Shift the image to the range (0, 1)
            norm_range  - Tuple (min, max) where min and max are numbers, then these numbers are 
                          used to normalize the image. By default, values are computed from the tensor.
            scale_each  - Scale each image in the batch of images separately
                          rather than the (min, max) over all images
            pad_value   - Value for the padded pixels
        """

        super().__init__()
        self.num_samples = num_samples
        self.nrow = nrow
        self.padding = padding
        self.log_epoch_interval = log_epoch_interval
        self.normalize = normalize
        self.norm_range = norm_range
        self.scale_each = scale_each
        self.pad_value = pad_value

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.log_epoch_interval == 0:

            # generate images
            with torch.no_grad():
                pl_module.eval()
                z = pl_module.sample_noise(self.num_samples)
                c = pl_module.sample_code(self.num_samples)
                images = pl_module(z, c)
                pl_module.train()

            if len(images.size()) == 2:
                img_dim = pl_module.img_dim
                images = images.view(self.num_samples, *img_dim)

            grid = torchvision.utils.make_grid(
                tensor=images,
                nrow=self.nrow,
                padding=self.padding,
                normalize=self.normalize,
                range=self.norm_range,
                scale_each=self.scale_each,
                pad_value=self.pad_value,
            )
            str_title = f"{pl_module.__class__.__name__}_images"
            trainer.logger.experiment.add_image(
                str_title, grid, global_step=trainer.global_step)


class CodeLatentDimInterpolator(Callback):
    """
    Interpolates the code latent space by varying the code variables for a given noise
    """

    def __init__(
        self,
        sigma_range: int = 4,
        num_samples: int = 5,
        epoch_interval: int = 20,
        normalize: bool = True,
    ):
        """
        Args:
            sigma_range    - Range of sigma value to use for continuous latent codes. 
            num_samples    - Number of samples to display
            epoch_interval - Number of epochs between each interpolation generation
            normalize      - Change image to (0, 1) range
        """

        super().__init__()
        self.sigma_range = sigma_range
        self.num_samples = num_samples
        self.epoch_interval = epoch_interval
        self.normalize = normalize

    def on_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if (trainer.current_epoch + 1) % self.epoch_interval == 0:
            grid = self.generate_latent_grid(pl_module)
            str_title = f"{pl_module.__class__.__name__}_code_latent_space"
            trainer.logger.experiment.add_image(
                str_title, grid, global_step=trainer.global_step)

    def generate_latent_grid(self, pl_module: LightningModule) -> List[torch.Tensor]:
        images = self.interpolate_latent_space(pl_module,
                                               noise_dim=pl_module.hparams.noise_dim,
                                               categ_code_dims=pl_module.hparams.categ_code_dims,
                                               cont_code_dim=pl_module.hparams.cont_code_dim)
        images = torch.cat(images, dim=0)
        return torchvision.utils.make_grid(images, nrow=self.num_samples, normalize=self.normalize)

    def interpolate_latent_space(self, pl_module: LightningModule, noise_dim: int,
                                 categ_code_dims: int, cont_code_dim: int) -> List[torch.Tensor]:
        images = []
        with torch.no_grad():
            pl_module.eval()
            z = pl_module.sample_noise(self.num_samples)
            c_fixed = pl_module.sample_code(self.num_samples)
            if len(categ_code_dims) > 0:
                start_dim = 0
                for categ_dim in categ_code_dims:
                    for categ in range(categ_dim):
                        c = F.one_hot(torch.tensor([categ], device=pl_module.device),
                                      categ_dim).repeat(self.num_samples, 1)
                        c_current = c_fixed.clone()
                        c_current[:, start_dim:start_dim+categ_dim] = c
                        img = pl_module(z, c_current)
                        images.append(img)
                    start_dim += categ_dim

            if cont_code_dim > 0:
                for c_var in range(cont_code_dim):
                    for sigma in np.linspace(-self.sigma_range, self.sigma_range, 10):
                        c = torch.zeros(
                            (self.num_samples, cont_code_dim), device=pl_module.device)
                        c[:, c_var] = sigma
                        c_current = c_fixed.clone()
                        c_current[:, start_dim:] = c
                        img = pl_module(z, c_current)
                        images.append(img)

        pl_module.train()
        return images
