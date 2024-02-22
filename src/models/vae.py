import torch
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
from typing import Any, Dict, Tuple
from torch import nn
import torch.nn.functional as F
from torchvision.utils import make_grid

import wandb

class VAE(LightningModule):
    def __init__(
        self,
        img_size: int,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        latent_dim: int = 128,
        kl_weight: float = 1e-4,
        compile: bool = False, # TODO: I am too lazy to implement this
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)



        self.encoder = nn.Sequential(
            nn.Linear(img_size ** 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.mean_linear = nn.Linear(32, latent_dim)
        self.var_linear  = nn.Linear(32, latent_dim)

        self.latent_dim = latent_dim

        # create decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 128),
            nn.ReLU(),
            nn.Linear(128, img_size ** 2),
            nn.Sigmoid(),
        )


        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_metric_best = MaxMetric()
        
        # loss function
        def loss_function(x, x_hat, mean, log_var, kl_weight=kl_weight):
            recon_loss = F.mse_loss(x_hat, x)
            kl_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var), 1), 0)
            loss = recon_loss + kl_loss * kl_weight
            return loss, recon_loss, kl_loss
        
        self.loss_function = loss_function
        

    def forward(self, x):
        x = x.reshape(-1, 784)
        encoded = self.encoder(x)
        encoded = torch.flatten(encoded, start_dim=1)
        mean = self.mean_linear(encoded)
        log_var = self.var_linear(encoded)
        eps = torch.randn_like(log_var)
        std = torch.exp(0.5 * log_var)
        z = mean + eps * std
        decoded = self.decoder(z)

        return decoded, mean, log_var
    
    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_metric_best.reset()

    def sample(self):
        # TODO
        z = torch.randn(32, self.latent_dim).to(self.device)
        decoded = self.decoder(z)
        decoded = decoded.reshape(-1, 1, 28, 28)
        return decoded
    
    def model_step(
        self, 
        batch: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, _ = batch
        decoded, mean, log_var = self.forward(x)
        loss, recon_loss, kl_loss = self.loss_function(x.reshape(-1, 784), decoded, mean, log_var)
        return loss, recon_loss, kl_loss, decoded
    
    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ):
        loss, recon_loss, kl_loss, decoded = self.model_step(batch)

        
        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=True)

        # visualize reconstruction results
        if batch_idx == 0:
            image_init = batch[0].reshape(-1, 1, 28, 28) 
            grid_init = make_grid(image_init)

            image_recon = decoded.reshape(-1, 1, 28, 28)
            grid_recon = make_grid(image_recon)

            self.logger.experiment.log({
                "train/init": [wandb.Image(grid_init.cpu())],
                "train/recon": [wandb.Image(grid_recon.cpu())]
            })
        
        

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss, recon_loss, kl_loss, decoded = self.model_step(batch)

        # update and log loss
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=True)

        # log reconstruction visualization results
        if batch_idx == 0:
            image_init = batch[0].reshape(-1, 1, 28, 28) 
            grid_init = make_grid(image_init)

            image_recon = decoded.reshape(-1, 1, 28, 28)
            grid_recon = make_grid(image_recon)

            image_sample = self.sample().reshape(-1, 1, 28, 28)
            grid_sample = make_grid(image_sample)

            self.logger.experiment.log({
                "valid/init": [wandb.Image(grid_init.cpu())],
                "valid/recon": [wandb.Image(grid_recon.cpu())],
                "valid/sample": [wandb.Image(grid_sample.cpu())]
            })

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        loss, recon_loss, kl_loss, decoded = self.model_step(batch)


        # update and log loss
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/recon_loss", recon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/kl_loss", kl_loss, on_step=False, on_epoch=True, prog_bar=True)

        

        # log reconstruction visualization results
        if batch_idx == 0:
            image_init = batch[0].reshape(-1, 1, 28, 28) 
            grid_init = make_grid(image_init)

            image_recon = decoded.reshape(-1, 1, 28, 28)
            grid_recon = make_grid(image_recon)

            image_sample = self.sample().reshape(-1, 1, 28, 28)
            grid_sample = make_grid(image_sample)

            self.logger.experiment.log({
                "test/init": [wandb.Image(grid_init.cpu())],
                "test/recon": [wandb.Image(grid_recon.cpu())],
                "test/sample": [wandb.Image(grid_sample.cpu())]
            })




    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                }
            }
        return {"optimizer": optimizer}
    
    def on_train_epoch_end(self) -> None:
        pass
    
    def on_test_epoch_end(self) -> None:
        pass

    def setup(self, stage: str) -> None:
        pass # TODO: compile if we need