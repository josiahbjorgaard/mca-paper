import os
import torch
import lightning.pytorch as pl
from biozorromodel import BioZorro, TokenTypes as T
from mudataloader import MuDataModule #get_dataloader
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from transformers.optimization import get_cosine_schedule_with_warmup
from typing import Tuple
from lightning.pytorch.loggers import WandbLogger

def get_optimizers_for_lightning(
    model: torch.nn.Module,
    learning_rate: float,
    adam_eps: float,
    adam_weight_decay: float,
    adam_betas: Tuple[int, int],
    warmup_steps: int,
    max_steps: int,
):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        betas=adam_betas,
        eps=adam_eps,
        weight_decay=adam_weight_decay,
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max_steps,
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class BioZorroLightningModule(pl.LightningModule):
    def __init__(
        self,
        learning_rate: float = 0.0002,
        adam_eps: float = 1.0e-08,
        adam_weight_decay: float = 0.01,
        adam_betas: Tuple[int, int] = (0.9, 0.999),
        warmup_steps: int = 2000,
        max_steps: int = 450000,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.adam_betas = adam_betas
        self.adam_weight_decay = adam_weight_decay
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        self.model = BioZorro(
                        512, #dim,
                        6, #depth,
                        16381, #rna_input_dim,
                        96162, #atac_input_dim,
                        dim_head = 64,
                        heads = 8,
                        ff_mult = 4,
                        num_fusion_tokens = 16,
                        )
    def _step(self, batch, batch_idx):
        atac, rna, prot = batch
        output = self.model(rna=rna, atac=atac)
        return output

    def training_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        losses = output.losses
        total_loss = 0
        for key in losses:
            if losses[key] is not None:
                total_loss += losses[key]
                self.log(f"train/losses/{key}", losses[key], prog_bar=True, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        output = self._step(batch, batch_idx)
        losses = output.losses
        total_loss = 0
        for key in losses:
            if losses[key] is not None:
                total_loss += losses[key]
                self.log(
                    f"validation/losses/{key}", losses[key], prog_bar=True, logger=True
                )

        return total_loss

    def configure_optimizers(self):
        return get_optimizers_for_lightning(
            self.model,
            self.learning_rate,
            self.adam_eps,
            self.adam_weight_decay,
            self.adam_betas,
            self.warmup_steps,
            self.max_steps,
        )

def main():
    
    #seed_everything(config.training.seed, workers=True)

    datamodule = MuDataModule('/efs-private/st_perceiver/pbmc_w3_teaseq.h5mu')

    model = BioZorroLightningModule()

    callbacks = [
        LearningRateMonitor(logging_interval="step"),
    #    MultimodalEvalCallback(imagenet_datamodule=imagenet_datamodule),
    ]

    #if config.training.lightning_checkpoint is not None:
    #    callbacks.append(
    #        ModelCheckpoint(
    #            **OmegaConf.to_container(config.training.lightning_checkpoint)
    #        )
    #    )

    wandb_logger = WandbLogger()

    trainer = pl.Trainer(
        accelerator="auto", devices="auto",
    #    **OmegaConf.to_container(config.training.lightning),
        callbacks=callbacks,
        logger=wandb_logger
    )
    #ckpt_path = config.training.lightning_load_from_checkpoint
    
    trainer.fit(model, datamodule=datamodule) #, ckpt_path=ckpt_path)
    trainer.validate(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
