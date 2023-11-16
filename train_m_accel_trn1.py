from datasets import load_from_disk

import logging
import os
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch import nn
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_backend
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedModel
from transformers import AutoConfig, AutoModel

from biozorromodel import BioZorro
from encoders import BioZorroCollator

from yacs.config import CfgNode as CN

import wandb 

config = CN()
config.epochs = 100
config.freeze_layers=6
config.batch_size = 2
config.warmup_steps = 1000
config.lr = 1e-4
config.dropout = 0.3
config.output_dir = "test_output1"
config.num_labels = 1
config.hidden_size = 512
config.dataset = "/efs-private/multimodal/data/filtered_protein_mrna_genes"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(0)

## define xla as device for using AWS Trainium Neuron Cores
device = "xla"

torch.distributed.init_process_group(device)

# Get the global number of workes.
world_size = xm.xrt_world_size()
logger.info("Workers: {}".format(world_size))
logger.info("Device: {}".format(device))

lm_datasets = load_from_disk(config.dataset).with_format('torch')
keep =  ['spliced_index', 'unspliced_index', 'spliced_data', 'unspliced_data']
remove = list()
for key in lm_datasets.features.keys():
    if key not in keep:
        remove.append(key)
lm_datasets = lm_datasets.remove_columns(remove)
lm_datasets = lm_datasets.train_test_split(0.1)

#BioZorro Collator
default_data_collator = BioZorroCollator(pad_len=2048, pad_token=0)


if __name__ == '__main__':
    path = os.path.abspath("data")
    
    ##Wandb causes an exception running in distributed mode
    if xm.is_master_ordinal(local=False):
        wandb.init(
            entity="josiahbjorgaard",
            project="Multimodal-Trn1",

        )
    
    ## Create a subsed of data sampler, for parallelizing the training across multiple cores
    if world_size > 1:
        train_sampler = DistributedSampler(
            lm_datasets["train"],
            num_replicas=world_size,
            rank=xm.get_ordinal(),
            shuffle=True,
        )

    ## Creating a DataLoader object for iterating over it during the training epochs
    train_dl = DataLoader(
        lm_datasets["train"],
        collate_fn=default_data_collator,
        batch_size=config.batch_size,
        sampler=train_sampler,
        shuffle=False if train_sampler else True)

    ## Loading a subset of the data in the different Neuron Cores provided as input
    train_device_loader = pl.MpDeviceLoader(train_dl, device)



    #### MODEL
    model_config = {
        "dim": 512, #hidden size
        "depth": 7, #layers
        "spliced_input_dim": 512, #embedding_size
        "unspliced_input_dim": 512,
        "dim_head":64, #don't know, head hidden size?
        "heads": 8, #num heads
        "ff_mult": 4, #Feed forward multiplier
        "num_fusion_tokens": 16,
    }

    model = BioZorro(**model_config).to(device)

    ## Haven't yet tried freezing layers
    #freeze_layers = config.freeze_layers
    #if freeze_layers is not None:
    #    modules_to_freeze = model.backbone.encoder.layer[:freeze_layers]
    #    for module in modules_to_freeze:
    #        for param in module.parameters():
    #            param.requires_grad = False

    current_timestamp = strftime("%Y-%m-%d-%H-%M", gmtime())

    optimizer = AdamW(model.parameters(), lr=config.lr * world_size)

    num_training_steps = config.epochs * len(train_dl)
    progress_bar = tqdm(range(num_training_steps))

    logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    ## Start model training and defining the training loop
    model.train()
    for epoch in range(config.epochs):
        for batch in train_device_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            optimizer.zero_grad()
            loss = outputs.loss
            loss.backward()
            ## xm.optimizer_step is performing the sum of all the gradients updates done in the different Cores
            xm.optimizer_step(optimizer)
            progress_bar.update(1)

        if xm.is_master_ordinal(local=False):
            wandb.log({"epoch_loss": loss.detach().to("cpu")})
        logger.info("Epoch {}, rank {}, Loss {:0.4f}".format(epoch, xm.get_ordinal(), loss.detach().to("cpu")))

    logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

    ## Using XLA for saving model after training for being sure only one copy of the model is saved
    os.makedirs(config.output_dir, exist_ok=True)
    checkpoint = {"state_dict": model.state_dict()}
    xm.save(checkpoint, f"{config.output_dir}/checkpoint.pt")
