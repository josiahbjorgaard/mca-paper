import logging
import os
import sys
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import get_scheduler

from multizorromodel import BioZorro
from encoders import BioZorroCollator
from training_utils import get_param_norm, get_grad_norm, count_parameters
from config_utils import training_config, get_model_config
from dataset_utils import setup_data

import wandb

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########
# XLA
########
device = "xla"

torch.distributed.init_process_group(device)

# Get the global number of workes.
world_size = xm.xrt_world_size()
logger.info("Workers: {}".format(world_size))
logger.info("Device: {}".format(device))
#########

config = training_config(sys.argv[1])

torch.manual_seed(0)

datasets = setup_data(config.dataset,
                      split=config.split,
                      ds_frac=config.ds_frac,
                      ds_seed=config.ds_seed)

# BioZorro Collator
default_data_collator = BioZorroCollator(pad_len=config.pad_len, pad_token=0)
model_config = get_model_config(config)
model = BioZorro(**model_config)

config.n_params_emb, config.n_params_nonemb = count_parameters(model, print_summary=False)

# Initialise your wandb run, passing wandb parameters and any config information
##Wandb causes an exception running in distributed mode
if xm.is_master_ordinal(local=False):
    wandb.init(
        entity="josiahbjorgaard",
        project="Multimodal2",
    )

## Create a subsed of data sampler, for parallelizing the training across multiple cores
if world_size > 1:
    train_sampler = DistributedSampler(
        datasets["train"],
        num_replicas=world_size,
        rank=xm.get_ordinal(),
        shuffle=True,
    )

## Creating a DataLoader object for iterating over it during the training epochs
train_dl = DataLoader(
    datasets["train"],
    collate_fn=default_data_collator,
    batch_size=config.batch_size,
    sampler=train_sampler,
    shuffle=False if train_sampler else True)

## Loading a subset of the data in the different Neuron Cores provided as input
train_device_loader = pl.MpDeviceLoader(train_dl, device)

xm.master_print(f"Number of embedding parameters: {config.n_params_emb/10**6}M")
xm.master_print(f"Number of non-embedding parameters: {config.n_params_nonemb/10**6}M")
xm.master_print(f"Number of training samples: {len(datasets['train'])}")
xm.master_print(f"Number of training batches per epoch: {len(train_dl)}")

num_training_steps = config.epochs * len(train_dl)

optimizer = AdamW(model.parameters(), lr=config.lr)
lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

if xm.is_master_ordinal(local=False): #.is_main_process:
    progress_bar = tqdm(range(num_training_steps))

logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

if config.restart:
    raise Exception("Checkpoint reload not implemented for trn1")
    logger.info(f"Loading saved state from {config.restart}")
    if config.reset_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.reset_lr

# Start model training and defining the training loop
model.to(device)
model.train()
world_size = torch.cuda.device_count()
for epoch in range(config.epochs):
    for idb, batch in enumerate(train_dl):
        # Training
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        optimizer.zero_grad()
        loss = outputs.loss
        loss.backward()
        xm.optimizer_step(optimizer)
        lr_scheduler.step()
        #xm.add_step_closure(training_metrics_closure, (epoch, global_step, loss.detach(), optimizer.param_groups[0]['lr'],grad_norm, param_norm),run_async=False) #no data dependency with next mark_step

    # Log and checkpoint
        if xm.is_master_ordinal(local=False):
            #print(outputs)
            #xm.rendezvous("Saving Checkpoint")
            progress_bar.update(world_size)
            #wandb.log({"total_loss": loss.detach().to("cpu")})
            #wandb.log({k: v.detach().to("cpu") for k,v in outputs.losses.items()})
            #wandb.log({"param_norm": get_param_norm(model).to("cpu"),
            #                "grad_norm": get_grad_norm(model).to("cpu")})
            #wandb.log({"lr": optimizer.param_groups[0]['lr']})

## Using XLA for saving model after training for being sure only one copy of the model is saved
os.makedirs(config.output_dir, exist_ok=True)
checkpoint = {"state_dict": model.state_dict()}
xm.rendezvous("Saving Checkpoint")
xm.save(checkpoint, f"{config.output_dir}/checkpoint.pt")

logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
