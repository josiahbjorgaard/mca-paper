import logging
import os
import sys
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
import transformers
from transformers import get_scheduler

from multizorromodel_trn1 import BioZorro
from encoders import BioZorroCollator
from training_utils import get_param_norm, get_grad_norm, count_parameters
from config_utils import training_config, get_model_config
from dataset_utils import setup_data
import datasets
import wandb

import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

########
# XLA
########
torch.distributed.init_process_group('xla')
device = xm.xla_device()
rank = xm.get_ordinal()
world_size = xm.xrt_world_size()
accelerator_log_kwargs = {}
accelerator = Accelerator( **accelerator_log_kwargs)

# Get the global number of workes.
print("Workers: {}".format(world_size))
print("Device: {}".format(device))
#########

# Make one log on every process with the configuration for debugging.
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.info(accelerator.state, main_process_only=False)
if accelerator.is_local_main_process:
    datasets.utils.logging.set_verbosity_warning()
    transformers.utils.logging.set_verbosity_info()
else:
    datasets.utils.logging.set_verbosity_error()
    transformers.utils.logging.set_verbosity_error()
config_file = 'config.yaml'
config = training_config(config_file)
# If passed along, set the training seed now.
if config.seed is not None:
    set_seed(config.seed, device_specific=False)

datasets = setup_data(config.dataset,
                      split=config.split,
                      ds_frac=config.ds_frac,
                      ds_seed=config.ds_seed)

# BioZorro Collator
default_data_collator = BioZorroCollator(pad_len=config.pad_len, pad_token=0, attn_mask=True)
model_config = get_model_config(config)
model = BioZorro(**model_config)

if xm.is_master_ordinal(local=False):
    config.n_params_emb, \
    config.n_params_nonemb = count_parameters(model, print_summary=False)

# Initialise your wandb run, passing wandb parameters and any config information
##Wandb causes an exception running in distributed mode
if xm.is_master_ordinal(local=False):
    wandb.init(
        entity="josiahbjorgaard",
        config=config,
        project="Multimodal2",
    )

datasets = setup_data(config.dataset,
                      split=config.split,
                      ds_frac=config.ds_frac,
                      ds_seed=config.ds_seed)

train_dataset = datasets['train']
print(train_dataset)
## Creating a DataLoader object for iterating over it during the training epochs
train_dataloader = DataLoader(
                        train_dataset, shuffle=True, collate_fn=default_data_collator,
                        batch_size=config.batch_size,
                        prefetch_factor=4, num_workers=2
                )

optimizer = AdamW(model.parameters(), lr=config.lr)

print(next(iter(train_dataloader)))
# Prepare everything with our `accelerator`.
model, optimizer, train_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader
)
if xm.is_master_ordinal(local=False):
    xm.master_print(f"Number of embedding parameters: {config.n_params_emb/10**6}M")
    xm.master_print(f"Number of non-embedding parameters: {config.n_params_nonemb/10**6}M")
    xm.master_print(f"Number of training samples: {len(datasets['train'])}")
    xm.master_print(f"Number of training batches per epoch: {len(train_dataloader)}")

num_training_steps = config.epochs * len(train_dataloader)

lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

#if accelerator.distributed_type == DistributedType.TPU:
#    model.tie_weights()

#if xm.is_master_ordinal(local=False): #.is_main_process:
progress_bar = tqdm(range(num_training_steps))

logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

if config.restart:
    raise Exception("Checkpoint reload not implemented for trn1")
    logger.info(f"Loading saved state from {config.restart}")
    if config.reset_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.reset_lr

# Start model training and defining the training loop
for epoch in range(config.epochs):
    model.train()
    for idb, batch in enumerate(train_dataloader):
        outputs = model(**batch)
        optimizer.zero_grad()
        loss = outputs.loss
        loss.backward()
        optimizer.optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        xm.mark_step()
        #xm.add_step_closure(training_metrics_closure, (epoch, global_step, loss.detach(), optimizer.param_groups[0]['lr'],grad_norm, param_norm),run_async=False) #no data dependency with next mark_step

    # Log and checkpoint
        #if xm.is_master_ordinal(local=False):
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
