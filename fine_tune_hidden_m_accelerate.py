from datasets import load_from_disk
from contextlib import redirect_stdout
import json
import logging
import os
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from multizorromodel import BioZorro

from transformers import get_scheduler

from encoders import BioZorroCollator
from multizorromodel import TokenTypes
from velocitymodel import VelocityHiddenModel

from yacs.config import CfgNode as CN
from datetime import datetime
from accelerate import Accelerator

from accelerate import DistributedDataParallelKwargs

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])

config = CN()
config.model_dir = 'training_output_22_47_25_10_2023'
config.input_size = 1024
#config.norm = [0.2, 0.5]
config.decoder_num_layers = 3
config.layers_to_unfreeze = None
config.load_checkpoint = True
config.epochs = 4
config.batch_size = 8
config.num_warmup_steps = 3000
config.lr_scheduler_type = "cosine"
config.lr = 1e-4
config.output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
config.dataset = "/shared/fcaa53cd-ba57-4bfe-af9c-eaa958f95c1a_combined_all_veloc_sparse"
config.gene_indices = []
config.ds_frac = 1 
config.ds_seed = 42
config.model = 3


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(0)

## Dataset processing
lm_datasets = load_from_disk(config.dataset).with_format('torch')
if config.ds_frac < 1.0:
    lm_datasets = lm_datasets.select(list(range(0,int(len(lm_datasets)*config.ds_frac))))
keep = ['expression_index','expression_data','spliced_index',
        'unspliced_index', 'spliced_data', 'unspliced_data',
        'velocity_index', 'velocity_data']
remove = list()
accelerator.print(lm_datasets)
for key in lm_datasets.features.keys():
    if key not in keep:
        remove.append(key)
lm_datasets = lm_datasets.remove_columns(remove)
lm_datasets = lm_datasets.train_test_split(0.1, seed=config.ds_seed)
accelerator.print(lm_datasets)

#BioZorro Collator
default_data_collator = BioZorroCollator(pad_len=512, pad_token=0)

#### MODEL
with open(os.path.join(config.model_dir,'model_config.json'),'r') as f:
    model_config = json.load(f)
accelerator.print(model_config)

model = BioZorro(**model_config) #.to(device)
if config.load_checkpoint:
    accelerator.print(f"Loading checkpoint from {config.model_dir}")
    #load_model(model, os.path.join(config.model_dir, 'model.safetensors'))
    checkpoint = torch.load(os.path.join(config.model_dir, 'pytorch_model.bin'))
    model.load_state_dict(checkpoint)

# Initialise your wandb run, passing wandb parameters and any config information
accelerator.init_trackers(
    project_name="Multimodal-Velocity",
    config=dict(config),
    init_kwargs={"wandb": {"entity": "josiahbjorgaard"}}
    )

## Creating a DataLoader object for iterating over it during the training epochs
train_dl = DataLoader(
    lm_datasets["train"],
    collate_fn=default_data_collator,
    batch_size=config.batch_size,
    shuffle = True)
#    sampler=train_sampler,
#    shuffle=False if train_sampler else True)

eval_dl = DataLoader(
        lm_datasets["test"],
        collate_fn=default_data_collator,
        batch_size=config.batch_size)

sample = next(iter(train_dl))

current_timestamp = strftime("%Y-%m-%d-%H-%M", gmtime())

num_training_steps = config.epochs * len(train_dl)

optimizer = AdamW(model.parameters(), lr=config.lr) # * world_size)

lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

if accelerator.is_main_process:
    progress_bar = tqdm(range(num_training_steps))


os.makedirs(config.output_dir, exist_ok=True)
with open(os.path.join(config.output_dir,'config.yaml'),'w') as f:
    with redirect_stdout(f): print(config.dump())
with open(os.path.join(config.output_dir,'model_config.json'),'w') as f:
    json.dump(model_config, f)

logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

## Start model training and defining the training loop
"""
VelocityHiddenModel init:
                 backbone_model,
                 output_size,
                 backbone_hidden_size = 256,
                 output_token_type = TokenTypes.GLOBAL,
                 layers_to_unfreeze=None,
                 dim_head = 64,
                 heads = 8,
                 decoder_num_layers=3,
                 dropout=0.1,
                 vocab_size=2000):
"""
model = VelocityHiddenModel(model,
                            output_size=config.input_size,
                            backbone_hidden_size=model_config['dim'],
                            output_token_type=TokenTypes.GLOBAL,
                            layers_to_unfreeze=config.layers_to_unfreeze,
                            dim_head=model_config['dim_head'],
                            heads=model_config['heads'],
                            decoder_num_layers=config.decoder_num_layers,
                            )
accelerator.print(model)
model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
     model, optimizer, train_dl, eval_dl, lr_scheduler
     )

model.train()
device=accelerator.device
world_size=torch.cuda.device_count()
for epoch in range(config.epochs):
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss, metrics = model(batch)
        optimizer.zero_grad()
        accelerator.backward(loss)
        ## xm.optimizer_step is performing the sum of all the gradients updates done in the different Cores
#           xm.optimizer_step(optimizer)
        optimizer.step()
        lr_scheduler.step()
        if accelerator.is_main_process:
            progress_bar.update(world_size)
        accelerator.log({"loss":loss.detach().to("cpu")})
        accelerator.log({"lr":optimizer.param_groups[0]['lr']})
        accelerator.log({k:v.detach().to("cpu") for k,v in metrics.items()})
    #Evaluation
    accelerator.save_state(config.output_dir)
    model.eval()
    with torch.no_grad():
        epoch_loss = 0.0
        for i, batch in enumerate(tqdm(eval_dl)):
            loss, metrics = model(batch)
            accelerator.log({"val_step_total_loss":loss.to("cpu")})
            accelerator.log({"val_step_"+k:v.detach().to("cpu") for k,v in metrics.items()})
logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
accelerator.save_model(model, config.output_dir, safe_serialization=True)
accelerator.end_training()
