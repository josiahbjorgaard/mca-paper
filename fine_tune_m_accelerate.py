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
#from multizorromodel import BioZorro
from multizorromodel import BioZorroWithLeaveOneOut as BioZorro
from safetensors.torch import load_model

from transformers import get_scheduler

from encoders import BioZorroCollatorWithTargets
from velocitymodel import VelocityModel

from yacs.config import CfgNode as CN
from collections import defaultdict
import wandb 
from datetime import datetime
from accelerate import Accelerator

from torchmetrics.functional.regression import pearson_corrcoef

from accelerate import DistributedDataParallelKwargs

config = CN()
#config.model_dir = 'training_output_02_39_30_10_2023' #Big one
#config.model_dir = 'training_output_23_45_06_11_2023'
config.model_dir = "training_output_01_30_07_11_2023"
#config.fit_indices = [ 286, 1037, 1519, 1752]
config.fit_indices = [209,265,286,347,369,503,749,1037,1047,
                        1410,1519,1524,1751,1752,1753,1754,1756,
                        1758,1760,1767,1779,1781,1783,1784,1791,]
config.norm = [0.2,0.0]
config.decoder_num_layers = 0
config.decoder_hidden_size = 1024
config.final_hidden_state = False #True
config.layers_to_unfreeze = [
        'return_tokens'
        'attn_pool.norm.gamma',
        'attn_pool.to_q.weight',
        'attn_pool.to_kv.weight',
        'attn_pool.to_out.weight'
            ]
config.load_checkpoint=False
config.epochs = 4
config.batch_size = 8
config.vocab_size = 36602
config.num_warmup_steps = 3000
config.lr_scheduler_type = "cosine"
config.lr = 1e-4
config.output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
config.dataset = "/efs-private/single_cell_data/out_veloc_data_unfiltered"
config.gene_indices = []
config.ds_frac = 1 
config.ds_seed = 42
config.model = 3
config.output_type = ["fusion", "fusion_floom_spliced", "fusion_floom_unspliced", "fusion_floom_expression"]
config.find_unused_parameters = False
config.pad_length = 1024

ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=config.find_unused_parameters)
accelerator = Accelerator(log_with="wandb", kwargs_handlers=[ddp_kwargs])

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
default_data_collator = BioZorroCollatorWithTargets(pad_len=config.pad_length, pad_token=0, target_ids = config.fit_indices, norm=config.norm)

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
if config.fit_indices:
    output_size = len(config.fit_indices)
else:
    output_size = 2000 #36601 #total vocab size
print(model_config)
backbone_hidden_size = model_config['dim'] #config.pad_length*3+model_config['num_fusion_tokens']
model = VelocityModel(model, decoder_hidden_size=config.decoder_hidden_size, decoder_num_layers=config.decoder_num_layers,
                        final_hidden_state=config.final_hidden_state,layers_to_unfreeze=config.layers_to_unfreeze,
                        backbone_hidden_size = backbone_hidden_size, output_types=config.output_type, 
                        output_size = output_size)
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
