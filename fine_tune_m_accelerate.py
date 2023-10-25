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
from safetensors.torch import load_model

from transformers import get_scheduler

from encoders import BioZorroCollatorWithTargets

from yacs.config import CfgNode as CN
from collections import defaultdict
import wandb 
from datetime import datetime
from accelerate import Accelerator

accelerator = Accelerator(log_with="wandb")

config = CN()
config.model_dir = 'training_output_21_31_23_10_2023'
config.epochs = 1
config.batch_size = 4
config.num_warmup_steps = 3000
config.lr_scheduler_type = "cosine"
config.lr = 1e-5
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
        'unspliced_index', 'spliced_data', 'unspliced_data'
        'velocity_index', 'velocity_data']
remove = list()
for key in lm_datasets.features.keys():
    if key not in keep:
        remove.append(key)
lm_datasets = lm_datasets.remove_columns(remove)
lm_datasets = lm_datasets.train_test_split(0.1, seed=config.ds_seed)


#BioZorro Collator
default_data_collator = BioZorroCollatorWithTargets(pad_len=1024, pad_token=0)

#### MODEL
with open(os.path.join(config.model_dir,'model_config.json'),'r') as f:
    model_config = json.load(f)
print(model_config)

model = BioZorro(**model_config) #.to(device)
load_model(model, os.path.join(config.model_dir, 'model.safetensors'))

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

logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

## Start model training and defining the training loop
class CustomModel(nn.Module):
    def __init__(self, model, size):
        super().__init__()
        self.model = model
        self.linear = nn.Linear(-1, size)
        self.loss = nn.MSELoss()
    def forward(self, batch):
        output = self.model(**{k:v for k,v in batch if 'velocity' not in k})
        logits = torch.cat([output.expression_output, output.spliced_output, output.unspliced_output, output.fusion_output])
        logits = self.linear(logits)
        return self.loss(logits, batch['velocity'])

model = CustomModel(model, 36601)
model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
     model, optimizer, train_dl, eval_dl, lr_scheduler
     )

model.train()
device=accelerator.device
world_size=torch.cuda.device_count()
for epoch in range(config.epochs):
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        loss = model(**batch)
        optimizer.zero_grad()
        accelerator.backward(loss)
        ## xm.optimizer_step is performing the sum of all the gradients updates done in the different Cores
#           xm.optimizer_step(optimizer)
        optimizer.step()
        if not config.reset_lr:
            lr_scheduler.step()
        if accelerator.is_main_process:
            progress_bar.update(world_size)
        accelerator.log({"loss":loss.detach().to("cpu")})
        accelerator.log({"lr":optimizer.param_groups[0]['lr']})
    #Evaluation
    accelerator.save_state(config.output_dir)
    model.eval()
    with torch.no_grad():
        epoch_loss = 0.0
        for i, batch in enumerate(tqdm(eval_dl)):
            loss = model(**batch)
            accelerator.log({"val_step_total_loss":loss.to("cpu")})
logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
os.makedirs(config.output_dir, exist_ok=True)
with open(os.path.join(config.output_dir,'config.yaml'),'w') as f:
    with redirect_stdout(f): print(config.dump())
with open(os.path.join(config.output_dir,'model_config.json'),'w') as f:
    json.dump(model_config, f)
accelerator.save_model(model, config.output_dir, safe_serialization=True)
accelerator.end_training()
