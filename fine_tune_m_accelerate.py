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
from torch.utils.data.distributed import DistributedSampler

from transformers import get_scheduler

from encoders import BioZorroCollator

from yacs.config import CfgNode as CN
from collections import defaultdict
import wandb 
from datetime import datetime
from accelerate import Accelerator

accelerator = Accelerator(log_with="wandb")

config = CN()
config.restart = 'training_output_21_31_23_10_2023' 
config.epochs = 1
config.batch_size = 4
config.num_warmup_steps = 3000
config.lr_scheduler_type = "cosine"
config.lr = 1e-4
config.output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
config.dataset = "/shared/fcaa53cd-ba57-4bfe-af9c-eaa958f95c1a_combined_all_veloc_sparse"
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
#lm_datasets = lm_datasets.rename_column('total_index','expression_index')
#lm_datasets = lm_datasets.rename_column('total_data','expression_data')
if config.model ==3:
    keep = ['expression_index','expression_counts','spliced_index', 'unspliced_index', 'spliced_counts', 'unspliced_counts']
    from multizorromodel import BioZorro
elif config.model ==2:
    keep = ['spliced_index', 'unspliced_index', 'spliced_counts', 'unspliced_counts']
    from biozorromodel import BioZorro
elif config.model ==1:
    keep = ['expression_index', 'expression_counts']
    from unizorromodel import BioZorro
else:
    raise Exception()

remove = list()
for key in lm_datasets.features.keys():
    if key not in keep:
        remove.append(key)
lm_datasets = lm_datasets.remove_columns(remove)
lm_datasets = lm_datasets.train_test_split(0.1, seed=config.ds_seed)

#BioZorro Collator
default_data_collator = BioZorroCollator(pad_len=1024, pad_token=0)

#### MODEL
model_config = {
    "dim": config.hidden_size, #hidden size
    "depth": config.layers, #layers
#    "spliced_input_dim": config.hidden_size, #embedding_size
#    "unspliced_input_dim": config.hidden_size,
#    "expression_input_dim": config.hidden_size,
    "dim_head": config.dim_head, #don't know, head hidden size?
    "heads": config.heads, #num heads
    "ff_mult": config.ff_mult, #Feed forward multiplier
    "num_fusion_tokens": config.num_fusion_tokens,
}
print(model_config)

model = BioZorro(**model_config) #.to(device)

config.n_params_emb, config.n_params_nonemb = count_parameters(model, print_summary=False)
print(f"Number of embedding parameters: {config.n_params_emb/10**6}M")
print(f"Number of non-embedding parameters: {config.n_params_nonemb/10**6}M")
print(f"Number of training samples: {len(lm_datasets['train'])}")


##Wandb causes an exception running in distributed mode
#    if xm.is_master_ordinal(local=False):
#wandb.init(
#        entity="josiahbjorgaard",
#        project="Multimodal",
#        config=dict(config),
#    )

# Initialise your wandb run, passing wandb parameters and any config information
accelerator.init_trackers(
    project_name="Multimodal",
    config=dict(config),
    init_kwargs={"wandb": {"entity": "josiahbjorgaard"}}
    )

## Create a subsed of data sampler, for parallelizing the training across multiple cores
#if world_size > 1:
#    train_sampler = DistributedSampler(
#        lm_datasets["train"],
#        num_replicas=world_size,
        #rank=xm.get_ordinal(), #Default to get from current distributed group
#        shuffle=True,
#    )
#else:
#    train_sampler = None
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
[print(f"{k}:{v.shape}") for k,v in sample.items()]
print(f"Number of batches: {len(train_dl)}")

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

model, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
     model, optimizer, train_dl, eval_dl, lr_scheduler
     )

if config.restart:
    logger.info(f"Loading saved state from {config.restart}")
    accelerator.load_state(config.restart)
    if config.reset_lr:
        for param_group in optimizer.param_groups:
            param_group['lr'] = config.reset_lr

## Start model training and defining the training loop

model.train()
device=accelerator.device
world_size=torch.cuda.device_count()
for epoch in range(config.epochs):
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        optimizer.zero_grad()
        loss = outputs.loss
        #losses = outputs.losses["losses"]
        #print(losses)
        #loss = losses.contrastive_loss + losses.fusion_loss_spliced + losses.fusion_loss_unspliced
        #loss.backward()
        accelerator.backward(loss)
        ## xm.optimizer_step is performing the sum of all the gradients updates done in the different Cores
#           xm.optimizer_step(optimizer)
        optimizer.step()
        if not config.reset_lr:
            lr_scheduler.step()
        if accelerator.is_main_process:
            progress_bar.update(world_size)
        accelerator.log({"total_loss":loss.detach().to("cpu")})
        accelerator.log({k:v.detach().to("cpu") for k,v in outputs.losses.items()})
        accelerator.log({"param_norm":get_param_norm(model).to("cpu"),"grad_norm":get_grad_norm(model).to("cpu")})
        accelerator.log({"lr":optimizer.param_groups[0]['lr']})
    #outputs.losses.contrastive_loss + outputs.losses.fusion_loss_spliced + outputs.losses.fusion_loss_unspliced
    #if xm.is_master_ordinal(local=False):
    #wandb.log({"epoch_loss":loss.detach().to("cpu")})
    #logger.info("Epoch {}, rank {}, Loss {:0.4f}".format(epoch, xm.get_ordinal(), loss.detach().to("cpu")))
    #Evaluation
    accelerator.save_state(config.output_dir)
    model.eval()
    with torch.no_grad():
        epoch_loss = 0.0
        losses = defaultdict(lambda: torch.Tensor([0.0]).to("cpu"))
        for i, batch in enumerate(tqdm(eval_dl)):
            outputs = model(**batch)
            loss = outputs.loss
            for k,v in outputs.losses.items():
                losses[k]+=v.detach().to("cpu")
                losses["total_loss"]+=loss.detach().to("cpu")
            accelerator.log({"val_step_total_loss":loss.to("cpu")})
            accelerator.log({"val_step_"+k:v.detach().to("cpu") for k,v in outputs.losses.items()})
        accelerator.log({'val_epoch_'+k:v/len(eval_dl) for k,v in losses.items()})
logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
os.makedirs(config.output_dir, exist_ok=True)
with open(os.path.join(config.output_dir,'config.yaml'),'w') as f:
    with redirect_stdout(f): print(config.dump())
with open(os.path.join(config.output_dir,'model_config.json'),'w') as f:
    json.dump(model_config, f)
accelerator.save_model(model, config.output_dir, safe_serialization=True)
accelerator.end_training()
## Using XLA for saving model after training for being sure only one copy of the model is saved
#os.makedirs(config.output_dir, exist_ok=True)
#checkpoint = {"state_dict": model.state_dict()}
#xm.save(checkpoint, f"{config.output_dir}/checkpoint.pt")
