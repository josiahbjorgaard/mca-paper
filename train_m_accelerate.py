from datasets import load_from_disk
from contextlib import redirect_stdout
import json
import logging
import os
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch import nn
#import torch_xla.core.xla_model as xm
#import torch_xla.distributed.parallel_loader as pl
#import torch_xla.distributed.xla_backend
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import get_scheduler

#from multizorromodel import BioZorro
from encoders import BioZorroCollator

from yacs.config import CfgNode as CN
from collections import defaultdict
import wandb 
from datetime import datetime
from accelerate import Accelerator

accelerator = Accelerator(log_with="wandb")

def count_parameters(model,print_summary=False):
    n_param_embedding = 0
    n_param_nonembedding = 0
    for n,p in model.named_parameters():
        if p.requires_grad:
            if print_summary:
                print(f"{n}:{p.numel()/10**6}M")
            if 'embedding' in n:
                n_param_embedding+=p.numel()
            else:
                n_param_nonembedding+=p.numel()
    #return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return n_param_embedding, n_param_nonembedding

def get_param_norm(model,norm_type=2.0):
    norm_type = float(norm_type)
    parameters = model.parameters()
    local_norm = torch.DoubleTensor([0.0]).to(next(iter(parameters)).device)
    grads_for_norm = []
    for param in parameters:
        param_norm = torch.norm(param.detach(), norm_type)
        local_norm += param_norm ** norm_type
    total_norm = local_norm**(1.0 / norm_type)
    return total_norm

def get_grad_norm(model,norm_type=2.0):
    norm_type = float(norm_type)
    parameters = model.parameters()
    local_norm = torch.FloatTensor([float(0.0)]).to(next(iter(parameters)).device)
    grads_for_norm = []
    for param in parameters:
        grad_not_none = param.grad is not None
        if grad_not_none:
            grad = param.grad.detach()
            grad_norm = torch.norm(grad, norm_type)
            local_norm += grad_norm ** norm_type
    total_norm = local_norm**(1.0 / norm_type)
    return total_norm


config = CN()
config.restart = False #'training_output_21_31_23_10_2023' 
config.epochs = 3
config.batch_size = 2
config.num_warmup_steps = 3000
config.lr_scheduler_type = "cosine"
config.lr = 1e-4
config.output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
config.hidden_size = 512
config.layers = 10
config.dim_head = 64 #64  # heads*dim_head = intermeidate size?
config.heads = 8  # num heads
config.ff_mult = 4  # Feed forward multiplier
config.num_fusion_tokens = 16
config.dataset = "/shared/dataset3M" #"/shared/fcaa53cd-ba57-4bfe-af9c-eaa958f95c1a_combined_all"
config.ds_frac = 1 
config.ds_seed = 42
config.model = 3
config.n_step_checkpoint = 20000
#If config.restart, will reset all config items to checkpoint yaml
if config.restart:
    # Allow creating new keys recursively.
    config.set_new_allowed(True)
    config.merge_from_file(os.path.join(config.restart, 'config.yaml'))
    config.epochs = 1 ### WILL NEED TO SPECIFY NUMBER OF EPOCHS TO CONTINUE WITH HERE
    ### New Output directory!!
    config.output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
    config.reset_lr = 0.0001

"""
# Create conf   
config = CN()
# Allow creating new keys recursively.
config.set_new_allowed(True)
config.merge_from_file(filename)
"""

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(0)

## define xla as device for using AWS Trainium Neuron Cores

#torch.distributed.init_process_group(device)
#torch.distributed.init_process_group()
#rank = torch.distributed.get_rank()
# create model and move it to GPU with id rank
#device = rank % torch.cuda.device_count()

# Get the global number of workes.
#world_size = xm.xrt_world_size()
#world_size = torch.distributed.get_world_size()
#logger.info("Workers: {}".format(world_size))
#logger.info("Device: {}".format(device))

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
for val in keep:
    if 'counts' in val:
        lm_datasets = lm_datasets.rename_column(val, val.split('_')[0]+'_data')
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
    project_name="Multimodal2",
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

os.makedirs(config.output_dir, exist_ok=True)
with open(os.path.join(config.output_dir,'config.yaml'),'w') as f:
    with redirect_stdout(f): print(config.dump())
with open(os.path.join(config.output_dir,'model_config.json'),'w') as f:
    json.dump(model_config, f)

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
    for idb, batch in enumerate(train_dl):
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
        lr_scheduler.step()
        if idb % config.n_step_checkpoint == 0:
            accelerator.save_state(config.output_dir)
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
    os.makedirs(os.path.join(config.output_dir,str(epoch)), exist_ok=True)
    accelerator.save_state(os.path.join(config.output_dir, str(epoch)))
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
accelerator.save_model(model, config.output_dir, safe_serialization=True)
accelerator.end_training()
## Using XLA for saving model after training for being sure only one copy of the model is saved
#os.makedirs(config.output_dir, exist_ok=True)
#checkpoint = {"state_dict": model.state_dict()}
#xm.save(checkpoint, f"{config.output_dir}/checkpoint.pt")
