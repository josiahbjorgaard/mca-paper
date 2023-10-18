from datasets import load_from_disk

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

from biozorromodel import BioZorro
from encoders import BioZorroCollator

from yacs.config import CfgNode as CN
from collections import defaultdict
import wandb 

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
config.epochs = 1
config.batch_size = 2
config.num_warmup_steps = 3000
config.lr_scheduler_type = "cosine"
config.lr = 1e-4
config.output_dir = "test_output6"
config.hidden_size = 512
config.layers = 10
config.dim_head = 64  # don't know, head hidden size?
config.heads = 8  # num heads
config.ff_mult = 4  # Feed forward multiplier
config.num_fusion_tokens = 16
config.dataset = "/efs-private/multimodal/data/filtered_protein_mrna_genes"
config.ds_frac = 0.1 
config.ds_seed = 42
config.spliced_only = True 

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

keep = ['spliced_index', 'unspliced_index', 'spliced_data', 'unspliced_data']
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
    "spliced_input_dim": config.hidden_size, #embedding_size
    "unspliced_input_dim": config.hidden_size,
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
## Start model training and defining the training loop
model.train()
device=accelerator.device
world_size=torch.cuda.device_count()
for epoch in range(config.epochs):
    for batch in train_dl:
        if config.spliced_only:
            batch['unspliced_data']=batch['spliced_data']
            batch['unspliced_index']=batch['spliced_index']
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
        if accelerator.is_main_process:
            progress_bar.update(world_size)
        accelerator.log({k:v.detach().to("cpu") for k,v in outputs.losses.items()})
        accelerator.log({"param_norm":get_param_norm(model).to("cpu"),"grad_norm":get_grad_norm(model).to("cpu")})
        accelerator.log({"lr":optimizer.param_groups[0]['lr']})
    #outputs.losses.contrastive_loss + outputs.losses.fusion_loss_spliced + outputs.losses.fusion_loss_unspliced
    #if xm.is_master_ordinal(local=False):
    #wandb.log({"epoch_loss":loss.detach().to("cpu")})
    #logger.info("Epoch {}, rank {}, Loss {:0.4f}".format(epoch, xm.get_ordinal(), loss.detach().to("cpu")))
    #Evaluation
    model.eval()
    with torch.no_grad():
        epoch_loss = 0.0
        losses = defaultdict(lambda: torch.Tensor([0.0]).to("cpu"))
        for i, batch in enumerate(tqdm(eval_dl)):
            if config.spliced_only:
                batch['unspliced_data']=batch['spliced_data']
                batch['unspliced_index']=batch['spliced_index']
            outputs = model(**batch)
            loss = outputs.loss
            for k,v in outputs.losses.items():
                losses[k]+=v.detach().to("cpu")
            accelerator.log({"val_step_"+k:v.detach().to("cpu") for k,v in outputs.losses.items()})
        accelerator.log({'val_epoch_'+k:v/len(eval_dl) for k,v in losses.items()})
logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
accelerator.end_training()
## Using XLA for saving model after training for being sure only one copy of the model is saved
#os.makedirs(config.output_dir, exist_ok=True)
#checkpoint = {"state_dict": model.state_dict()}
#xm.save(checkpoint, f"{config.output_dir}/checkpoint.pt")
