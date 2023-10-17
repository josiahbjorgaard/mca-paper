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

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

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
config.epochs = 4
config.batch_size = 16
config.num_warmup_steps = 1000
config.lr_scheduler_type = "cosine"
config.lr = 1e-4
config.output_dir = "test_output1"
config.hidden_size = 512
config.layers = 5
config.dim_head = 64  # don't know, head hidden size?
config.heads = 4  # num heads
config.ff_mult = 4  # Feed forward multiplier
config.num_fusion_tokens = 16
config.dataset = "/efs-private/multimodal/data/filtered_protein_mrna_genes"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(0)

## define xla as device for using AWS Trainium Neuron Cores
device = "cuda"

#torch.distributed.init_process_group(device)
torch.distributed.init_process_group()
rank = torch.distributed.get_rank()
print(f"Start running basic DDP example on rank {rank}.")
# create model and move it to GPU with id rank
device = rank % torch.cuda.device_count()

# Get the global number of workes.
#world_size = xm.xrt_world_size()
world_size = torch.distributed.get_world_size()
logger.info("Workers: {}".format(world_size))
logger.info("Device: {}".format(device))

lm_datasets = load_from_disk(config.dataset).with_format('torch')

keep = ['spliced_index', 'unspliced_index', 'spliced_data', 'unspliced_data']
remove = list()
for key in lm_datasets.features.keys():
    if key not in keep:
        remove.append(key)
lm_datasets = lm_datasets.remove_columns(remove)
lm_datasets = lm_datasets.train_test_split(0.1)

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

model = BioZorro(**model_config).to(device)

config.n_params = count_parameters(model)
print(f"Number of parameters: {config.n_params/10**6}M")
print(f"Number of training samples: {len(lm_datasets['train'])}")


##Wandb causes an exception running in distributed mode
#    if xm.is_master_ordinal(local=False):
wandb.init(
        entity="josiahbjorgaard",
        project="Multimodal",
        config=dict(config),
    )

## Create a subsed of data sampler, for parallelizing the training across multiple cores
if world_size > 1:
    train_sampler = DistributedSampler(
        lm_datasets["train"],
        num_replicas=world_size,
        #rank=xm.get_ordinal(), #Default to get from current distributed group
        shuffle=True,
    )
else:
    train_sampler = None
## Creating a DataLoader object for iterating over it during the training epochs
train_dl = DataLoader(
    lm_datasets["train"],
    collate_fn=default_data_collator,
    batch_size=config.batch_size,
    sampler=train_sampler,
    shuffle=False if train_sampler else True)

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

progress_bar = tqdm(range(num_training_steps))

logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

## Start model training and defining the training loop
model.train()
for epoch in range(config.epochs):
    for batch in train_dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        optimizer.zero_grad()
        loss = outputs.loss
        loss.backward()
        ## xm.optimizer_step is performing the sum of all the gradients updates done in the different Cores
#           xm.optimizer_step(optimizer)
        optimizer.step()
        lr_scheduler.step()
        progress_bar.update(1)
        wandb.log({"loss":loss.detach().to("cpu")})
        wandb.log({k:v.detach().to("cpu") for k,v in outputs.losses.items()})
        wandb.log({"param_norm":get_param_norm(model).to("cpu"),"grad_norm":get_grad_norm(model).to("cpu")})
        wandb.log({"lr":optimizer.param_groups[0]['lr']})
    #outputs.losses.contrastive_loss + outputs.losses.fusion_loss_spliced + outputs.losses.fusion_loss_unspliced
    #if xm.is_master_ordinal(local=False):
    #wandb.log({"epoch_loss":loss.detach().to("cpu")})
    #logger.info("Epoch {}, rank {}, Loss {:0.4f}".format(epoch, xm.get_ordinal(), loss.detach().to("cpu")))
    #Evaluation
    model.eval()
    with torch.no_grad():
        epoch_loss = 0.0
        losses = defaultdict(torch.Tensor(0.0).to("cpu"))
        for i, batch in enumerate(tqdm(eval_dl)):
            outputs = model(**batch)
            loss = outputs.loss
            global_eval_step+=1
            epoch_loss += running_loss_reduced
            for k,v in outputs.losses.items():
                losses[k]+=v.detach().to("cpu")
            wandb.log({k:v.detach().to("cpu") for k,v in outputs.losses.items()})
        wandb.log({'epoch'+k:v/len(eval_dl) for k,v in losses.items()})
logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

## Using XLA for saving model after training for being sure only one copy of the model is saved
#os.makedirs(config.output_dir, exist_ok=True)
#checkpoint = {"state_dict": model.state_dict()}
#xm.save(checkpoint, f"{config.output_dir}/checkpoint.pt")
