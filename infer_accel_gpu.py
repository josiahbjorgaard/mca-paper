import logging
import os
import sys
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from collections import defaultdict

from model import MFDOOM
from encoders import MultimodalCollator
from utils.training import get_param_norm, get_grad_norm, count_parameters, move_to
from utils.config import training_config, get_model_config
from utils.dataset import setup_data
from utils.metrics import Alignment, Uniformity

from accelerate import Accelerator

accelerator = Accelerator(log_with="wandb")

config = training_config(sys.argv[1])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(0)


datasets = setup_data(config.dataset,
                      split=config.split,
                      ds_frac=config.ds_frac,
                      ds_seed=config.ds_seed,
                      predrop=config.predrop,
                      predrop_config=config.modality_config)

# Collator
default_data_collator = MultimodalCollator(config.modality_config, labels=config.label_col)
model_config = get_model_config(config)
device = accelerator.device

# Metrics config
metrics_alignment = {k: Alignment() for k in config.modality_config.keys()}
metrics_uniformity = {k: Uniformity() for k in config.modality_config.keys()}
metrics_uniformity['fusion'] = Uniformity() #add fusion token

# Model
model = MFDOOM(**model_config)

config.n_params_emb, config.n_params_nonemb = count_parameters(model, print_summary=False)

# Initialise your wandb run, passing wandb parameters and any config information
init_kwargs={"wandb": {"entity": "josiahbjorgaard"}}
if config.wandb_restart:
    init_kwargs["wandb"]["id"]=config.wandb_restart
    init_kwargs["wandb"]["resume"]="must"
accelerator.init_trackers(
    project_name="MFDOOM_Paper_Inference",
    config=dict(config),
    init_kwargs=init_kwargs
    )

# Creating a DataLoader object for iterating over it during the training epochs
# Both are for test here since we generate inferences on all data
train_dl = DataLoader( datasets["train"], collate_fn=default_data_collator, batch_size=config.batch_size, drop_last=True, shuffle=False)
eval_dl = DataLoader( datasets["test"], collate_fn=default_data_collator, batch_size=config.batch_size, drop_last=True, shuffle=False)

accelerator.print(f"Number of embedding parameters: {config.n_params_emb/10**6}M")
accelerator.print(f"Number of non-embedding parameters: {config.n_params_nonemb/10**6}M")
accelerator.print(f"Number of training samples: {len(datasets['train'])}")
accelerator.print(f"Number of training batches per epoch: {len(train_dl)}")

num_training_steps = len(train_dl)

if accelerator.is_main_process:
    progress_bar = tqdm(range(num_training_steps))

logger.info("Start training set inference: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

#model, train_dl, eval_dl = accelerator.prepare(
#     model, train_dl, eval_dl
#     )
model = accelerator.prepare(model)

assert config.restart
logger.info(f"Loading saved state from {config.restart}")
accelerator.load_state(config.restart)

model.eval()
world_size = torch.cuda.device_count()
assert world_size == 1
with torch.no_grad():
    for tv,dl in {'train': train_dl, 'eval': eval_dl}.items():
        embeddings = defaultdict(list)
        masks = defaultdict(list)
        labels = list()
        for idb, batch in tqdm(enumerate(dl)):
            # Training
            batch_labels = batch['Labels']
            batch = move_to(batch, device)
            outputs = model(batch)
            loss = outputs.pop('loss')
            losses = outputs.pop('losses')
            modality_masks = outputs.pop('modality_sample_mask')
            for k,v in outputs.items():
                embeddings[k].append(v.detach().cpu())
            for k,v in modality_masks.items():
                masks[k].append(v.detach().cpu())
            labels.append(batch_labels['data'].detach().cpu())
                # Embedding space metrics
            """
            for k in metrics_uniformity.keys():
                if k != 'fusion':
                   sample_mask = modality_masks[k]
                   metrics_uniformity[k].update(outputs[k][sample_mask]) #.detach().to('cpu'))
                else:
                   metrics_uniformity[k].update(outputs[k]) #.detach().to('cpu'))
            for k in metrics_alignment.keys():
                sample_mask = modality_masks[k]
                metrics_alignment[k].update(outputs[k][sample_mask],#.detach().to('cpu'),
                                                outputs['fusion'][sample_mask]) #.detach().to('cpu'))
            """
            accelerator.log({"total_loss": loss.detach().to("cpu")})
            accelerator.log({k: v.detach().to("cpu") for k,v in losses.items() if '|' not in k})
            #Eval looop
        masks = {k:torch.cat(v, dim=0) for k,v in masks.items()}
        torch.save(masks, f"{config.output_dir}/{tv}_masks.pt")
        embeddings = {k:torch.cat(v, dim = 0) for k,v in embeddings.items()}
        torch.save(embeddings, f"{config.output_dir}/{tv}_embeddings.pt")
        labels = torch.cat(labels, dim = 0)
        torch.save(labels, f"{config.output_dir}/{tv}_labels.pt")
        """
        #Epoch Log
        uniformity = {'uniformity_'+k: v.compute() for k, v in metrics_uniformity.items()}
        accelerator.log(uniformity)
        alignment = {'alignment_'+k: v.compute() for k, v in metrics_alignment.items()}
        accelerator.log(alignment)
        accelerator.log({'unformity_avg': torch.mean(torch.stack(list(uniformity.values())))})
        accelerator.log({'alignment_avg': torch.mean(torch.stack(list(alignment.values())))})
        uniformity = {'norm_uniformity_'+k: v.compute(norm=True) for k, v in metrics_uniformity.items()}
        accelerator.log(uniformity)
        alignment = {'norm_alignment_'+k: v.compute(norm=True) for k, v in metrics_alignment.items()}
        accelerator.log(alignment)
        accelerator.log({'norm_unformity_avg': torch.mean(torch.stack(list(uniformity.values())))})
        accelerator.log({'norm_alignment_avg': torch.mean(torch.stack(list(alignment.values())))})
        for v in metrics_uniformity.values():
            v.reset()
        for v in metrics_alignment.values():
            v.reset()
        """
logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

accelerator.end_training()
