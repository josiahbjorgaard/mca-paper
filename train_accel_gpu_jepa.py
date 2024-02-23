import logging
import os
import sys
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_scheduler
from collections import defaultdict
import random
from model import MFDOOM
from encoders import MultimodalCollator
from utils.training import get_param_norm, get_grad_norm, count_parameters, copy_batch, move_to
from utils.config import training_config, get_model_config
from utils.dataset import setup_data
from utils.metrics import Alignment, Uniformity
from accelerate import Accelerator
from torch_ema import ExponentialMovingAverage
torch.autograd.set_detect_anomaly(True)
def zero_modes(batch, modes_to_zero):
    """
    Mask out modalities in a batch - needs custom changes for each dataset
    """
    for mode in modes_to_zero:
        if mode == 'aud':
            batch[mode]['values'] = torch.ones_like(batch[mode]['values'])*-10000
        elif mode == 'vid':
            batch[mode]['attention_mask']=torch.ones_like(batch[mode]['attention_mask'])
        else:
            batch[mode]['attention_mask']=torch.ones_like(batch[mode]['attention_mask'])
    return batch

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
default_data_collator = MultimodalCollator(config.modality_config)
model_config = get_model_config(config)
device = accelerator.device

# Metrics config
metrics_alignment = defaultdict(Alignment) #{k: Alignment() for k in config.modality_config.keys()}
metrics_uniformity = defaultdict(Uniformity) #{k: Uniformity() for k in config.modality_config.keys()}

# Model
model = MFDOOM(**model_config)

config.n_params_emb, config.n_params_nonemb = count_parameters(model, print_summary=False)

# Initialise your wandb run, passing wandb parameters and any config information
init_kwargs={"wandb": {"entity": "josiahbjorgaard"}}
if config.wandb_restart:
    init_kwargs["wandb"]["id"]=config.wandb_restart
    init_kwargs["wandb"]["resume"]="must"
accelerator.init_trackers(
    project_name="MFDOOM_Paper_CMU",
    config=dict(config),
    init_kwargs=init_kwargs
    )

# Creating a DataLoader object for iterating over it during the training epochs
train_dl = DataLoader( datasets["train"], collate_fn=default_data_collator, batch_size=config.batch_size, shuffle=True, num_workers=8, prefetch_factor=16)
eval_dl = DataLoader( datasets["test"], collate_fn=default_data_collator, batch_size=config.batch_size)

accelerator.print(f"Number of embedding parameters: {config.n_params_emb/10**6}M")
accelerator.print(f"Number of non-embedding parameters: {config.n_params_nonemb/10**6}M")
accelerator.print(f"Number of training samples: {len(datasets['train'])}")
accelerator.print(f"Number of training batches per epoch: {len(train_dl)}")

num_training_steps = config.epochs * len(train_dl)

optimizer = AdamW(model.parameters(), lr=config.lr) # * world_size)
lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

if accelerator.is_main_process:
    progress_bar = tqdm(range(num_training_steps), initial = config.start_epoch * len(train_dl))

logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

loss_function = nn.L1Loss() #ContrastiveLossWithTemperature()

predictor_ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

model, optimizer, train_dl, eval_dl, lr_scheduler, loss_function, predictor_ema = accelerator.prepare(
     model, optimizer, train_dl, eval_dl, lr_scheduler, loss_function, predictor_ema
     )

if config.restart:
    logger.info(f"Loading saved state from {config.restart}")
    accelerator.load_state(config.restart)
    #if config.reset_lr:
    #    for param_group in optimizer.param_groups:
    #        param_group['lr'] = config.reset_lr

# Start model training and defining the training loop

model.train()
world_size = torch.cuda.device_count()
for epoch in range(config.start_epoch,config.epochs):
    for idb, batch in tqdm(enumerate(train_dl)):
        # Training
        #In JEPA we'll randomly mask one or more of the modalities to the predictor
        mods = list(batch.keys())
        losses = {}
        output = model(batch, no_loss=True)
        batch_copy = copy_batch(batch)
        mask = [x for x in mods if x not in [random.choice(mods)]] #The modalities to keep
        a_batch = move_to(zero_modes(batch_copy, mask), device)
        with predictor_ema.average_parameters():
            predictor_output = model(a_batch, no_loss=True)
        losses['jepa_l1'] = loss_function(output[k1], predictor_output[k2]) #, mask = mask)
        loss = sum(list(losses.values()))
        optimizer.zero_grad()
        accelerator.backward(loss)
        if config.clip:
            accelerator.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()
        lr_scheduler.step()
        predictor_ema.update()

        # Log and checkpoint
        if idb % config.n_step_checkpoint == 0:
            accelerator.save_state(config.output_dir)
        if accelerator.is_main_process:
            progress_bar.update(world_size)
        accelerator.log({"total_loss": loss.detach().to("cpu")})
        #accelerator.log({k: v.detach().to("cpu") for k,v in outputs['losses'].items() if '|' not in k})
        accelerator.log({"param_norm": get_param_norm(model).to("cpu"),
                         "grad_norm": get_grad_norm(model).to("cpu")})
        accelerator.log({"lr": optimizer.param_groups[0]['lr']})

    #Epoch end log and checkpoint
    os.makedirs(os.path.join(config.output_dir, str(epoch)), exist_ok=True)
    accelerator.save_state(os.path.join(config.output_dir, str(epoch)))
    #Eval looop
    if config.run_eval_loop:
        model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            losses = defaultdict(lambda: torch.Tensor([0.0]).to("cpu"))
            for i, batch in enumerate(tqdm(eval_dl)):
                # Training
                # In JEPA we'll randomly mask one or more of the modalities to the predictor
                outputs = dict()
                masks = dict()
                mods = list(batch.keys())
                losses = {}
                output = model(batch, no_loss=True)
                batch_copy = copy_batch(batch)
                mask = [x for x in mods if x not in [random.choice(mods)]]  # The modalities to keep
                a_batch = move_to(zero_modes(batch_copy, mask), device)
                with predictor_ema.average_parameters():
                    predictor_output = model(a_batch, no_loss=True)
                losses['jepa_l1'] = loss_function(output[k1], predictor_output[k2])  # , mask = mask)
                loss = sum(list(losses.values()))

                # Embedding space metrics
                for k in metrics_uniformity.keys():
                    metrics_uniformity[k].update(output[k]) #.detach().to('cpu'))
                    for k2 in metrics_alignment.keys():
                        metrics_alignment[k].update(output[k],#.detach().to('cpu'),
                                                predictor_output[k]) #.detach().to('cpu'))

                #Step Log
                losses["total_loss"] += loss.detach().to("cpu")
                accelerator.log({"val_step_total_loss":loss.to("cpu")})
                accelerator.log({"val_step_"+k: v.detach().to("cpu") for k, v in outputs['losses'].items() if '|' not in k})
            #Epoch Log
            accelerator.log({'val_epoch_'+k: v/len(eval_dl) for k, v in losses.items() if '|' not in k})
            uniformity = {'val_epoch_uniformity_'+k: v.compute() for k, v in metrics_uniformity.items()}
            accelerator.log(uniformity)
            alignment = {'val_epoch_alignment_'+k: v.compute() for k, v in metrics_alignment.items()}
            accelerator.log(alignment)
            accelerator.log({'val_epoch_unformity_avg': torch.mean(torch.stack(list(uniformity.values())))})
            accelerator.log({'val_epoch_alignment_avg': torch.mean(torch.stack(list(alignment.values())))})
            uniformity = {'val_epoch_norm_uniformity_'+k: v.compute(norm=True) for k, v in metrics_uniformity.items()}
            accelerator.log(uniformity)
            alignment = {'val_epoch_norm_alignment_'+k: v.compute(norm=True) for k, v in metrics_alignment.items()}
            accelerator.log(alignment)
            accelerator.log({'val_epoch_norm_unformity_avg': torch.mean(torch.stack(list(uniformity.values())))})
            accelerator.log({'val_epoch_norm_alignment_avg': torch.mean(torch.stack(list(alignment.values())))})

            for v in metrics_uniformity.values():
                v.reset()
            for v in metrics_alignment.values():
                v.reset()

logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

accelerator.save_model(model, config.output_dir, safe_serialization=True)
accelerator.end_training()
