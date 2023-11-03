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

from multizorromodel import BioZorro
from encoders import BioZorroCollator
from training_utils import get_param_norm, get_grad_norm, count_parameters
from config_utils import training_config, get_model_config
from dataset_utils import setup_data

from accelerate import Accelerator

accelerator = Accelerator(log_with="wandb")

config = training_config(sys.argv[1])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(0)

datasets = setup_data(config.dataset,
                      split=config.split,
                      ds_frac=config.ds_frac,
                      ds_seed=config.ds_seed)

# BioZorro Collator
default_data_collator = BioZorroCollator(pad_len=config.pad_len, pad_token=0)
model_config = get_model_config(config)
model = BioZorro(**model_config)

config.n_params_emb, config.n_params_nonemb = count_parameters(model, print_summary=False)

# Initialise your wandb run, passing wandb parameters and any config information
accelerator.init_trackers(
    project_name="Multimodal2",
    config=dict(config),
    init_kwargs={"wandb": {"entity": "josiahbjorgaard"}}
    )

# Creating a DataLoader object for iterating over it during the training epochs
train_dl = DataLoader( datasets["train"], collate_fn=default_data_collator, batch_size=config.batch_size, shuffle=True)
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

# Start model training and defining the training loop

model.train()
device = accelerator.device
world_size = torch.cuda.device_count()
for epoch in range(config.epochs):
    for idb, batch in enumerate(train_dl):
        # Training
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        optimizer.zero_grad()
        loss = outputs.loss
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()

        # Log and checkpoint
        if idb % config.n_step_checkpoint == 0:
            accelerator.save_state(config.output_dir)
        if accelerator.is_main_process:
            progress_bar.update(world_size)
        accelerator.log({"total_loss": loss.detach().to("cpu")})
        accelerator.log({k: v.detach().to("cpu") for k,v in outputs.losses.items()})
        accelerator.log({"param_norm": get_param_norm(model).to("cpu"),
                         "grad_norm": get_grad_norm(model).to("cpu")})
        accelerator.log({"lr": optimizer.param_groups[0]['lr']})

    #Epoch end log and checkpoint
    os.makedirs(os.path.join(config.output_dir,str(epoch)), exist_ok=True)
    accelerator.save_state(os.path.join(config.output_dir, str(epoch)))

    #Eval looop
    if config.run_eval_loop:
        model.eval()
        with torch.no_grad():
            epoch_loss = 0.0
            losses = defaultdict(lambda: torch.Tensor([0.0]).to("cpu"))
            for i, batch in enumerate(tqdm(eval_dl)):
                outputs = model(**batch)
                loss = outputs.loss
                for k, v in outputs.losses.items():
                    losses[k] += v.detach().to("cpu")
                    losses["total_loss"] += loss.detach().to("cpu")
                accelerator.log({"val_step_total_loss":loss.to("cpu")})
                accelerator.log({"val_step_"+k: v.detach().to("cpu") for k, v in outputs.losses.items()})
            accelerator.log({'val_epoch_'+k: v/len(eval_dl) for k, v in losses.items()})

logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))

accelerator.save_model(model, config.output_dir, safe_serialization=True)
accelerator.end_training()
