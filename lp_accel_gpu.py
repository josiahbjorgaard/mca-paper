"""
Linear probe the Labels from the dataset
"""

import logging
import os
import sys
from time import gmtime, strftime
from tqdm.auto import tqdm
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from transformers import get_scheduler
from collections import defaultdict
from utils.training import get_param_norm, get_grad_norm, count_parameters, move_to
from utils.config import embedding_eval_config
from utils.metrics import Alignment, Uniformity, get_rank_metrics
from torchmetrics import PearsonCorrCoef, F1Score
from accelerate import Accelerator

class FineTuneDataset(Dataset):
    def __init__(self, embeddings, labels, key='fusion',index = 0,transform=None, target_transform=None):
        #self.embeddings = torch.cat([embeddings[k] for k in embeddings.keys()],dim=1)
        self.embeddings = embeddings[key]
        self.labels = labels[:,index] #First is a sentiment [-3,3] -3: negative 3:postive, next is 6 labels for emotions [0,3] where 3 is strong and 0 is no x and 3 is strong x
        print(f"{self.labels.shape = }")
        if index > 0:
            self.labels = self.labels/3.0
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

accelerator = Accelerator(log_with="wandb")

config = embedding_eval_config(sys.argv[1])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(config.seed)

device = accelerator.device

#TODO check if these exist, and tell user ot run inference first if not
e_train=torch.load(f'{config.embedding_dir}/train_embeddings.pt', map_location="cpu")
#m_train=torch.load(f'{config.embedding_dir}/train_masks.pt', map_location="cpu")
s_train=torch.load(f'{config.embedding_dir}/train_labels.pt', map_location="cpu").squeeze()
e_test=torch.load(f'{config.embedding_dir}/eval_embeddings.pt', map_location="cpu")
#m_test=torch.load(f'{config.embedding_dir}/eval_masks.pt', map_location="cpu")
s_test=torch.load(f'{config.embedding_dir}/eval_labels.pt', map_location="cpu").squeeze()

print(f"Shape of test labels: {s_test.shape}")
print(f"Shape of train labels: {s_train.shape}")

# Creating a DataLoader object for iterating over it during the training epochs
train_dl = DataLoader(FineTuneDataset(e_train, s_train, index=config.task), batch_size=config.batch_size, shuffle=True)
eval_dl = DataLoader(FineTuneDataset(e_test, s_test, index=config.task), batch_size=config.batch_size)

# Metrics config
metrics_alignment = Alignment()
metrics_uniformity = Uniformity()
if config.metric == "F1":
    met = F1Score().to(device)
elif config.metric == "PCC":
    met = PearsonCorrCoef().to(device)
else:
    raise Exception("Didn't recognize config.metric")

# Model
num_emb = next(iter(train_dl))[0].shape[1]
if config.model_type == 'linear':
    model = nn.Linear(num_emb, 1)
elif config.model_type == 'MLP':
    model = nn.Sequential(nn.Linear(num_emb, config.hidden_size),
                          nn.Dropout(config.dropout),
                          nn.ReLU(),
                          nn.Linear(config.hidden_size, 1))
else:
    raise Exception(f"Model type {config.model_type} not recognized")
if config.loss_type == "L1":
    loss_fn = nn.L1Loss().to(device)
elif config.loss_type == "MSE":
    loss_fn = nn.MSELoss().to(device)
elif config.loss_type == "BCE":
    loss_fn = nn.BCEWithLogitsLoss().to(device)

# Initialise your wandb run, passing wandb parameters and any config information
init_kwargs={"wandb": {"entity": "josiahbjorgaard", "name":config.wandb_job_name}}
accelerator.init_trackers(
    project_name=config.wandb_name,
    config=dict(config),
    init_kwargs=init_kwargs,
    )

num_training_steps = config.epochs * len(train_dl)
optimizer = AdamW(model.parameters(), lr=config.lr) # * world_size)
lr_scheduler = get_scheduler(
        name=config.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=config.num_warmup_steps,
        num_training_steps=num_training_steps,
    )

logger.info("Start training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
#model, loss_fn, optimizer, train_dl, eval_dl, lr_scheduler = accelerator.prepare(
#     model, loss_fn, optimizer, train_dl, eval_dl, lr_scheduler
#     )

train_len = len(train_dl)
test_len = len(eval_dl)

world_size = torch.cuda.device_count()

if world_size > 1:
    raise Exception("Only one GPU allowed, but got >1 world size")

if config.rank_metrics:
    #First log the ranking metrics (median_rank, r1, r5, r10)
    for k in [x for x in e_train.keys() if x != "fusion"]:
        accelerator.print(f"Ranking embeddings for {k}. This may take awhile")
        train_rank_mets = get_rank_metrics(e_train, k, device=device)
        test_rank_mets = get_rank_metrics(e_test, k, device=device)
        train_um, train_am = metrics_uniformity(e_train[k].to(device)), \
                             metrics_alignment(e_train[k].to(device),e_train['fusion'].to(device))
        test_um, test_am = metrics_uniformity(e_test[k].to(device)), \
                             metrics_alignment(e_test[k].to(device),e_test['fusion'].to(device))
        metrics = {
                "train_median_rank": train_rank_mets[0],
                "train_r1":train_rank_mets[1],
                'train_r5':train_rank_mets[2],
                'train_r10':train_rank_mets[3],
                "test_median_rank": test_rank_mets[0],
                "test_r1": test_rank_mets[1],
                'test_r5': test_rank_mets[2],
                'test_r10': test_rank_mets[3],
                "train_uniformity": train_um,
                "train_alignment": train_am,
                "test_uniformity": test_um,
                "test_alignment": test_am,
            }
        accelerator.log({f"{k}_{x}": v for x,v in metrics.items()})

#Linear probe to a value
model = model.to(device)
for epoch in tqdm(range(config.epochs)):
    epoch_loss_train, epoch_loss_eval = torch.Tensor([0.0]), torch.Tensor([0.0])
    model.train()
    for batch in train_dl:
        embedding, label = batch
        embedding, label =embedding.to(device), label.to(device)
        pred = model(embedding).squeeze()
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        accelerator.backward(loss)

        met.update(pred, label)
        # met.update(pred,(label>0.5).to(torch.int))
        epoch_loss_train += loss.detach().cpu()

        if config.clip:
            accelerator.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()
        lr_scheduler.step()
    train_met = met.compute()
    met.reset()
    #Epoch end log and checkpoint
    model.eval()
    #Eval looop
    with torch.no_grad():
        for batch in eval_dl:
            embedding, label = batch
            embedding, label = embedding.to(device),label.to(device)
            pred = model(embedding).squeeze()
            loss = loss_fn(pred, label)
            met.update(pred, label)
            epoch_loss_eval += loss.detach().cpu()
        eval_met = met.compute()
        met.reset()
    accelerator.log({'train_loss':epoch_loss_train/train_len,
                     'eval_loss':epoch_loss_eval/test_len,
                     f'train_{config.metric}': train_met,
                     f'eval_{config.metric}': eval_met,
                     'lr': optimizer.param_groups[0]['lr'],
                     "param_norm": get_param_norm(model).to("cpu"),
                    "grad_norm": get_grad_norm(model).to("cpu"),
                     })
logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
accelerator.end_training()