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
import torchmetrics as tm
from accelerate import Accelerator

class FineTuneDataset(Dataset):
    def __init__(self, embeddings, labels, key='fusion',index = 0,transform=None, target_transform=None):
        #self.embeddings = torch.cat([embeddings[k] for k in embeddings.keys()],dim=1)
        self.embeddings = embeddings[key]
        if index == -1:
            self.labels=labels
        else:
            self.labels = labels[:,index] #First is a sentiment [-3,3] -3: negative 3:postive, next is 6 labels for emotions [0,3] where 3 is strong and 0 is no x and 3 is strong x
        print(f"{self.labels.shape = }")
        #if index > 0:
        #    self.labels = self.labels/3.0
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return self.labels.shape[0]
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

accelerator = Accelerator(log_with="wandb")
config = embedding_eval_config(sys.argv[1])

# Initialise your wandb run, passing wandb parameters and any config information
init_kwargs={"wandb": {"entity": "josiahbjorgaard", "name":config.wandb_job_name}}
accelerator.init_trackers(
    project_name=config.wandb_name,
    config=dict(config),
    init_kwargs=init_kwargs,
    )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.manual_seed(config.seed)

device = accelerator.device

#TODO check if these exist, and tell user ot run inference first if not
e_train=torch.load(f'{config.embedding_dir}/train_embeddings.pt', map_location="cpu")
m_train=torch.load(f'{config.embedding_dir}/train_masks.pt', map_location="cpu")
s_train=torch.load(f'{config.embedding_dir}/train_labels.pt', map_location="cpu").squeeze()
e_test=torch.load(f'{config.embedding_dir}/eval_embeddings.pt', map_location="cpu")
m_test=torch.load(f'{config.embedding_dir}/eval_masks.pt', map_location="cpu")
s_test=torch.load(f'{config.embedding_dir}/eval_labels.pt', map_location="cpu").squeeze()

print(f"Shape of test labels: {s_test.shape}")
print(f"Shape of train labels: {s_train.shape}")

metrics_alignment = Alignment()
metrics_uniformity = Uniformity()

if config.rank_metrics:
    #First log the ranking metrics (median_rank, r1, r5, r10)
    targets = torch.stack([e_train['fusion'],e_test['fusion']])
    for k in [x for x in e_train.keys() if isinstance(x, str) and x != "fusion"]:
        accelerator.print(f"Ranking embeddings for {k}. This may take awhile")
        train_rank_mets = get_rank_metrics(e_train[k][m_train[k]], targets, device=device)
        test_rank_mets = get_rank_metrics(e_test[k][m_test[k]], targets, device=device)
        train_um, train_am = metrics_uniformity(e_train[k][m_train[k]].to(device)), \
                             metrics_alignment(e_train[k][m_train[k]].to(device),e_train['fusion'][m_train[k]].to(device))
        test_um, test_am = metrics_uniformity(e_test[k][m_test[k]].to(device)), \
                             metrics_alignment(e_test[k][m_test[k]].to(device),e_test['fusion'][m_test[k]].to(device))
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
    accelerator.log({f"train_uniformity_fusion": metrics_uniformity(e_train['fusion'].to(device)),
        "test_uniformity_fusion": metrics_uniformity(e_test['fusion'].to(device))})

    # Creating a DataLoader object for iterating over it during the training epochs
train_dl = DataLoader(FineTuneDataset(e_train, s_train, index=config.task), batch_size=config.batch_size, shuffle=True)
eval_dl = DataLoader(FineTuneDataset(e_test, s_test, index=config.task), batch_size=config.batch_size)
# Model
e,l = next(iter(train_dl))
try:
    num_labels = l.shape[1]
except:
    num_labels = 1
num_emb = e.shape[1]
if config.model_type == 'linear':
    model = nn.Linear(num_emb, num_labels)
elif config.model_type.lower() == 'mlp':
    model = nn.Sequential(nn.Linear(num_emb, config.hidden_size),
                          nn.Dropout(config.dropout),
                          nn.ReLU(),
                          nn.Linear(config.hidden_size,num_labels ))
else:
    exit()


# Metrics config
if config.loss_type == "BCE":
    metrics = {
        'precision': tm.Precision(task='binary').to(device),
        'recall': tm.Recall(task='binary').to(device),
        'accuracy': tm.Accuracy(task='binary').to(device),
        'cm': tm.ConfusionMatrix(task='binary').to(device),
        'f1': tm.F1Score(task='binary').to(device),
        'specificity': tm.Specificity(task='binary').to(device),
        'auroc': tm.AUROC(task='binary').to(device),
        'auprc': tm.AveragePrecision(task='binary').to(device),
    }
elif config.loss_type == "CE":
    metrics = {
        'precision': tm.Precision(task='multiclass', num_classes=num_labels).to(device),
        'recall': tm.Recall(task='multiclass', num_classes=num_labels).to(device),
        'accuracy': tm.Accuracy(task='multiclass', num_classes=num_labels).to(device),
        'cm': tm.ConfusionMatrix(task='multiclass', num_classes=num_labels).to(device),
        'f1': tm.F1Score(task='multiclass', num_classes=num_labels).to(device),
        'specificity': tm.Specificity(task='multiclass', num_classes=num_labels).to(device),
        'auroc': tm.AUROC(task='multiclass', num_classes=num_labels).to(device),
        'auprc': tm.AveragePrecision(task='multiclass', num_classes=num_labels).to(device),
    }


elif config.loss_type in ["L1","MSE"]:
    metrics = {"PCC":tm.PearsonCorrCoef().to(device)}
else:
    raise Exception("Didn't recognize config.metric")

if config.loss_type == "L1":
    loss_fn = nn.L1Loss().to(device)
elif config.loss_type == "MSE":
    loss_fn = nn.MSELoss().to(device)
elif config.loss_type == "BCE":
    loss_fn = nn.BCEWithLogitsLoss().to(device)
elif config.loss_type == "CE":
    loss_fn = nn.CrossEntropyLoss()

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
#Linear probe to a value
model = model.to(device)
for epoch in tqdm(range(config.epochs)):
    epoch_loss_train, epoch_loss_eval = torch.Tensor([0.0]), torch.Tensor([0.0])
    model.train()
    for batch in train_dl:
        embedding, label = batch
        #if config.loss_type == 'BCE':
        #    label = (label > config.threshold).to(torch.float)
        embedding, label = embedding.to(device), label.to(device)
        pred = model(embedding).squeeze()
        loss = loss_fn(pred, label)
        optimizer.zero_grad()
        accelerator.backward(loss)
        if config.loss_type in ['BCE', 'CE']:
            label = label.to(torch.long)
        for k,v in metrics.items():
            v.update(pred, label)
        # met.update(pred,(label>0.5).to(torch.int))
        epoch_loss_train += loss.detach().cpu()

        if config.clip:
            accelerator.clip_grad_norm_(model.parameters(), config.clip)
        optimizer.step()
        lr_scheduler.step()
    train_met = {f"train_{k}":met.compute() for k,met in metrics.items()}
    for met in metrics.values():
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
            if config.loss_type in ['BCE','CE']:
                label = label.to(torch.long)
            for k,v in metrics.items():
                v.update(pred, label)
            epoch_loss_eval += loss.detach().cpu()
        eval_met = {f"eval_{k}":met.compute() for k, met in metrics.items()}
        for met in metrics.values():
            met.reset()
    print(train_met)
    accelerator.log({'train_loss':epoch_loss_train/train_len,
                     'eval_loss':epoch_loss_eval/test_len,
                     'lr': optimizer.param_groups[0]['lr'],
                     "param_norm": get_param_norm(model).to("cpu"),
                    "grad_norm": get_grad_norm(model).to("cpu"),
                     }|train_met|eval_met)
logger.info("End training: {}".format(strftime("%Y-%m-%d %H:%M:%S", gmtime())))
accelerator.end_training()
