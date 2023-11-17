#!/usr/bin/env python
# coding: utf-8
import os
import matplotlib.pyplot as plt
import numpy as np

#from sklearn.metrics import accuracy_score, auc, confusion_matrix, ConfusionMatrixDisplay, roc_curve
#from sklearn.model_selection import KFold

import torch
from torch import nn

from tqdm import tqdm
#import anndata

from transformers import PreTrainedModel
from transformers import AutoConfig, AutoModel
#from transformers import Trainer
from optimum.neuron import NeuronTrainer as Trainer
from optimum.neuron import NeuronTrainingArguments as TrainingArguments
from transformers import get_scheduler

from datasets import load_from_disk

from accelerate import Accelerator

from torchmetrics.regression import PearsonCorrCoef, MeanSquaredError, MeanAbsoluteError

from yacs.config import CfgNode as CN

import wandb

#For figure
import colorcet as cc
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from fast_histogram import histogram2d

config = CN()
config.epochs = 500
config.freeze_layers=6
config.train_batchsize = 2
config.val_batchsize = 3
config.warmup_steps = 1000
config.lr = 1e-4
#config.val_split = 0.3
#config.train_split = 0.05
config.dropout = 0.3
config.output_dir = "../stdata/output/test1"
config.num_labels = 1
config.hidden_size = 512
#adata = anndata.read('DGvelo_targets1.ann')
config.model_path = "/efs-private/Geneformer/geneformer-12L-30M"

os.makedirs(config.output_dir, exist_ok=True)


#wandb.init()
training_args = TrainingArguments(
        max_steps=32,
        output_dir=config.output_dir,
        )

wandb.init(project = "velociformer",
        config = dict(config),
        entity= "josiahbjorgaard")
class CustomModel(PreTrainedModel):
    def __init__(self, hf_config, config):
        super().__init__(hf_config)
        self.backbone = AutoModel.from_pretrained(config.model_path)
        self.dropout = nn.Dropout(config.dropout)
        self.output = nn.Linear(config.hidden_size, config.num_labels)

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        labels=None,
    ):
        #print(input_ids)
        outputs = self.backbone(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        outputs['logits'] = self.output(sequence_output)
        return outputs['logits'].squeeze()


hf_config = AutoConfig.from_pretrained(config.model_path)
model = CustomModel(hf_config, config) #, model_name)
freeze_layers = config.freeze_layers
if freeze_layers is not None:
    modules_to_freeze = model.backbone.encoder.layer[:freeze_layers]
    for module in modules_to_freeze:
        for param in module.parameters():
            param.requires_grad = False
            
padded_dataset = load_from_disk("DGvelo1_split.dataset")
    
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        mask = inputs.get('attention_mask').flatten()
        labels = inputs.get("labels").flatten()
        print(labels.shape)
        # forward pass
        outputs = model(**inputs)
        #logits = outputs.get("logits")
        print(outputs.shape)
        logits = outputs.flatten()
        loss_fct = nn.MSELoss()
        loss = loss_fct(labels[mask], logits[mask])
        return (loss, outputs) if return_outputs else loss

for dataset in padded_dataset.keys():
    print(dataset)
    padded_dataset[dataset]=padded_dataset[dataset].rename_column("label","labels")

trainer = CustomTrainer(model, args=training_args, train_dataset = padded_dataset['train'].with_format('torch'), eval_dataset = padded_dataset['val'].with_format('torch'))
trainer.train()

