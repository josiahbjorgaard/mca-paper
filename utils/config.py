from yacs.config import CfgNode as CN
from datetime import datetime
import os
from contextlib import redirect_stdout
import json
import yaml


def get_cfg_defaults_train():
    """
    Default config options for training
    """
    config = CN(new_allowed=True)
    config.encoder_configs = CN(new_allowed=True) # Encoder configuration
    config.modality_configs = CN(new_allowed=True) # Collator configuration


    # Training and dataset configuration
    config.restart = "" # Model weights path for loading from file
    config.wandb_name = "No Name" # Name for WandB Project
    config.wandb_account_name = "" # Your WandB account name
    config.wandb_restart = "" # WandB experiment code for restarting
    config.epochs = 3 # Number of epochs for training
    config.start_epoch = 0 # Epoch number for start (for restarting)
    config.batch_size = 32 # Batch size
    config.n_step_checkpoint = 0
    config.num_warmup_steps = 3000 # Number of warmup steps for scheduler
    config.lr_scheduler_type = "cosine" # Scheduler type
    config.lr = 1e-4 # Learning rate
    config.output_dir = "" # Output directory, created based on timestamp otherwise
    config.label_col = "Labels" # Column name in dataset for labels
    config.dataset = "" # Hugginfaces Datasets library dataset path
    config.split = 0.1 # Train/Test split if not already split
    config.ds_frac = 1.0 # Fraction of dataset to use
    config.ds_seed = 42 # Dataset random Seed
    config.clip = 0.0 # Gradient clipping factor


    # Model configuration
    config.hidden_size = 512 # Model hidden size
    config.layers = 10 # Number of model layers
    config.heads = 8  # num heads
    config.dim_head = 64 #Dimension of heads, generally hidden_size/layers
    config.ff_mult = 4  # Feed forward multiplier
    config.num_fusion_tokens = 256 # Number of fusion tokens to use - must be divisible by number of channels (see paper)
    config.seed = 42 # Python random seed
    config.mean_pool = False # Use mean pooling instead of attentive pooling when True
    config.dropout = 0.1 # Global dropout parameter for all dropout layers
    config.zorro = False # Use the Zorro-type Masked-Multimodal Attention (No Modal Fusion Channels)
    config.eao = False # Use Everything at Once model (No overall fusion)
    config.run_eval_loop = True # Set to run eval loop or disable it
    config.bimodal_contrastive = True # If set to True, non-fusion unimodal-unimodal token pairs are contrasted
    config.non_fusion_fcl = True # If set to True, fusion - non-fusion unimodal token pairs are contrasted
    config.fcl = True # If set tot True, each fusion-fusion token pair is contrasted
    config.no_fusion = False # If set to True, no fusion tokens are used
    config.fcl_root = [1,2,3,4] # The root tuple of modalities, as ordered in encoder config. Generally is the combination of all modalities.
    config.fusion_combos = [4,3,2] # The cardinalities of combinations to use. For example, [4,3] will use all 4-wise and 3-wise combinations of different modalities for fusion channels.
    config.return_logits = True # If True, model will return logits with the loss
    #N.B. If config.restart, will reset all config items to checkpoint yaml

    return config.clone()

def restart_cfg(config):
    """
    Revise config options if restarting
    """
    if config.restart:
        # Allow creating new keys recursively.
        config.set_new_allowed(True)
        config.merge_from_file(os.path.join(config.restart, 'config.yaml'))
        config.epochs = 1
        config.output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
        config.reset_lr = 0.0001
    return config

def training_config(filename):
    config = get_cfg_defaults_train()
    with open(filename, "r") as stream:
        config_dict = yaml.safe_load(stream)
    new_config = CN(config_dict)
    if not config.output_dir:
        output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
        config.output_dir = output_dir
        i = 1
        while os.path.isdir(config.output_dir):
            config.output_dir = output_dir + f'_{i}' 
            i+=1 
    print(new_config)
    #config.merge_from_file(filename)
    config.merge_from_other_cfg(new_config)
    dump_configs(config, config.output_dir)
    #config.freeze()
    return config


def get_model_config(config):
    #### MODEL
    model_config = {
        "dim": config.hidden_size,  # hidden size
        "depth": config.layers,  # layers
        "heads": config.heads,  # num heads
        "dim_head": config.dim_head, # heads * dim_head = intermediate size
        "ff_mult": config.ff_mult,  # Feed forward multiplier
        "num_fusion_tokens": config.num_fusion_tokens,
        "encoder_configs": config.encoder_configs,
        "batch_size": config.batch_size,
        "fcl": config.fcl,
        "fcl_root": config.fcl_root,
        "bimodal_contrastive": config.bimodal_contrastive,
        "non_fusion_fcl": config.non_fusion_fcl,
        "fusion_combos": config.fusion_combos,
        "zorro": config.zorro,
        "eao": config.eao,
        "no_fusion": config.no_fusion,
        "mean_pool": config.mean_pool
    }
    return model_config


def dump_configs(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir,'config.yaml'),'w') as f:
        with redirect_stdout(f): print(config.dump())

def dump_model_configs(config, output_dir):
    with open(os.path.join(output_dir,'model_config.json'),'w') as f:
        json.dump(get_model_config(config), f)

def get_cfg_defaults_embedding_eval():
    """
    Default config options for training
    """
    config = CN(new_allowed=True)
    config.embedding_dir = ""
    config.task = 0
    config.loss_type = "L1"
    config.model_type = "linear"
    config.hidden_size = 256
    config.dropout = 0.1
    config.wandb_name = "MCA"
    config.lr = 1e-5
    config.lr_scheduler_type = "cosine"
    config.num_warmup_steps = 1000
    config.rank_metrics = True
    config.epochs = 1024
    config.clip = 2.0
    config.metric = "PCC"
    config.output_dir = ""
    config.wandb_job_name = "MCA-DefaultJobName"
    config.seed = 42
    config.batch_size = 1024
    config.threshold = 0.0
    return config.clone()

def embedding_eval_config(filename):
    config = get_cfg_defaults_embedding_eval()
    with open(filename, "r") as stream:
        config_dict = yaml.safe_load(stream)
    new_config = CN(config_dict)
    if not config.output_dir:
        output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
        config.output_dir = output_dir
        i = 1
        while os.path.isdir(config.output_dir):
            config.output_dir = output_dir + f'_{i}'
            i+=1
    print(new_config)
    config.merge_from_other_cfg(new_config)
    dump_configs(config, config.output_dir)
    return config
