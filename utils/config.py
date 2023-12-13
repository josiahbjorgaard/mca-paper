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
    config.encoder_configs = CN(new_allowed=True) #None #{}
    config.modality_configs = CN(new_allowed=True)
    config.restart = False #'training_output_21_31_23_10_2023'
    config.epochs = 3
    config.batch_size = 2
    config.num_warmup_steps = 3000
    config.lr_scheduler_type = "cosine"
    config.lr = 1e-4
    config.output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
    config.hidden_size = 512
    config.layers = 10
    config.heads = 8  # num heads
    config.dim_head = 64
    config.ff_mult = 4  # Feed forward multiplier
    config.num_fusion_tokens = 256
    config.dataset = "/shared/dataset3M" #"/shared/fcaa53cd-ba57-4bfe-af9c-eaa958f95c1a_combined_all"
    config.split = 0.1
    config.ds_frac = 1.0
    config.ds_seed = 42
    config.seed = 42
    config.dropout = 0.1
    config.clip = 0.0
    config.isolate_fusion_tokens = True
    config.pad_len = 1024
    config.model = 3
    config.n_step_checkpoint = 20000
    config.run_eval_loop = True
    config.vocab_size = 20000 #36602
    config.inverse_doom = False
    #If config.restart, will reset all config items to checkpoint yaml
    return config.clone()

def restart_cfg(config):
    """
    Revise config options if restarting
    """
    if config.restart:
        # Allow creating new keys recursively.
        config.set_new_allowed(True)
        config.merge_from_file(os.path.join(config.restart, 'config.yaml'))
        config.epochs = 1 ### WILL NEED TO SPECIFY NUMBER OF EPOCHS TO CONTINUE WITH HERE
        ### New Output directory!!
        config.output_dir = datetime.now().strftime('training_output_%H_%M_%d_%m_%Y')
        config.reset_lr = 0.0001
    return config


def training_config(filename):
    config = get_cfg_defaults_train()
    with open(filename, "r") as stream:
        config_dict = yaml.safe_load(stream)
    new_config = CN(config_dict)
    print(new_config)
    #config.merge_from_file(filename)
    config.merge_from_other_cfg(new_config)
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
       #"vocab_size": config.vocab_size,
        "batch_size": config.batch_size,
        "inverse_doom": config.inverse_doom
    }
    return model_config


def dump_configs(config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir,'config.yaml'),'w') as f:
        with redirect_stdout(f): print(config.dump())
    with open(os.path.join(output_dir,'model_config.json'),'w') as f:
        json.dump(get_model_config(config), f)
