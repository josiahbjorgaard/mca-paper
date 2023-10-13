# HF orginal code from https://raw.githubusercontent.com/huggingface/transformers/v4.26.1/examples/pytorch/language-modeling/run_clm_no_trainer.py
#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import json
import logging
import math
import os
import random
from itertools import chain
from pathlib import Path
import pickle
import torch_neuronx
import datasets
import torch
from datasets import load_dataset
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import queue
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import Repository, create_repo
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    BertForMaskedLM,
    AutoModel,
    AutoTokenizer,
    SchedulerType,
    #default_data_collator,
    DataCollatorForLanguageModeling,
    get_scheduler,
    BertLayer,
    GPT2Config,
    GPT2Model,
    GPT2LMHeadModel,
    GPTNeoConfig,
    GPTNeoModel,
    GPTNeoForCausalLM
)

from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from transformers.models.gpt_neo.modeling_gpt_neo import GPTNeoBlock

import time
import contextlib
from transformers.modeling_utils import PreTrainedModel
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

import numpy as np
import torch_xla.utils.serialization as xser
import functools
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch_xla.distributed.xla_backend
from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
from neuron_utils import *
from accelerate.utils.imports import is_tpu_available

from geneformer.pretrainer import GeneformerPreCollator
# we need to use the torch_xla checkpoint. Otherwise the some checkpointing patterns will be eliminated by the compiler common expression elimination
torch.utils.checkpoint.checkpoint = torch_xla.utils.checkpoint.checkpoint

import wandb


try:
    from utilities.reporting import Metric, post_metrics
except ImportError:
    Metric = post_metrics = lambda *args, **kwargs: None

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

logger = get_logger(__name__)

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

os.environ['NEURON_RT_ONE_TMPBUF_PAGE_SIZE_MB']='2048'
os.environ['TMPDIR']="/shared/tmp/"
os.environ['HF_HOME']="/shared/.cache/huggingface"
os.environ['HF_DATASETS_CACHE']="/shared/.cache/datasets"
os.environ['TRANSFORMERS_CACHE']="/shared/.cache/transformers"
os.environ['NEURON_COMPILE_CACHE_URL']="/shared/.cache/neuron"

# Uncomment below to keep only 2 subgraphs loaded at a time
os.environ['NEURON_NUM_RECENT_MODELS_TO_KEEP'] = '3' #4 will result in OOM
if os.environ.get("XLA_DOWNCAST_BF16") == '1':
    Bf16 = torch.finfo(torch.bfloat16)
    Fp32 = torch.finfo(torch.float32)
    torch.finfo = lambda a: Fp32 if a == torch.float64 else Bf16

if os.environ.get("XLA_USE_BF16") == '1':
    Bf16 = torch.finfo(torch.bfloat16)
    torch.finfo = lambda a:Bf16


def get_param_norm(
    args,
    model,
    norm_type=2.0,
    groups=None,
) -> torch.Tensor:

    norm_type = float(norm_type)
    local_norm = torch.DoubleTensor([0.0]).to('xla')
    parameters = model.parameters()
    grads_for_norm = []
    for param in parameters:
        param_norm = torch.norm(param.detach(), norm_type)
        local_norm += param_norm ** norm_type

    if args.use_fsdp:
        total_norm = model.all_reduce_op(xm.REDUCE_SUM, local_norm, groups=groups)
        total_norm = total_norm**(1.0 / norm_type)
    elif args.use_zero1:
        total_norm = xm.all_reduce(xm.REDUCE_SUM, local_norm, groups=groups, pin_layout=False)
        total_norm = total_norm**(1.0 / norm_type)
    else:
        total_norm = local_norm**(1.0 / norm_type)
    #return total_norm.cpu().item()
    return total_norm

def get_grad_norm(
    args,
    model,
    norm_type=2.0,
    groups=None,
) -> torch.Tensor:

    norm_type = float(norm_type)
    local_norm = torch.FloatTensor([float(0.0)]).to('xla')
    parameters = model.parameters()
    grads_for_norm = []
    for param in parameters:
        grad_not_none = param.grad is not None
        if grad_not_none:
            grad = param.grad.detach()
            grad_norm = torch.norm(grad, norm_type)
            local_norm += grad_norm ** norm_type

    if args.use_fsdp:
        #Gradients are scattered, so need to add all of them together
        total_norm = model.all_reduce_op(xm.REDUCE_SUM, local_norm, groups=groups)
        total_norm = total_norm**(1.0 / norm_type)
    elif args.use_zero1:
        total_norm = xm.all_reduce(xm.REDUCE_SUM, local_norm, groups=groups, pin_layout=False)
        total_norm = total_norm**(1.0 / norm_type)
    else:
        total_norm = local_norm**(1.0 / norm_type)
    #return total_norm.cpu().item()
    return total_norm


def training_metrics_closure(logger_metrics, epoch, global_step, loss, learning_rate, tp, grad_norm=None, param_norm=None):
    loss_val = loss.detach().to('cpu').item()
    grad_norm_val = grad_norm.detach().to('cpu').item() if grad_norm else None
    param_norm_val = param_norm.detach().to('cpu').item() if param_norm else None
    if logger_metrics != None:
        logger_metrics.log(epoch, global_step, loss_val, learning_rate, tp, grad_norm_val, param_norm_val, noisy_check=True)
        #if args.with_tracking:
        wandb.log({"loss":loss_val, "grad_norm": grad_norm_val, "param_norm":param_norm_val, "learning_rate":learning_rate, "throughput":tp}, step=global_step)

def build_chkpt_path(output_dir, step, rank, world_size):
    chkpt_path = os.path.join(output_dir, f"step-{step}-rank-{rank}-of-{world_size}.ckpt")
    return chkpt_path

def main():

    torch.distributed.init_process_group('xla')
    device = xm.xla_device()
    rank = xm.get_ordinal()
    world_size = xm.xrt_world_size()

    print(f'rank: {rank}, world size {world_size}')

    args = parse_args()

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm_no_trainer", args)

    # Initialize the accelerator. We will let the accelerator handle device placement for us in this example.
    # If we're using tracking, we also need to initialize it here and it will by default pick up all supported trackers
    # in the environment
    accelerator_log_kwargs = {}

    #This doesn't seem to work with wandb
    #if args.with_tracking:
        #accelerator_log_kwargs["log_with"] = args.report_to
        #accelerator_log_kwargs["logging_dir"] = args.output_dir

    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps, **accelerator_log_kwargs)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        if args.use_mics:
            set_seed(args.seed, device_specific=True)
            # Do not need this, since device_specific=False in set_seed above
            seed_group_size = int(os.environ.get("NEURON_MICS_PARTITION_GROUP_SIZE", 32))
            seed = args.seed + rank % seed_group_size
            np.random.seed(seed=seed)
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            if is_tpu_available():
                xm.set_rng_state(seed)
        else:
            set_seed(args.seed, device_specific=False)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.push_to_hub:
            if args.hub_model_id is None:
                repo_name = get_full_repo_name(Path(args.output_dir).name, token=args.hub_token)
            else:
                repo_name = args.hub_model_id
            create_repo(repo_name, exist_ok=True, token=args.hub_token)
            repo = Repository(args.output_dir, clone_from=repo_name, token=args.hub_token)

            with open(os.path.join(args.output_dir, ".gitignore"), "w+") as gitignore:
                if "step_*" not in gitignore:
                    gitignore.write("step_*\n")
                if "epoch_*" not in gitignore:
                    gitignore.write("epoch_*\n")
        elif args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if args.load_tokenized_dataset is not None:
        xm.master_print("Loading tokenized dataset from ", args.load_tokenized_dataset)
        lm_datasets = load_from_disk(args.load_tokenized_dataset).train_test_split(test_size=0.05)
    
    if args.config_name:
        config = AutoConfig.from_pretrained(args.config_name)
    elif args.model_name_or_path:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
    else:
        config = CONFIG_MAPPING[args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    if args.tokenizer_name and args.tokenizer_name != 'None':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=not args.use_slow_tokenizer)
    elif args.model_name_or_path and args.tokenizer_name != 'None':
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_fast=not args.use_slow_tokenizer)
    elif args.block_size is not None: #Geneformer hack
        tokenizer = type('obj', (object,), {'model_max_length' : args.block_size})
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    if args.block_size is not None and args.block_size > tokenizer.model_max_length:
        tokenizer.model_max_length = args.block_size
        logger.warning(
            f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
            f"({tokenizer.model_max_length}). Setting tokenizer.model_max_length to {args.block_size}."
        )

    if args.model_name_or_path:
        model_config = {
            #"vocab_size": 50304 if args.use_zero1 else len(tokenizer),  # zero1 not support padding
            "vocab_size": 25426, #Geneformer hack
            "max_length": args.block_size,
            #"fused_scaled_masked_softmax": True, #args.fused_scaled_masked_softmax,
            #"fused_gelu": args.fused_gelu,
            "gradient_checkpointing": args.gradient_checkpointing,
            "use_cache": not args.gradient_checkpointing,
        }

        config = AutoConfig.from_pretrained(args.model_name_or_path)
        #model = AutoModelForMaskedLM.from_config(config)
        model = BertForMaskedLM(config)

        if xm.is_master_ordinal(local=False):
            print('==========================================================================')
            print(f'TOTAL PARAMETERS: {count_parameters(model)}')
            print('==========================================================================')

        # remove model = model.to('xla') before FSDP wrapper, so that the sharding will happen in CPU and only the sharded tensor will be sent to device
        # model = model.to('xla')

        model_dtype = get_dtype(model)
        extract_graphs_only = os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", None)
        if xm.is_master_ordinal(local=False) and not extract_graphs_only:
            logger_metrics = Logger(args, world_size, model_dtype)
        else:
            logger_metrics = None

        # Moved here for FSDP because once we wrap with FSDP the weights will be sharded
        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        # Comment out for Geneformer
        #if len(tokenizer) > embedding_size:
        #    model.resize_token_embeddings(len(tokenizer))
        if args.use_fsdp:
            auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=(BertLayer,)
            )
            fsdp_params = dict(flatten_parameters=False,
                               shard_param_on_dim_0=True,
                               optimization_barrier_in_forward=True,
                               optimization_barrier_in_backward=True,
                               reshard_after_forward=True,  # Save memory by keep all-gathers in bwd
                               disable_reshard_on_root=False,
                               coalesce_all_gather_ops=False,
                               auto_wrap_policy=auto_wrap_policy,
                               _debug_print=True, _debug_msg=f'Worker {rank}')
            if os.environ.get('TRAINING_PRECISION', default='') == 'MIXED':
                fsdp_params['compute_dtype'] = torch.bfloat16
            if args.use_mics:
                from torch_neuronx.distributed.fsdp_mics import XlaFullyShardedDataParallelMiCS as FSDPMiCS
                model = FSDPMiCS(model, **fsdp_params)
            else:
                model = FSDP(model, **fsdp_params)
            # Need to re-assign the shared module to use the correct FSDP wrapper
            # In BERT:
            # model.cls.predictions.decoder = model.bert.embeddings.word_embeddings
            # Here counter-part, but need to verify
            # model.lm_head = model.transformer.wte
        elif args.use_zero1:
            if model_dtype == "torch.float32":
                model = model.to(device='xla', dtype=torch.float32)
            elif model_dtype == "torch.bfloat16":
                model = model.to(device='xla', dtype=torch.bfloat16)
    else:
        logger.info("Training new model from scratch")
        #model = AutoModelForCausalLM.from_config(config)
        model = AutoModelForMaskedLM.from_config(config)
        # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
        # on a small vocab and want a smaller embedding size, remove this test.
        embedding_size = model.get_input_embeddings().weight.shape[0]
        if len(tokenizer) > embedding_size:
            model.resize_token_embeddings(len(tokenizer))


    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    if args.block_size is None:
        block_size = tokenizer.model_max_length
    else:
        block_size = args.block_size

    for part in lm_datasets.keys():
        lm_datasets[part] = lm_datasets[part].remove_columns("length")
    train_dataset = lm_datasets["train"]#.with_format('torch')
    eval_dataset = lm_datasets["test"]#.with_format('torch')
    
    # Log a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    #Geneformer Collator
    with open("/efs-private/Genecorpus-30M/token_dictionary.pkl", "rb") as fp:
        token_dictionary = pickle.load(fp)
    precollator = GeneformerPreCollator(token_dictionary=token_dictionary)
    default_data_collator = DataCollatorForLanguageModeling(
        tokenizer=precollator, mlm=True, mlm_probability=0.15,
        pad_to_multiple_of=2048
        )

    # DataLoaders creation:

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size,
        prefetch_factor=4, num_workers=3, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size,
        prefetch_factor=4, num_workers=3, pin_memory=True
    )

    # Optimizer
    # Split weights in two groups, one with weight decay and the other not.
    no_decay = ["bias", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True


    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader, eval_dataloader
    )

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        model.tie_weights()

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    #if args.with_tracking:
    experiment_config = vars(args)
        # TensorBoard cannot log Enums, need the raw value
    experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        #accelerator.init_trackers("clm_no_trainer", experiment_config)
    if xm.is_master_ordinal(local=False):
        wandb.init(
                project="Geneformer Pretraining",
                config=experiment_config,
                entity="josiahbjorgaard",
                )
                            
#
    # Train!
    if xm.is_master_ordinal(local=False):
        print(args)
    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    if xm.is_master_ordinal(local=False):
        print("***** Running training *****")
        print(f"  Num examples = {len(train_dataset)}")
        print(f"  Num Epochs = {args.num_train_epochs}")
        print(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        print(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        print(f"  Total optimization steps = {args.max_train_steps}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    throughput = Throughput(args.per_device_train_batch_size, world_size, args.gradient_accumulation_steps)
    starting_epoch = 0
    resume_step = 0

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint is not None or args.resume_from_checkpoint != "":
            if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", "0") == "0":
                print("Resuming from checkpoint")
                resume_step = args.resume_step if args.resume_step is not None else 0
                ckpt_path = build_chkpt_path(args.resume_from_checkpoint, resume_step, rank, world_size)
                ckpt = xser.load(ckpt_path)
                model.load_state_dict(ckpt['model'])
                optimizer.load_state_dict(ckpt["optimizer"])
                starting_epoch = ckpt["epoch"]
                del ckpt
                xm.rendezvous("Checkpoint loaded")
        else:
            raise ValueError(f"Please specify a checkpoint to resume from")

    # update the progress_bar if load from checkpoint
    progress_bar.update(starting_epoch * num_update_steps_per_epoch)
    global_step = starting_epoch * num_update_steps_per_epoch

    optimizer.zero_grad()
    xm.mark_step()

    optimizer_step_done_at_least_once=0
    torch_neuronx.xla_impl.ops.set_unload_prior_neuron_models_mode(True)
    running_loss = torch.zeros(1, ).to(device)
    for epoch in range(starting_epoch, args.num_train_epochs):
        model.train()
        #if args.with_tracking:
        total_loss = 0
        for step, batch in enumerate(train_dataloader):
            start_time = time.time()
            # We need to skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == starting_epoch:
                if resume_step is not None and global_step < resume_step:
                    if step % args.gradient_accumulation_steps == 0:
                        progress_bar.update(1)
                        lr_scheduler.step()
                        global_step += 1
                    continue
            if optimizer_step_done_at_least_once < 2:
                optimizer_step_done_at_least_once+=1
                if optimizer_step_done_at_least_once==2:
                    torch_neuronx.xla_impl.ops.set_unload_prior_neuron_models_mode(False)
                    time.sleep(1)
                    xm.rendezvous("Init Complete")

            if args.use_fsdp and args.use_mics:
                param_norm = get_param_norm(args, model, groups=model.mics_sharding_cfg.partition_groups)
            elif args.use_zero1:
                param_norm = None
            else:
                param_norm = get_param_norm(args, model)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            #running_loss_div = loss.detach() / world_size
            #xm.master_print(f"Printing the running loss div on master: {running_loss_div}")
            #running_loss_reduced = xm.all_reduce(xm.REDUCE_SUM, running_loss_div, groups=None)
            #running_loss.zero_()
            #xm.master_print(f"Printing the running loss reduced on master: {running_loss_reduced}")
            # Record grad norm
            grad_norm = get_grad_norm(args, model)

                # gradient norm clipping
                #if args.use_grad_clipping and not args.use_zero1:
                #    if args.use_fsdp:
                #        model.clip_grad_norm_(max_norm=args.max_grad_norm)
                #    else:
                #        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            optimizer.optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step+=1
            tp = throughput.get_throughput()
            #if not extract_graphs_only:
            xm.mark_step()
            xm.add_step_closure(training_metrics_closure, (logger_metrics, epoch, global_step, loss.detach(), optimizer.param_groups[0]['lr'], tp, grad_norm, param_norm),run_async=False) #no data dependency with next mark_step

            progress_bar.update(1)

            if isinstance(checkpointing_steps, int):
                if global_step % checkpointing_steps == 0:
                    if os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY", "0") == "0":
                        ckpt_path = build_chkpt_path(args.output_dir, global_step, rank, world_size)
                        ckpt = {
                            "model": model.state_dict(),
                            # also save "shard_metadata" for checkpoint consolidation later via
                            # `python3 -m torch_xla.distributed.fsdp.consolidate_sharded_ckpts`
                            "shard_metadata": model.get_shard_metadata() if isinstance(model, FSDP) else None,
                            "optimizer": optimizer.state_dict(),
                            "epoch": epoch,
                        }
                        xser.save(ckpt, ckpt_path, master_only=False)
                        xm.master_print(f"Checkpoint saved to {ckpt_path}", flush=True)
                        xm.rendezvous("Checkpoint saved")

            if global_step >= args.max_train_steps:
                if xm.is_master_ordinal(local=False) and not extract_graphs_only:
                    average_throughput = round(sum(logger_metrics.throughputs)/len(logger_metrics.throughputs), 4)
                    if not os.environ.get("FI_EFA_USE_DEVICE_RDMA", None) and world_size > 32:
                       # multi-node/4-nodes throughput check
                        assert(average_throughput >= 45), "Average throughput :{} is  below derived expected derived threshold: {}".format(average_throughput, str(45))
                    else:
                        # single node throughput check
                        assert(average_throughput >= 14), "Average throughput :{} is  below derived expected derived threshold: {}".format(average_throughput, str(14))
                break

    #if args.with_tracking:
    accelerator.end_training()

    if args.output_dir is not None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            #tokenizer.save_pretrained(args.output_dir)
            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", auto_lfs_prune=True)


if __name__ == "__main__":
    main()
