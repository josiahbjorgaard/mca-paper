"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...)
on a text file or a dataset without using HuggingFace Trainer.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""

import logging
import pickle
import torch_neuronx
import datasets
import torch
from datasets import load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import transformers
from accelerate import Accelerator, DistributedType
from accelerate.utils import set_seed
from transformers import get_scheduler

from transformers.utils import check_min_version, get_full_repo_name, send_example_telemetry
from transformers.utils.versions import require_version

import torch_xla.utils.serialization as xser
import functools
import torch_xla.core.xla_model as xm
from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch_xla.distributed.xla_backend
from neuron_utils import *

from .biozorromodel import BioZorro, BioZorroLayer
from .encoders import BioZorroCollator

# we need to use the torch_xla checkpoint. Otherwise the some checkpointing patterns will be eliminated by the compiler common expression elimination
torch.utils.checkpoint.checkpoint = torch_xla.utils.checkpoint.checkpoint

import wandb

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
#os.environ['NEURON_NUM_RECENT_MODELS_TO_KEEP'] = '3' #4 will result in OOM

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
    else:
        total_norm = local_norm**(1.0 / norm_type)
    return total_norm


def training_metrics_closure(epoch, global_step, loss, learning_rate, grad_norm=None, param_norm=None):
    if xm.is_master_ordinal(local=False) and args.with_tracking:
        loss_val = loss.detach().to('cpu').item()
        grad_norm_val = grad_norm.detach().to('cpu').item() if grad_norm else None
        param_norm_val = param_norm.detach().to('cpu').item() if param_norm else None
        wandb.log({"loss":loss_val, "grad_norm": grad_norm_val, "param_norm":param_norm_val, "learning_rate":learning_rate}) #, step=global_step)

def eval_metrics_closure(loss, loss_type, step):
    if xm.is_master_ordinal(local=False) and args.with_tracking:
        loss_val = loss.detach().to('cpu').item()
        wandb.log({f"val_{loss_type}_loss":loss_val}) #, step=step)

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
    accelerator_log_kwargs = {}
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
        set_seed(args.seed, device_specific=False)

    #### MODEL
    config = {
        "dim": 512, #hidden size
        "depth": 7, #layers
        "spliced_input_dim": 1024, #embedding_size
        "unspliced_input_dim": 1024,
        "dim_head":64, #don't know, head hidden size?
        "heads": 8, #num heads
        "ff_mult": 4, #Feed forward multiplier
        "num_fusion_tokens": 16,
    }

    model = BioZorro(**config)
    
    if xm.is_master_ordinal(local=False):
        n_params = count_parameters(model)
        print(f'TOTAL PARAMETERS: {n_params}')

    if args.use_fsdp:
        auto_wrap_policy = functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=(BioZorroLayer,)
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
        model = FSDP(model, **fsdp_params)

    #### DATASET
    xm.master_print("Loading tokenized dataset from ", args.load_tokenized_dataset)
    lm_datasets = load_from_disk(args.load_tokenized_dataset)
    for part in lm_datasets.keys():
        lm_datasets[part] = lm_datasets[part].remove_columns("length")
    train_dataset = lm_datasets["train"]
    eval_dataset = lm_datasets["test"]
    
    #BioZorro Collator
    default_data_collator = BioZorroCollator(pad_len=2048, pad_token=0)

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=default_data_collator, batch_size=args.per_device_train_batch_size,
        prefetch_factor=4, num_workers=2 #, pin_memory=True
    )
    eval_dataloader = DataLoader(
        eval_dataset, collate_fn=default_data_collator, batch_size=args.per_device_eval_batch_size,
        prefetch_factor=4, num_workers=2 #, pin_memory=True
    )

    #### OPTIMIZER
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
    num_update_steps_per_epoch = len(train_dataloader)
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
    args.max_train_steps = args.num_train_epochs * len(train_dataloader) 

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if xm.is_master_ordinal(local=False) and args.with_tracking:
        experiment_config = {**vars(args), **config.to_dict()}
        experiment_config["lr_scheduler_type"] = experiment_config["lr_scheduler_type"].value
        experiment_config["n_params"] = n_params
        wandb.init(
                project="BioZorro",
                config=experiment_config,
                entity="josiahbjorgaard",
                )
                            
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
    global_eval_step = 0
    optimizer.zero_grad()
    xm.mark_step()

    # Figure out how many steps we should save the Accelerator states
    checkpointing_steps = args.checkpointing_steps
    if checkpointing_steps is not None and checkpointing_steps.isdigit():
        checkpointing_steps = int(checkpointing_steps)
    elif checkpointing_steps == "epoch":
        checkpointing_steps = num_update_steps_per_epoch

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

            param_norm = get_param_norm(args, model)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            grad_norm = get_grad_norm(args, model)
            optimizer.optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            global_step+=1
            #if not extract_graphs_only:
            xm.mark_step()
            xm.add_step_closure(training_metrics_closure, (epoch, global_step, loss.detach(), optimizer.param_groups[0]['lr'],grad_norm, param_norm),run_async=False) #no data dependency with next mark_step

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
        
        #Evaluation
        xm.master_print(f"Running Eval Now")
        model.eval() 
        with torch.no_grad():
            epoch_loss = 0.0
            for i, batch in enumerate(tqdm(eval_dataloader)):
                outputs = model(**batch)
                loss = outputs.loss
                running_loss_div = loss.detach() / world_size
                running_loss_reduced = xm.all_reduce(xm.REDUCE_SUM, running_loss_div, groups=None)
                #xm.master_print(f"Loss: {running_loss_reduced}")
                global_eval_step+=1
                epoch_loss += running_loss_reduced
                xm.mark_step()
                xm.add_step_closure(eval_metrics_closure, (running_loss_reduced, 'step', global_eval_step), run_async=False)
            xm.add_step_closure(eval_metrics_closure, (epoch_loss/len(eval_dataloader), 'epoch', epoch), run_async=False)

    if args.with_tracking():
        wandb.finish()

if __name__ == "__main__":
    main()
