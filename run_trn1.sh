#!/bin/bash
set -o pipefail

export NEURON_COMPILE_CACHE_URL="/shared/.cache/neuron"

sudo rmmod neuron; sudo modprobe neuron
sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620
sudo sysctl -w kernel.threads-max=10000000
ulimit -c unlimited

NUM_NEURONCORES=2
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"

LD_PRELOAD="/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4"
MALLOC_ARENA_MAX=64
echo "MALLOC_ARENA_MAX" $MALLOC_ARENA_MAX
echo "LD_PRELOAD" $LD_PRELOAD

if [ ! -z "$SLURM_NTASKS" ]; then
    # TBD this code path has not been tested
    # if running inside slurm, handle here
    MASTER_ADDR=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    MASTER_PORT=2022
    WORLD_SIZE_JOB=$SLURM_NTASKS
    RANK_NODE=$SLURM_NODEID
    JOB_ID_TAG=job-"$SLURM_JOB_ID"
    DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $WORLD_SIZE_JOB --node_rank $RANK_NODE --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
    echo $DISTRIBUTED_ARGS
    export NEURON_RT_ROOT_COMM_ID=$MASTER_ADDR:46820
    export FI_EFA_FORK_SAFE=1
    export FI_EFA_USE_DEVICE_RDMA=1
    export FI_PROVIDER=efa
    echo "WORLD_SIZE_JOB=$WORLD_SIZE_JOB,  RANK_NODE=$RANK_NODE,  MASTER_ADDR_JOB=$MASTER_ADDR_JOB, NODE_LIST=$NODE_LIST"
fi

#Print Slurm Config
date;hostname;

export TRAINING_PRECISION="BF16"
export NEURON_RT_STOCHASTIC_ROUNDING_EN=1
export XLA_USE_BF16=1
#export NEURON_CC_FLAGS="--retry_failed_compilation -O1"
export NEURON_CC_FLAGS=""
export NEURON_RT_EXEC_TIMEOUT=600
export TF_NUM_INTEROP_THREADS=8192
export NEURON_ENABLE_NOSEED_DROPOUT=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1

torchrun $DISTRIBUTED_ARGS train_trn1.py 
