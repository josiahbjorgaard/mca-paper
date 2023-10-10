#!/bin/bash
#SBATCH --nodes=1 #4
#SBATCH --cpus-per-task=124
#SBATCH --tasks-per-node=1
##SBATCH --exclusive
#SBATCH --output=slurm-%x-%j.out
#conda activate work39
ulimit -Sn unlimited
ulimit -Hn unlimited
ulimit -u unlimited
ulimit -c unlimited
ulimit -Sn
ulimit -Hn
ulimit -u
ulimit -c
srun ./run_clm_geneformer_pretraining.sh MIXED
