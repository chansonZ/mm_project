#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 4-00:00:00
#SBATCH -J MMLinWorkflow
python wfmmlin.py MMLinear \
    --dataset-name=acd_logd \
    --sampling-method=random \
    --lin-type=0 \
    --lin-cost=1000 \
    --slurm-project=b2013262 \
    --runmode=hpc
