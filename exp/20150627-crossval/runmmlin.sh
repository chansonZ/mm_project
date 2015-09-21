#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 4-00:00:00
#SBATCH -J MMLinWorkflow
python wfmmlin.py MMLinear \
    --dataset-name=acd_logd \
    --sampling-method=random \
    --test-size=50000 \
    --train-size=10000 \
    --lin-type=12 \
    --slurm-project=b2013262 \
    --runmode=hpc \
    --workers=2
