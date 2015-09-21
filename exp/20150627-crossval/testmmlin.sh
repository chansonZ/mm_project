#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 4-00:00:00
#SBATCH -J MMLinWorkflow
python wfmmlin.py MMLinear \
    --dataset-name=mm_test \
    --sampling-method=random \
    --test-size=1000 \
    --train-size=3000 \
    --lin-type=0 \
    --slurm-project=b2013262 \
    --runmode=local \
    --workers 2
