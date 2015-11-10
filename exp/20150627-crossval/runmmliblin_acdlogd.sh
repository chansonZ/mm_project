#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 4-00:00:00
#SBATCH -J MMLinWorkflow
python wfmm.py MMWorkflow \
    --dataset-name=acd_logd \
    --run-id=acdlogd_liblin_$(date +%Y%m%d_%H%M%S) \
    --sampling-method=random \
    --train-method=liblinear \
    --train-sizes=100,1000,5000,10000,20000,rest \
    --test-size=50000 \
    --lin-type=12 \
    --slurm-project=b2013262 \
    --runmode=local \
    --workers=16
