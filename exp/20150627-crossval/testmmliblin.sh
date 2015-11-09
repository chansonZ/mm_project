#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 4-00:00:00
#SBATCH -J MMLinWorkflow
python wfmm.py MMWorkflow \
    --dataset-name=mm_test \
    --run-id=test_mmliblin_$(date +%Y%m%d_%H%M%S) \
    --replicate-id=r1 \
    --sampling-method=random \
    --train-method=liblinear \
    --train-size=3000 \
    --test-size=1000 \
    --lin-type=12 \
    --lin-cost=0.01 \
    --slurm-project=b2013262 \
    --runmode=local \
    --workers 4
