#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 4-00:00:00
#SBATCH -J MM_SVMRBF_WF
python wfmm.py MMWorkflow \
    --dataset-name=mm_test \
    --replicate-ids=r1,r2,r3 \
    --sampling-method=random \
    --train-method=svmrbf \
    --train-sizes=1000,2000,3000 \
    --test-size=1000 \
    --lin-type=12 \
    --lin-cost=0.01 \
    --svm-gamma=0.001 \
    --svm-cost=100 \
    --svm-type=3 \
    --svm-kernel-type=2 \
    --slurm-project=b2013262 \
    --runmode=local \
    --workers 1
