#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 8
#SBATCH -t 10-00:00:00
#SBATCH -J MMWF_ACDLOGD_SVMRBF

# Get directory path of current directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

projbin=$DIR/../../bin
export PATH=$projbin:$PATH

python wfmm.py MMWorkflow \
    --dataset-name=acd_logd \
    --run-id=mainwfrun_acd_logd_20151115_013855 \
    --sampling-method=random \
    --replicate-ids=r1,r2,r3 \
    --train-sizes=100,1000,5000,10000,20000,rest \
    --train-method=svmrbf \
    --test-size=50000 \
    --slurm-project=b2013262 \
    --runmode=hpc \
    --workers=1
