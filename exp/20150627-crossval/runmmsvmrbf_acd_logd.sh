#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 1
#SBATCH -t 10-00:00:00
#SBATCH -J MMWF_SVM_ACDLOGD

# Get directory path of current directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

projbin=$DIR/../../bin
export PATH=$projbin:$PATH

module load openmpi/default

python wfmm.py MMWorkflow \
    --dataset-name=acd_logd \
    --run-id=mmsvmrbf_acdlogd_20151115_013855 \
    --sampling-method=random \
    --replicate-ids=r3 \
    --train-sizes=1000 \
    --train-method=svmrbf \
    --test-size=50000 \
    --slurm-project=b2013262 \
    --runmode=hpc \
    --workers=1
