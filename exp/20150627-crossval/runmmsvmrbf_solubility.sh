#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p node
#SBATCH -n 16
#SBATCH -t 10-00:00:00
#SBATCH -J MMWF_SOLUBILITY_SVMRBF

# Get directory path of current directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

projbin=$DIR/../../bin
export PATH=$projbin:$PATH

python wfmm.py MMWorkflow \
    MMWorkflow \
    --dataset-name=solubility \
    --sampling-method=random \
    --train-method=svmrbf \
    --train-sizes='100,1000,5000,10000,20000,rest' \
    --train-method=svmrbf \
    --test-size=50000 \
    --slurm-project=b2013262 \
    --workers=64 \
    --runmode=hpc \
    --workers=8
