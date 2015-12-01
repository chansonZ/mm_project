#!/bin/bash -l
#SBATCH -A b2015001
#SBATCH -p node
#SBATCH -n 4
#SBATCH -t 4-12:00:00
#SBATCH -J MMWF_SOLUBILITY_SVMRBF

# Get directory path of current directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

projbin=$DIR/../../bin
export PATH=$projbin:$PATH

module load openmpi/default

python wfmm.py MMWorkflow \
    --dataset-name=solubility \
    --run-id=mmsvmrbf_solubility_20151110_001640 \
    --sampling-method=random \
    --replicate-ids=r1,r2,r3 \
    --train-sizes=100,1000,5000,10000,20000,rest \
    --train-method=svmrbf \
    --test-size=5000 \
    --parallel-svm-train \
    --slurm-project=b2015001 \
    --runmode=hpc \
    --workers=18
