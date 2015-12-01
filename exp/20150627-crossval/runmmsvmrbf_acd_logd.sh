#!/bin/bash -l
#SBATCH -A b2015001
#SBATCH -p node
#SBATCH -n 4
#SBATCH -t 10-00:00:00
#SBATCH -J MMWF_ACDLOGD_SVMRBF

# Get directory path of current directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

projbin=$DIR/../../bin
export PATH=$projbin:$PATH

module load openmpi/default

python wfmm.py MMWorkflow \
    --dataset-name=acd_logd \
    --run-id=mmsvmrbf_acdlogd_20151115_013855 \
    --sampling-method=random \
    --replicate-ids=r1,r2,r3 \
    --train-sizes=100,1000,5000,10000,20000,80000,160000,320000,rest \
    --train-method=svmrbf \
    --test-size=50000 \
    --slurm-project=b2015001 \
    --runmode=hpc \
    --workers=27
