#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 16
#SBATCH -t 4-00:00:00
#SBATCH -J MMFindCostSol

# Get directory path of current directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

projbin=$DIR/../../bin
export PATH=$projbin:$PATH

python wffindcost.py\
    CrossValidate\
    --dataset-name=solubility \
    --folds-count=10\
    --min-height=1\
    --max-height=3\
    --replicate-id=r1\
    --test-size=5000\
    --train-sizes='100,1000,5000,10000,20000,rest'\
    --randomdatasize-mb=100\
    --workers=64\
    --runmode hpc
