#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 2
#SBATCH -t 4-00:00:00
#SBATCH -J MMFindCostAcdLogD

# Get directory path of current directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

projbin=$DIR/../../bin
export PATH=$projbin:$PATH

python wffindcost.py\
    CrossValidate\
    --dataset-name=mm_test\
    --folds-count=10\
    --min-height=1\
    --max-height=3\
    --replicate-id=r1\
    --test-size=10\
    --train-size=rest\
    --randomdatasize-mb=10\
    --workers=1\
    --runmode=local
