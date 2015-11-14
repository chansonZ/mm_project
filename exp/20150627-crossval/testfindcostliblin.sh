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

#--run-id=$(date +%Y%m%d_%H%M%S)\
python wffindcost.py\
    CrossValidate\
    --dataset-name=mm_test \
    --run-id='mm_test_findcost_20151113_164511' \
    --replicate-ids=r1,r2,r3 \
    --folds-count=10 \
    --min-height=1 \
    --max-height=3 \
    --train-sizes=100,1000,rest \
    --test-size=10 \
    --randomdatasize-mb=10 \
    --workers=16 \
    --runmode=local
