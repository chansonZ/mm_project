#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 16
#SBATCH -t 4-00:00:00
#SBATCH -J MMFindCostAcdLogD

# Get directory path of current directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

projbin=$DIR/../../bin
export PATH=$projbin:$PATH

python wffindcost.py \
    CrossValidate \
    --dataset-name=acd_logd \
    --run-id='findcost_acd_logd_'$(date +%Y%m%d_%H%M%S) \
    --replicate-ids=r1,r2,r3 \
    --folds-count=10 \
    --min-height=1 \
    --max-height=3 \
    --train-sizes="100,1000,5000,10000,20000,80000,160000,320000,rest" \
    --test-size=50000 \
    --randomdatasize-mb=100 \
    --workers=64 \
    --runmode=hpc
