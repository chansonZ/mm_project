#!/bin/bash -l
#SBATCH -A b2013262
#SBATCH -p core
#SBATCH -n 1
#SBATCH -t 7-00:00:00
#SBATCH -J MMFindCostSolubility

# Get directory path of current directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

projbin=$DIR/../../bin
export PATH=$projbin:$PATH

python wffindcost.py \
    CrossValidate \
    --dataset-name=solubility \
    --run-id="findcost_solubility_20151115_012621" \
    --replicate-ids=r1 \
    --folds-count=10 \
    --min-height=1 \
    --max-height=3 \
    --train-sizes='1000' \
    --test-size=5000 \
    --randomdatasize-mb=100 \
    --workers=1 \
    --runmode=hpc
