#!/bin/bash -l
#SBATCH -A b2015001
#SBATCH -p core
#SBATCH -n 8
#SBATCH -t 4-00:00:00
#SBATCH -J MMFindCostSolubility

# Get directory path of current directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

projbin=$DIR/../../bin
export PATH=$projbin:$PATH

# "findcost_solubility_$(date +%Y%m%d_%H%M%S)" \
python wffindcost.py \
    CrossValidate \
    --dataset-name=solubility \
    --run-id=findcost_solubility_20151115_012621 \
    --replicate-ids=r1,r2,r3 \
    --folds-count=10 \
    --min-height=1 \
    --max-height=3 \
    --train-sizes='100,1000,5000,10000,20000,rest' \
    --test-size=5000 \
    --randomdatasize-mb=100 \
    --workers=12 \
    --runmode=local
