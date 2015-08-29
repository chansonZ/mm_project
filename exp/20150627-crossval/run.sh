#!/bin/bash

# Get directory path of current directory
DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

projbin=$DIR/../../bin
export PATH=$projbin:$PATH

python workflow.py\
    --task assess_linear\
    --folds-count 10\
    --min-height 1\
    --max-height 3\
    --replicate-id r1\
    --workers 80
