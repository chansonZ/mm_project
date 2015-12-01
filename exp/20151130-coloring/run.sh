#!/bin/bash
python wfcoloring.py ColoringWorkflow \
    --runmode=local \
    --slurm-project=2015001 \
    --workers=1
