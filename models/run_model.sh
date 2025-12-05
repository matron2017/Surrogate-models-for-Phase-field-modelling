#!/bin/bash
#SBATCH --job-name=smoke_train
#SBATCH --account=project_2008261
#SBATCH --partition=gputest
#SBATCH --gres=gpu:a100:1
#SBATCH --time=0:15:00
#SBATCH --output=%x_%j.out

export PYTHONPATH="/scratch/project_2008261/alloy_solidification/src:/scratch/project_2008261/rapid_solidification"


/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11 /scratch/project_2008261/rapid_solidification/models/test_models.py
#/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11 /scratch/project_2008261/rapid_solidification/models/backbones/unet_conv_att_cond.py

#/scratch/project_2008261/rapid_solidification/models/unet3d.py
#/scratch/project_2008261/rapid_solidification/physics_ml/bin/python3.11 /scratch/project_2008261/rapid_solidification/models/fno3d.py
