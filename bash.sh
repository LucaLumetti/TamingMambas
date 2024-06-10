#!/bin/bash
. /usr/local/anaconda3/etc/profile.d/conda.sh
module unload cuda/12.1
module load cuda/11.8
conda deactivate
source /work/grana_maxillo/Mamba3DMedModels/venv/bin/activate 
which pip
export WANDB_MODE=disabled
srun -Q --immediate=1000 --mem=10G --partition=all_serial --gres=gpu:1 --nodes=1 --time 60:00 --pty --account=grana_maxillo bash