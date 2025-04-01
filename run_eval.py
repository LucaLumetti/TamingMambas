#!/usr/local/anaconda3/bin/python

import os
import sys
import time
import subprocess
import random
import string
import argparse
import socket

SLEEP_SECONDS = 2

ailb_cluster = ["aimagelab-srv-10", "ajeje", "carabbaggio", "ailb-login-01", "ailb-login-02", "ailb-login-03", "aimagelab-srv-00"]

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True, help='Dataset name')
parser.add_argument('-m', '--model', required=True, help='Model name')
parser.add_argument('-f', '--fold', required=True, help='Fold number')
parser.add_argument('-c', '--config', required=True, help='Train Config: 2d, 3d_fullres')
parser.add_argument('-p', '--plans', required=False, default='nnUNetPlans', help='nnUNetPlans, nnUNetPlansSwinUnet')
parser.add_argument('--high', required=False, default=False, action='store_true')
parser.add_argument('--boost', required=False, default=False, action='store_true')
parser.add_argument('--debug', required=False, default=False, action='store_true')
parser.add_argument('--ngpu', required=False, default=1)

args = parser.parse_args()

if args.boost and args.high:
    print("Cannot set --boost and --high")
    sys.exit(1)

current_hostname = socket.gethostname()
current_path = os.path.realpath(os.path.dirname(__file__))
venv_path = os.path.join(current_path, 'venv', 'bin', 'activate')

raws_path = os.path.join(current_path, 'data', 'nnUNet_raw')
results_path = os.path.join(current_path, 'data', 'nnUNet_results')
preprocessed_path = os.path.join(current_path, 'data', 'nnUNet_preprocessed')

subfolder_name = f'nnUNetTrainer{args.model}__nnUNetPlans__{args.config}'


datasets = os.listdir(raws_path)
dataset_dir_name = None
for dataset in datasets:
    prefix = f"Dataset{int(args.dataset):03d}"
    if not dataset.startswith(prefix):
        continue
    if not os.path.isdir(os.path.join(raws_path, dataset)):
        continue
    dataset_dir_name = dataset
    break

if dataset_dir_name is None:
    print(f'I cannot find the dataset folder. I looked for "Dataset{int(args.dataset):03d}*" inside {raws_path}\nFolders: {datasets}')
    sys.exit(1)

#imagesTs = os.path.join(raws_path, dataset_dir_name, 'imagesTs')
imagesTs = '/work/grana_maxillo/Mamba3DMedModels/data/nnUNet_raw/Dataset001_BrainTumour/imagesTs'
#imagesTs = '/work/grana_maxillo/Mamba3DMedModels/data/test_sets/Radboud/volumes'
inferTs = os.path.join(raws_path, dataset_dir_name, f'inferTs_{args.model}_{args.fold}')
print(f'ImagesTs: {imagesTs}')
print(f'InferTs: {inferTs}')
config_results = os.path.join(results_path, dataset_dir_name, subfolder_name, f'fold_{args.fold}')
checkpoint_final_path = os.path.join(config_results, 'checkpoint_final.pth')

continue_training = ""
if os.path.exists(config_results) or True:
    continue_training = "--c"

if current_hostname in ailb_cluster:
    # Configuration specific to AImageLab cluster
    print(f"Detected AImageLab Cluster, please press Ctrl+C if I'm wrong. Running sbatch in {SLEEP_SECONDS} seconds")
    cluster = "AImageLab"
    slurm_account = "grana_maxillo"
    slurm_partition = "all_usr_prod"
    slurm_time = "8:00:00"
    slurm_gres = "gpu:1"
    if args.boost:
        slurm_partition = "boost_usr_prod"
        slurm_time = "8:00:00"
else:
    # Configuration specific to Aries cluster
    print(f"Detected Aries Cluster, please press Ctrl+C if I'm wrong. Running sbatch in {SLEEP_SECONDS} seconds")
    cluster = "Aries"
    slurm_partition = "ice4hpc"
    slurm_account = "cgr"
    slurm_time = "72:00:00"
    slurm_gres = "gpu:1g.20gb:1"
    if args.boost or args.high:
        slurm_gres = "gpu:a100:1"

job_name = f"Inference_{args.model}_{args.dataset}_{args.fold}_{args.plans}"
if job_name[0] == '_':
    job_name = f'base{job_name}'
print(f"Cluster: {cluster}")
print(f"Partition: {slurm_partition}")
print(f"Running model {args.model} {args.plans} on dataset {args.dataset} - jobname: {job_name}")
time.sleep(SLEEP_SECONDS)

random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
sbatch_file = f"sbatch_files/{job_name}.sbatch"

os.makedirs('sbatch_files', exist_ok=True)

# Common configuration for both clusters
with open(sbatch_file, 'w') as f:
    f.write("#!/bin/bash\n")
    f.write(f"#SBATCH --job-name={job_name}\n")
    f.write(f"#SBATCH --output=./slurm_out/{job_name}.out\n")
    f.write(f"#SBATCH --error=./slurm_out/{job_name}.err\n")
    f.write(f"#SBATCH --time={slurm_time}\n")
    f.write(f"#SBATCH --mem=150G\n")
    f.write(f"#SBATCH --cpus-per-task=8\n")
    f.write(f"#SBATCH --gres={slurm_gres}\n")
    f.write(f"#SBATCH --partition={slurm_partition}\n")
    f.write(f"#SBATCH --account={slurm_account}\n")
    # f.write(f"#SBATCH --signal=B:SIGUSR1@10\n")
    # f.write(f"#SBATCH --exclude=vegeta,carabbaggio,lurcanio\n")

    if current_hostname in ailb_cluster and args.boost:
        #f.write("#SBATCH --constraint=gpu_A40_48G\n")
        pass
    if current_hostname in ailb_cluster and args.high:
        f.write("#SBATCH --constraint=\"gpu_RTX6000_24G|gpu_RTXA5000_24G|gpu_A40_48G|gpu_L40S_48G\" ")

    f.write('\n')

    # f.write(f"export WANDB__SERVICE_WAIT=300\n")
    # f.write(f"export WANDB_MODE=online\n")
    # f.write(f"export TORCH_DISTRIBUTED_DEBUG=detail\n")
    f.write(f"source {venv_path}\n\n")
    #f.write(f'\tsrun nnUNetv2_predict -i {imagesTs} -o {inferTs} -d {args.dataset} -tr nnUNetTrainer{args.model} -p {args.plans} -c {args.config} -f {args.fold} -chk checkpoint_final.pth\n')
    f.write(f'\tsrun python3 /work/grana_maxillo/Mamba3DMedModels/evaluation/Dataset001_BrainTumour/eval.py {args.model} {args.fold} {args.config}\n')


if args.debug:
    print("DEBUG MODE, file has not been sbatched, the content of the sbatch file is:")
    with open(sbatch_file, 'r') as f:
        print(f.read())
    os.remove(sbatch_file)
else:
    subprocess.call(f"sbatch {sbatch_file}", shell=True)
    print(f"Submitted sbatch file {sbatch_file}")
    subprocess.call(f"squeue --partition={slurm_partition}", shell=True)

