import os
import sys
import time
import subprocess
import random
import string
import argparse
import socket

# Check if virtualenv is activated
if subprocess.call("command -v nnUNetv2_train > /dev/null 2>&1", shell=True) != 0:
    print("nnUNetv2_train not found, please activate the environment")
    sys.exit(1)

ailb_cluster = ["aimagelab-srv-10", "ajeje", "carabbaggio", "ailb-login-01", "ailb-login-02", "aimagelab-srv-00"]

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-d', '--dataset', required=True, help='Dataset name')
parser.add_argument('-m', '--model', required=True, help='Model name')
parser.add_argument('-f', '--fold', required=True, help='Fold number')
parser.add_argument('--debug', required=False, default=False, action='store_true')
args = parser.parse_args()

current_hostname = socket.gethostname()
current_path = os.path.realpath(os.path.dirname(__file__))
venv_path = os.path.join(current_path, 'venv', 'bin', 'activate')

raws_path = os.path.join(current_path, 'data', 'nnUNet_raw')
results_path = os.path.join(current_path, 'data', 'nnUNet_results')
preprocessed_path = os.path.join(current_path, 'data', 'nnUNet_preprocessed')

subfolder_name = f'nnUNetTrainer{args.model}__nnUNetPlans__3d_fullres'


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

config_results = os.path.join(results_path, dataset_dir_name, subfolder_name, f'fold_{args.fold}')
checkpoint_final_path = os.path.join(config_results, 'checkpoint_final.pth')

if current_hostname in ailb_cluster:
    # Configuration specific to AImageLab cluster
    print("Detected AImageLab Cluster, please press Ctrl+C if I'm wrong. Running sbatch in 5 seconds")
    cluster = "AImageLab"
    slurm_partition = "boost_usr_prod"
    slurm_account = "grana_maxillo"
    slurm_time = "12:00:00"
    slurm_gres = "gpu:1"
else:
    # Configuration specific to Aries cluster
    print("Detected Aries Cluster, please press Ctrl+C if I'm wrong. Running sbatch in 5 seconds")
    cluster = "Aries"
    slurm_partition = "ice4hpc"
    slurm_account = "cgr"
    slurm_time = "48:00:00"
    slurm_gres = "gpu:a100:1"

job_name = f"{args.model}_{args.dataset}_{args.fold}_nnUNet"
if job_name[0] == '_':
    job_name = f'base{job_name}'
print(f"Cluster: {cluster}")
print(f"Partition: {slurm_partition}")
print(f"Running model {args.model} on dataset {args.dataset} - jobname: {job_name}")
time.sleep(5)

random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=4))
sbatch_file = f"/tmp/{job_name}.sbatch"

# Common configuration for both clusters
with open(sbatch_file, 'w') as f:
    f.write("#!/bin/bash\n")
    f.write(f"#SBATCH --job-name={job_name}\n")
    f.write(f"#SBATCH --output=./slurm_out/{job_name}.out\n")
    f.write(f"#SBATCH --error=./slurm_out/{job_name}.err\n")
    f.write(f"#SBATCH --time={slurm_time}\n")
    f.write(f"#SBATCH --mem=64G\n")
    f.write(f"#SBATCH --cpus-per-task=8\n")
    f.write(f"#SBATCH --gres={slurm_gres}\n")
    f.write(f"#SBATCH --partition={slurm_partition}\n")
    f.write(f"#SBATCH --account={slurm_account}\n")

    if current_hostname in ailb_cluster:
        f.write("#SBATCH --constraint=gpu_A40_48G\n")

    f.write((
        'handle_sigusr() {\n'
            f'if [[ -f "{checkpoint_final_path} ]]; then\n'
                'echo "Found checkpoint_final.pth, the job has completed"\n'
                'sbatch $0\n'
                'rm -f "$flag_file"\n'
            'fi\n'
            'exit 0\n'
        '}'
    ))

    f.write(f"source {venv_path}\n")
    f.write(f"srun nnUNetv2_train {args.dataset} 3d_fullres {args.fold} -tr nnUNetTrainer{args.model} -c &\n")
    f.write("wait\n")
    # f.write(f"srun nnUNetv2_predict -i data/nnUNet_raw/Dataset027_ACDC/imagesTs -o data/void -d 1 -tr nnUNetTrainer{args.model} -c 3d_fullres -f 0 -chk checkpoint_latest.pth\n")

print(f"Submitted sbatch file {sbatch_file}")

if args.debug:
    print("DEBUG MODE, the content of the sbatch file is:")
    with open(sbatch_file, 'r') as f:
        print(f.read())

    os.remove(sbatch_file)
