ailb_cluster=("aimagelab-srv-10" "ajeje" "carabbaggio" "ailb-login-01" "ailb-login-02" "aimagelab-srv-00")

while getopts d:m: flag
do
    case "${flag}" in
        d) dataset=${OPTARG};;
        m) model=${OPTARG};;
    esac
done

current_hostname=$(hostname)

if [[ " ${ailb_cluster[@]} " =~ " ${current_hostname} " ]]; then
  echo "Detected AImageLab Cluster, please press Ctrl+C if I'm wrong. Running sbatch in 5 seconds"
  slurm_partition="boost_usr_prod"
  slurm_account="grana_maxillo"
  slurm_time="12:00:00"
  slurm_gres="gpu:1"
  venv_path="source /work/grana_maxillo/ECCV_MICCAI/U-Mamba/umamba/venv/bin/activate"
else
  echo "Detected Aries Cluster, please press Ctrl+C if I'm wrong. Running sbatch in 5 seconds"
  slurm_partition="ice4hpc"
  slurm_account="cgr"
  slurm_time="48:00:00"
  slurm_gres="gpu:a100:1"
  venv_path="source /unimore_home/llumetti/ECCV_MICCAI/BiUMamba/venv/bin/activate"
fi

job_name=$model"_"$dataset"_nnUNet"
echo "Running model $model on dataset $dataset - jobname: $job_name"
sleep 1


random_string=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 4 | head -n 1)
sbatch_file="/tmp/$job_name.sbatch"
echo "#!/bin/bash" > $sbatch_file
echo "#SBATCH --job-name=$job_name" >> $sbatch_file
echo "#SBATCH --output=./slurm_out/$job_name.out" >> $sbatch_file
echo "#SBATCH --error=./slurm_out/$job_name.err" >> $sbatch_file
echo "#SBATCH --time=$slurm_time" >> $sbatch_file
echo "#SBATCH --mem=64G" >> $sbatch_file
echo "#SBATCH --cpus-per-task=8" >> $sbatch_file
echo "#SBATCH --gres=$slurm_gres" >> $sbatch_file
echo "#SBATCH --partition=$slurm_partition" >> $sbatch_file
echo "#SBATCH --account=$slurm_account" >> $sbatch_file

if [[ " ${ailb_cluster[@]} " =~ " ${current_hostname} " ]]; then
  echo "#SBATCH --constraint gpu_A40_48G" >> $sbatch_file
fi

echo $venv_path >> $sbatch_file
echo "nnUNetv2_train $dataset 3d_fullres 0 -tr nnUNetTrainer"$model"" >> $sbatch_file

sbatch $sbatch_file
echo "Submitted sbatch file $sbatch_file"

