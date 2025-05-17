#!/bin/bash
#
#SBATCH --job-name=<job_name>
#SBATCH --output=/<output_directory>/error_definition_generation-%A.out
#SBATCH --error=/<output_directory>/error_definition_generation-%A.err
#SBATCH --mail-user=<mail_address>
#SBATCH --mail-type=ALL
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --export=ALL,MLFLOW_TRACKING_URI=sqlite:////<path_to_mlflow_output_file>.db,MPLCONFIGDIR=/<path_to_mpl_config>,TRITON_CACHE_DIR=/<path_to_triton_cache>

module load cuda/11.1
source activate <path_to_venv>

CUDA_LAUNCH_BLOCKING=1 srun python ../src/syncid.py \
    --dataset <path_to_dataset> \
    --token <huggingface_token> \
    --device cuda \
    --alpha 0.8 \
    --beta 0.7 \
    --error_turn True \
    --batch_size 16 \
    --novelty 0.0 \
    --save_dir <path_to_save_dir> \
    --epochs 50 \
    --visualize True \
    --experiment <experiment_name> \
    #--pretrained
