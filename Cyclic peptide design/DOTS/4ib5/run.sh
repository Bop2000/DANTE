#!/bin/bash

#SBATCH -o ./slurm_result/slurm_test_%j.txt
#SBATCH -e ./slurm_result/slurm_error_%j.txt
#SBATCH --job-name=MCTS
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=40G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=yangtao.chen@helmholtz-munich.de
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --nice=10000

source $HOME/.bashrc

conda activate peptide

python3 cyclic_peptide_binder_design.py
