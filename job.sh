#!/bin/bash
#SBATCH --job-name=tfps
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=230G

nvidia-smi  # 若集群装了NVIDIA驱动，可查看GPU详情