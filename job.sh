#!/bin/bash
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=230G