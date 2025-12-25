#!/bin/bash
#SBATCH --job-name=tfps
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH -p long
#SBATCH -q gpu-12
#SBATCH --time=3-00:00:00
#SBATCH --gres=gpu:4
#SBATCH --mem=230G

source activate TFPS
cd /home/guangyi.chen/causal_group/zijian/dmir_tb/tfps_1/scripts
./electricity.sh