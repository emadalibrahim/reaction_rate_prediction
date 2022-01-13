#!/bin/bash
#SBATCH -N 1
#SBATCH --partition=batch
#SBATCH -J hebo
#SBATCH -o hebo.%J.out
#SBATCH -e hebo.%J.err
#SBATCH --mail-user=Emad.ibrahim@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=10:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=[gpu]


# module load anaconda3/2018.12
# module load cuda/10.1.243
# module load gcc
module load tensorflow/1.14.0-cuda10.0-cudnn7.6-py3.7
source activate chemprop

#run the application:
#python gas_sensing.py --cfg config/runs/net.yaml wandb.use_wandb True --count 20 --resume
wandb agent emadalibrahim/RRP/jpiakejl --count 10
