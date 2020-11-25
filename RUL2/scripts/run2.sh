#!/bin/bash
#SBATCH -J tcn
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=out_%j.log
#SBATCH --error=err_%j.log
#SBATCH -t 100:00:00
#SBATCH --gres=gpu:1
module load anaconda3/5.3.0
python main.py --batch_size 5 --epoch 5 --lr 0.001

