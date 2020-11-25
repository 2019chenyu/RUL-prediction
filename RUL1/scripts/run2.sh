#!/bin/bash
#SBATCH -J 1e-2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=out%j.log
#SBATCH --error=err%j.log
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:1
module load anaconda3/5.3.0
python 1dcov+lstm+attention.py --batch_size 5 --lr 0.001 --epochs 2 --seed 10 \
--weight_decay 1e-4 --sequence_length 100 --lstm1 30 --lstm2 100 --dense1 50 \
--filters1 16 --filters2 32 --dropout1 0.6 --dropout2 0.6


