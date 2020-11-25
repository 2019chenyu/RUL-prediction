#!/bin/bash
#SBATCH -J 1e-2
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --output=out%j.log
#SBATCH --error=err%j.log
#SBATCH -t 30:00:00
#SBATCH --gres=gpu:1
module load anaconda3/5.3.0
python baseline.py --batch_size 5 --lr 0.1 --epochs 50 --seed 10
python baseline.py --batch_size 10 --lr 0.1 --epochs 50 --seed 10
python baseline.py --batch_size 15 --lr 0.1 --epochs 50 --seed 10
python baseline.py --batch_size 20 --lr 0.1 --epochs 50 --seed 10
python baseline.py --batch_size 25 --lr 0.1 --epochs 50 --seed 10

