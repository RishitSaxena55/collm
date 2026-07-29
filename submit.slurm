#!/bin/bash
#SBATCH --job-name=clip_tuned
#SBATCH --partition=a100
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=12
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

source ~/.bashrc
conda activate /home/anirban/rishit.saxena24b/miniconda3/envs/collm

accelerate launch train.py --config configs/clip_stage2_after_stage1.yaml
