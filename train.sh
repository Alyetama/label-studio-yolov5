#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name='train'
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --constraint='gpu_32gb&gpu_v100'
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node=48
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#-------------------------------------
nvidia-smi
#-------------------------------------
module unload python
module load anaconda
module load cuda/10.2
conda activate yolov5
#-------------------------------------
PROJECT_ID=1  # Label Studio project ID
BATCH_SIZE=32
EPOCHS=100
PRETRAINED_WEIGHTS='https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt'
#-------------------------------------
python prepare_dataset.py -p "$PROJECT_ID" || exit 1
#-------------------------------------
python yolov5/train.py \
  --data 'dataset/dataset_config.yml' \
  --batch "$BATCH_SIZE" \
  --epochs "$EPOCHS" \
  --weights "$PRETRAINED_WEIGHTS"
#-------------------------------------
