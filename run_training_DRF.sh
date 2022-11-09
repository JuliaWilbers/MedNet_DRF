#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=12G
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -t 12:00:00
#SBATCH -o "/trinity/home/jwilbers/MedNet/output/out_%j.log"
#SBATCH -e "/trinity/home/jwilbers/MedNet/error/err_%j.log"

module purge
module load Python/3.7.2-GCCcore-8.2.0

source "/trinity/home/jwilbers/MedNet/MedicalNet/venv_mednet_2/bin/activate"

python train_DRF_clas.py  --gpu_id 2 --batch_size 3 --num_workers 1 --model_depth 10 --setnr '1' --methodnr '3' --version '12' --augmentation 'True' --n_epochs 300