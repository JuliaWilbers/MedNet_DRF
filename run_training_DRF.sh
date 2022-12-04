#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o "/trinity/home/jwilbers/MedNet/output/out_%j.log"
#SBATCH -e "/trinity/home/jwilbers/MedNet/error/err_%j.log"
module purge
module load Python/3.7.2-GCCcore-8.2.0

source "/trinity/home/jwilbers/MedNet/MedicalNet/venv_mednet_2/bin/activate"

python train_DRF_clas.py --batch_size 7 --num_workers 1 --model_depth 10 --setnr '4' --methodnr '2' --version '53' --augmentation 'True' --n_epochs 50