#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH -p short
#SBATCH --gres=gpu:1
#SBATCH -t 24:00:00
#SBATCH -o "/trinity/home/jwilbers/MedNet/test/output/out_%j.log"
#SBATCH -e "/trinity/home/jwilbers/MedNet/test/error/err_%j.log"

module purge
module load Python/3.7.2-GCCcore-8.2.0

source "/trinity/home/jwilbers/MedNet/MedicalNet/venv_mednet_2/bin/activate"

python test_DRF.py --gpu_id 0 --setnr 1 --version 20 --methodnr 3 --resume_path "./trails/DRF_models/resnet_10_set1_method3_v22_best.pth.tar" 