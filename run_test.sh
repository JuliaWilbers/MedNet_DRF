#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=14G
#SBATCH -p long
#SBATCH --gres=gpu:1
#SBATCH -t 48:00:00
#SBATCH -o "/trinity/home/jwilbers/MedNet/test/output/out_%j.log"
#SBATCH -e "/trinity/home/jwilbers/MedNet/test/error/err_%j.log"

module purge
module load Python/3.7.2-GCCcore-8.2.0

source "/trinity/home/jwilbers/MedNet/MedicalNet/venv_mednet_2/bin/activate"

python test_DRF.py --gpu_id 0 --resume_path trails/models/resnet_10_epoch_99_batch_0_method3.pth.tar  