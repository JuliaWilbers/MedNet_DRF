#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --mem=4G
#SBATCH -p express
#SBATCH --gres=gpu:1
#SBATCH -t 1:00:00
#SBATCH -o "/trinity/home/jwilbers/MedNet/test/output/out_%j.log"
#SBATCH -e "/trinity/home/jwilbers/MedNet/test/error/err_%j.log"

module purge
module load Python/3.7.2-GCCcore-8.2.0

source "/trinity/home/jwilbers/MedNet/MedicalNet/venv_mednet_2/bin/activate"

python test_DRF.py --setnr 2 --version 492 --methodnr 2 --resume_path "./trails/DRF_models/method2_a_v49_set_2best.pth.tar" 