#!/bin/bash
#SBATCH --ntasks=4
#SBATCH --mem=48G
#SBATCH -p hm
#SBATCH --gres=gpu:2
#SBATCH -t 72:00:00
#SBATCH -o "/trinity/home/jwilbers/MedNet/output/out_%j.log"
#SBATCH -e "/trinity/home/jwilbers/MedNet/error/err_%j.log"

module purge
module load Python/3.7.2-GCCcore-8.2.0

source "/trinity/home/jwilbers/MedNet/MedicalNet/venv_mednet_2/bin/activate"

python train_DRF_clas.py --gpu_id 0 --batch_size 5 --num_workers 4 --model_depth 10 --pretrain_path pretrain/resnet_10.pth --n_epochs 100