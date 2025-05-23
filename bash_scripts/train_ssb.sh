#!/bin/bash
#SBATCH --gres=gpu:rtx2080ti:1


#SBATCH --time=23:00:00
#SBATCH --mem=64G
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH -J w7NaTS
#SBATCH --cpus-per-task=16

#SBATCH --array=0-0
#SBATCH --output outputs/temp.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amit314@umd.edu

source ~/.bashrc

source /fs/nexus-scratch/amit314/venv_torch/bin/activate

DATASET='cub'
AUG_M=30
AUG_N=2
TN=0.5
TP=2.0
SHIFT=1.0
SPLIT=0

python3 CE_tr.py --dataset=${DATASET} --split_idx=${SPLIT} --temperature=${TN}\
                 --lr=0.001 --architecture='resnet50_pretrained' --resnet50_pretrain='places' \
                 --rand_aug_m=${AUG_M} --rand_aug_n=${AUG_N}  --image_size=448 \
                 --batch_size=12 --num_workers=16  --feat_dim=2048\
                 --Tp=${TP} --temperature_scheduling=True --temp_scheduler='gcosm' --shift=${SHIFT} --T=200 \
                      > outputs/work7.out


