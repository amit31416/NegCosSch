#!/bin/bash
#SBATCH --gres=gpu:rtx2080ti:1


#SBATCH --time=1-00:00:00
#SBATCH --mem=64G
#SBATCH --partition=scavenger
#SBATCH --account=scavenger
#SBATCH --qos=scavenger
#SBATCH -J w4NsT
#SBATCH --cpus-per-task=16

#SBATCH --array=0-0
#SBATCH --output outputs/temp.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amit314@umd.edu

source ~/.bashrc

source /fs/nexus-scratch/amit314/venv_torch/bin/activate

DATASET='tinyimagenet'
TN=0.1
TP=0.4

for SPLIT in 0 1 2 3 4; do
   python3 SupCon_tr.py --dataset=${DATASET} --split_idx=${SPLIT} --temperature=${TN}\
             --Tp=${TP} --temperature_scheduling=True --temp_scheduler='gcosm' --shift=1.0 --T=200\
                      > outputs/work4.out

done

