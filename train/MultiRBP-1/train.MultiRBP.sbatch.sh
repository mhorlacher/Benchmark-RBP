sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J jupyter
##SBATCH -p gpu_p
##SBATCH --gres=gpu:1
##SBATCH --qos=gpu

#SBATCH -p cpu_p
#SBATCH -c 48
#SBATCH --mem=200G
#SBATCH -t 12:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate multirbp-cpu
#conda activate multirbp-gpu

python scripts/train_multirbp.py --train-input $1 --output-folder-name $2

EOF