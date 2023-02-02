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
##SBATCH -c 4
#SBATCH --mem=200G
##SBATCH --mem=40G
#SBATCH -t 48:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate multirbp
#conda activate multirbp-cpu
#conda activate multirbp_gpu

python scripts/train_multirbp.py --train-input $1 --output-folder-name $2

EOF


