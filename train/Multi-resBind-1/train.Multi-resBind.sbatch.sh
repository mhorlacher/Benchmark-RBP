sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J multiresbind
#SBATCH -p cpu_p
#SBATCH -c 48

##SBATCH -p gpu_p
##SBATCH --gres=gpu:1
##SBATCH --qos=gpu
##SBATCH -c 4

#SBATCH --mem=48G
#SBATCH -t 2-00:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate multiresbind-2 # multiresbind-gpu

python scripts/train.py --train-input $1 --train-input-region $2 --output-folder-name $3

EOF
