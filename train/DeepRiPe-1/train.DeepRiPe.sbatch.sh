sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J deepripe
#SBATCH -p cpu_p

## SBATCH -p gpu_p
## SBATCH --gres=gpu:1
## SBATCH --qos=gpu

#SBATCH -c 48
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate multiresbind-2

python scripts/train_deepripe.py --train-input $1 --train-input-region $2 --output-folder-name $3

EOF