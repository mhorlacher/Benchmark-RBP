sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J predict-rnaprotext
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate rnaprotenv

rnaprot predict --in $1 --train-in $2 --mode 1 --out $1

EOF