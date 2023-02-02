sbatch --wait <<- EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J snakemake
#SBATCH -p gpu_p
#SBATCH --exclude=supergpu05,supergpu07,supergpu08
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 06:00:00
#SBATCH --nice=10000

mkdir logs

source $HOME/.bashrc
conda activate prismnet

python -u ../../methods/PrismNet/main.py --train --eval --lr 0.001 --data_dir $1 --p_name all.train --out_dir $1

EOF
