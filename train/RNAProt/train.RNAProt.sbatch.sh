sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J snakemake
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH -c 4
#SBATCH --mem=15G
#SBATCH -t 06:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate rnaprotenv

rnaprot train --in $1 --out $1 --verbose-train

EOF