sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J Pysster-snakemake
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH -c 4
#SBATCH --mem=15G
#SBATCH -t 06:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate pysster-same-padding

python scripts/train_Pysster.py --params $1 --save-model-only -o $2 --in-fasta $3 $4

EOF