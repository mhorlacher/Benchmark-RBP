sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J DeepCLIP-snakemake
#SBATCH -p cpu_p
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 30:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate deepclip

python deepclip/DeepCLIP.py --runmode train --data_split 0.90 0.09 0.01 -n $1 -P $2 --sequences $3 --background_sequences $4 --num_epochs 1 --early_stopping 1

EOF