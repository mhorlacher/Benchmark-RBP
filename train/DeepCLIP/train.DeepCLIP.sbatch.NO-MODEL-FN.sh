sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J DeepCLIP-snakemake
#SBATCH -p cpu_p
#SBATCH -c 5
#SBATCH --mem=16G
#SBATCH -t 30:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate deepclip

python deepclip/DeepCLIP.py --runmode train --data_split 0.90 0.09 0.01 -n $1 --sequences $3 --background_sequences $4 --num_epochs 200 --early_stopping 20

EOF