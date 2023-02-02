sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J pred-DeepCLIP-snakemake
#SBATCH -p cpu_p
#SBATCH -c 4
#SBATCH --mem=10G
#SBATCH -t 14:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate deepclip

python deepclip/DeepCLIP.py --runmode predict -P $1 --sequences $2 --predict_output_file $3

EOF