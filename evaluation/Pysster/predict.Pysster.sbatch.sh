sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J pred-Pysster-snakemake
#SBATCH -p cpu_p
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 06:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate rbpnet

python scripts/predict_Pysster.py $1 --model $2 --output $3

EOF