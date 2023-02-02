sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J pred-GraphProt-snakemake
#SBATCH -p cpu_p
#SBATCH -c 2
#SBATCH --mem=8G
#SBATCH -t 14:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate graphprot

GraphProt.pl --action predict -fasta $1 -model $2 -prefix $3

EOF