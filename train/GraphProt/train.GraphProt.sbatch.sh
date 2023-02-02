sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J GraphProt-snakemake
#SBATCH -p cpu_p
#SBATCH -c 4
#SBATCH --mem=22G
#SBATCH -t 30:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate graphprot

GraphProt.pl --action train -fasta $1 -negfasta $2
mv ./GraphProt.model $3

EOF