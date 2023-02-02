sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J iDeepS-snakemake
#SBATCH -p cpu_p
#SBATCH -c 2
#SBATCH --mem=80G
#SBATCH -t 24:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate ideeps

cp $1 sequences.fa.gz
python iDeepS/ideeps.py --train=True --data_file=sequences.fa.gz --model_dir=./
cp model.pkl $2

EOF