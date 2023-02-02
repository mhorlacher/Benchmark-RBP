sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J pred-iDeepS-snakemake
#SBATCH -p cpu_p
#SBATCH -c 4
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate ideeps

cp $1 sequences.fa.gz
cp $2 model.pkl

python iDeepS/ideeps.py --predict=True --data_file=sequences.fa.gz --model_dir=./ --out_file=predictions.txt
cp predictions.txt $3

EOF