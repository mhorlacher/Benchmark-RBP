sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J multirbp-predict
#SBATCH -p cpu_p
#SBATCH -c 48
#SBATCH --mem=32G
#SBATCH -t 04:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate multirbp-cpu

python -u scripts/eval.py --test-input $1 --model-file $2 --output-csv $3

EOF