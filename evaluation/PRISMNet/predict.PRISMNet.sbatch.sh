sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J predict-prismnet
#SBATCH -p cpu_p
#SBATCH -c 4
#SBATCH --mem=10G
#SBATCH -t 02:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate prismnet

python -u ../../methods/PrismNet/main.py --load_best --infer --infer_file $1 --p_name all.train --out_dir $2

cp $3 $4

EOF