sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J pred-DeepRAM-snakemake
#SBATCH -p cpu_p
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 14:00:00
#SBATCH --nice=10000

echo $HOSTNAME

source $HOME/.bashrc
conda activate deepram

python deepRAM/deepRAM.py --data_type DNA --train False --test_data $1  --model_path $2 --word2vec_model $3 --out_file $4 --predict_only True --evaluate_performance False --Embedding True --Conv True --conv_layers 1 --RNN True --RNN_type BiLSTM

EOF