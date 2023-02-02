sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J DeepRAM-snakemake
#SBATCH --exclude=supergpu02pxe,supergpu03pxe,supergpu05,supergpu07,supergpu08
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH -c 4
#SBATCH --mem=15G
#SBATCH -t 24:00:00
#SBATCH --nice=10000

echo HOSTNAME=$HOSTNAME

source $HOME/.bashrc
conda activate deepram

python deepRAM/deepRAM.py --data_type DNA --train True --train_data $1 --test_data $2 --model_path $3 --word2vec_model $4 --Embedding True --Conv True --conv_layers 1 --RNN True --RNN_type BiLSTM

EOF