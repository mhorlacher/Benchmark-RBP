sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J pred-BERT-RBP-snakemake
#SBATCH -p cpu_p
#SBATCH -c 4
#SBATCH --mem=8G
#SBATCH -t 03:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate bert-rbp

mkdir tmp.workdir
cp $1 tmp.workdir/dev.tsv
cp $2/* tmp.workdir/

python3 bert-rbp/examples/run_finetune.py --model_type dna --tokenizer_name dna3 --model_name_or_path tmp.workdir/ --do_predict --data_dir tmp.workdir/ --output_dir tmp.workdir/ --predict_dir tmp.workdir/ --max_seq_length 101 --per_gpu_train_batch_size 32 --overwrite_output --task_name dnaprom

cp tmp.workdir/pred_results.npy $3

EOF