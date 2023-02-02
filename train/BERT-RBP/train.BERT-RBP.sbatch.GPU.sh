sbatch --wait << EOF
#!/bin/bash

#SBATCH -o logs/%j.job
#SBATCH -e logs/%j.job
#SBATCH -J BERT-RBP-snakemake
# #SBATCH --exclude=supergpu02pxe,supergpu03pxe,supergpu05,supergpu07,supergpu08
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH -c 4
#SBATCH --mem=15G
#SBATCH -t 16:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate bert-rbp

python3 bert-rbp/examples/run_finetune.py --save_steps 1000000 --model_type dna --tokenizer_name dna3 --model_name_or_path $1 --task_name dnaprom --data_dir $2 --output_dir $2 --do_train --max_seq_length 101 --per_gpu_eval_batch_size 32 --per_gpu_train_batch_size 32 --learning_rate 2e-4 --num_train_epochs 3 --logging_steps 200 --warmup_percent 0.1 --hidden_dropout_prob 0.1 --overwrite_output_dir --weight_decay 0.01 --n_process 4

EOF