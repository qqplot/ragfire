#!/bin/bash
#SBATCH --job-name=72b-deepspeed
#SBATCH --partition=laal_a6000
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=200GB
#SBATCH --output=./S-%x.%j.out
#SBATCH --time=12:00:00

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate rag

export MASTER_PORT=6006
master_addr=$(scontrol show hostnames “$SLURM_JOB_NODELIST” | head -n 1)
export MASTER_ADDR=$master_addr

export USE_TF=0 

BSZ=4 # should be >=num_gpus 
MODEL_PATH="/home/shared/RAG/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"

deepspeed --num_gpus 4 run_model_eval.py --model ${MODEL_PATH} --batch-size ${BSZ} --gen-len 200 --loops 1 --cpu-offload