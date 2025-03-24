#!/bin/bash

#SBATCH --job-name=rag_fe_exaone
#SBATCH --nodes=1                    
#SBATCH --gres=gpu:1           
#SBATCH --mem=100GB                     
#SBATCH --cpus-per-task=6         
#SBATCH --output=./S-%x.%j.log   
#SBATCH --time=2:00:00       
#SBATCH --partition=laal_3090                  

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate rag  
MODEL_NAME="exaone"

python run_rag_inference_exaone.py
# deepspeed --num_gpus 2 run_rag_inference_qwen.py --model_name ${MODEL_NAME}
# 