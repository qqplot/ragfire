#!/bin/bash

#SBATCH --job-name=inference_rag
#SBATCH --nodes=1                    
#SBATCH --gres=gpu:1              
#SBATCH --mem=200GB                     
#SBATCH --cpus-per-task=16            
#SBATCH --output=./S-%x.%j.out   
#SBATCH --time=5:00:00       
#SBATCH --partition=laal_a6000                  

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate rag  
MODEL_NAME="qwen"

# srun python code/langgraph/run_rag_inference.py 
deepspeed --num_gpus 1 run_rag_inference_qwen.py --model_name ${MODEL_NAME}
