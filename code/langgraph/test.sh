#!/bin/bash

#SBATCH --job-name=exaone_rag
#SBATCH --nodes=1                    
#SBATCH --gres=gpu:1                  
#SBATCH --mem=100GB                     
#SBATCH --cpus-per-task=8            
#SBATCH --output=./S-%x.%j.out   
#SBATCH --time=5:00:00       
#SBATCH --partition=laal_3090                  

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate rag  

srun python code/langgraph/run_rag_exaone.py 
