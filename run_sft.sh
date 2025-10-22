#!/bin/bash

#SBATCH --job-name=unsloth-sft   # 작업 이름
#SBATCH --nodes=1                  # 1개 노드 사용
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100GB                # 노드 메모리
#SBATCH --time=0-720:00:00         # 최대 실행 시간
#SBATCH --partition=laal_rtx6000      # 사용할 파티션
#SBATCH --output=./S-%x.%j.out     # 로그 파일

# --------------------------------------------------------------------------
# Slurm + Accelerate (compute_environment=SLURM)를 위한 표준 설정입니다.
# --------------------------------------------------------------------------

set -euo pipefail # 스크립트 오류 시 즉시 중단

echo "Job started on $(hostname) at $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID, SLURM_NTASKS: $SLURM_NTASKS"

# (필요시) Conda 또는 가상환경 활성화
# source ~/anaconda3/bin/activate unsloth

# --- 환경 변수 설정 ---
# 1. VRAM 조각화 방지 (OOM 방지)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 2. Tokenizer 데드락 경고 방지
# export TOKENIZERS_PARALLELISM=false

# 3. Triton 캐시를 로컬 디스크(/tmp)에 저장 (NFS 경고 방지)
export TRITON_CACHE_DIR="/tmp/${USER}_triton_cache"

# --- 실행 ---
# 'srun'으로 Python 스크립트를 직접 실행합니다.
# Accelerate 라이브러리가 Slurm 환경 변수를 자동으로 감지하여
# 4-GPU 분산 학습(DDP)을 설정합니다.
echo "Starting SFT training with srun..."

# srun python train_cpt.py

# --------------------------------------------------------------------------
# (참고) 만약 스크립트의 기본값 대신 다른 인수를 사용하고 싶다면,
# 다음과 같이 srun 명령어 뒤에 인수를 추가하면 됩니다:

# N_GPUS=2
export NCCL_DEBUG=warn
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCHELASTIC_ERROR_FILE=$PWD/torchelastic_error.json
export TOKENIZERS_PARALLELISM=false

accelerate launch ./train_sft.py \
    --model_name "unsloth/Qwen3-1.7B" \
    --dataset_path="/home/qqplot/ragfire/rag_sft_dataset.jsonl" \
    --max_seq_length 8192 \
    --num_epochs 1 \
    --learning_rate 2e-4 \
    --per_device_batch_size 4 \
    --grad_accum 2 \
    --experiment_name "unsloth_sft_1.7b"



# torchrun --nproc_per_node=${SLURM_GPUS_PER_NODE:-4} train_cpt.py \
#   --model_name unsloth/Qwen3-8B \
#   --max_seq_length 21848 \
#   --per_device_batch_size 2 \
#   --grad_accum 4 \
#   --learning_rate 5e-5 \
#   --max_steps 12719 \
#   --experiment_name opensloth_cpt_8b

  