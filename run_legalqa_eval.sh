#!/bin/bash
#SBATCH --job-name=evaluation_results_400_sample_cpt1.7b
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --time=0-12:00:00
#SBATCH --partition=laal_3090           # MOD: 필요시 파티션 이름 확인
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err

# ────────────────────────────────────────────────
# 0. 사용자 정의 파라미터 (사용자 환경에 맞게 수정해주세요)
# ────────────────────────────────────────────────
EVAL_SCRIPT="/home/qqplot/ragfire/run_legalqa_eval.py"  # MOD: 실행할 파이썬 스크립트 경로
INPUT_FILE="/home/qqplot/ragfire/data/Legal_QA_Multi-Hop.xlsx" # MOD: 평가 데이터 Excel 파일 경로
OUT_DIR_BASE="/home/qqplot/ragfire/legalqa_eval_results" # MOD: 결과가 저장될 기본 폴더
OPENAI_KEY_FILE=""      # MOD: OpenAI API 키 파일 경로 (필요시)

# 평가할 모델 목록
MODELS=(
  #"naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"
  #"Qwen/Qwen3-8B"
  # "Qwen/Qwen3-1.7B"
  "unsloth/Qwen3-1.7B"
  # "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct"
  # "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
  #"openai:gpt-4o"
  # "naver-hyperclovax/HyperCLOVAX-SEED-Think-14B"
)

# ────────────────────────────────────────────────
# 1. (옵션) 가상 환경 활성화
# ────────────────────────────────────────────────
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate your-env-name # MOD: 사용하는 가상환경 이름

# ────────────────────────────────────────────────
# 2. 작업 정보 출력
# ────────────────────────────────────────────────
echo "============================================="
echo " LegalQA Evaluation Batch Job"
echo "  SCRIPT      = $EVAL_SCRIPT"
echo "  INPUT FILE  = $INPUT_FILE"
echo "  OUTPUT DIR  = $OUT_DIR_BASE"
echo "  MODELS      = ${MODELS[*]}"
echo "============================================="

# ────────────────────────────────────────────────
# 3. 모델 루프 실행
# ────────────────────────────────────────────────
for MODEL in "${MODELS[@]}"; do
  echo
  echo "========== Evaluating model: $MODEL =========="

  # --- ① 출력 폴더 및 파일명 설정 ---
  # 모델 이름의 '/'를 '_'로 변경하여 폴더명으로 사용 (예: Qwen/Qwen3-8B -> Qwen_Qwen3-8B)
  MODEL_NAME_SAFE=$(echo "$MODEL" | tr '/' '_')
  OUT_DIR="$OUT_DIR_BASE/$MODEL_NAME_SAFE"
  OUT_FILE="$OUT_DIR/evaluation_results_cpt1.7b_400_no_sample.xlsx"
  mkdir -p "$OUT_DIR"

  echo "  Results will be saved to: $OUT_FILE"

  # --- ② 실행 커맨드 생성 ---
  CMD=(python "$EVAL_SCRIPT"
       --input_file    "$INPUT_FILE"
       --output_file   "$OUT_FILE"
       --model_id      "$MODEL"
  )

  # --- ③ (옵션) OpenAI 모델일 경우 키 파일 인자 추가 ---
  # is_openai_model() 함수와 동일한 로직
  if [[ ${MODEL,,} == "openai:"* ]]; then
    # -z : 문자열이 null이면 참
    if [ ! -z "$OPENAI_KEY_FILE" ]; then
        CMD+=(--openai_key_file "$OPENAI_KEY_FILE")
        echo "  Using OpenAI key file: $OPENAI_KEY_FILE"
    else
        echo "  Warning: OpenAI model specified, but no key file path provided."
    fi
  fi

  # --- ④ 실행 ---
  # CMD 배열의 모든 원소를 각각의 인자로 전달하기 위해 "${CMD[@]}" 사용
  "${CMD[@]}"

  echo
  echo "--- Finished model: $MODEL ---"
done

echo
echo "All model evaluations completed."