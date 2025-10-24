import os
import pandas as pd
import time
import random
import re
import json
import numpy as np
import argparse
import torch
import gc
import sys
from pathlib import Path
from typing import List, Dict
from peft import PeftModel

# Hugging Face Imports
from transformers import AutoModelForCausalLM, AutoTokenizer

# Conditional imports for OpenAI
try:
    import openai
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("Warning: openai library not found. OpenAI API calls will be disabled.")

# ======================================================================================
# Configuration and Constants
# ======================================================================================

# Column Name Mapping (사용하시는 Excel 파일의 컬럼 이름에 맞게 확인해주세요)
COL_DOC_A = 'chapter_body_from\n(문서 A)'
COL_DOC_B = 'chapter_body_to\n(문서 B)'
COL_QUESTION = 'question'
COL_ANS_A = 'answer_scenario_A'
COL_ANS_B = 'answer_scenario_B'

# Prompt Templates (기존과 동일)
SYSTEM_PROMPT_BASE = """
당신은 주어진 객관식 질문에 답하는 평가자입니다.
질문과 보기를 주의 깊게 읽고 가장 적절한 답을 선택하십시오.
당신의 응답은 반드시 정답에 해당하는 옵션의 숫자 하나여야 합니다 (예: 1, 2, 3, 4, 또는 5). 다른 설명은 포함하지 마십시오.
"""

SYSTEM_PROMPT_CONTEXT = """
당신은 주어진 컨텍스트(문서)를 바탕으로 객관식 질문에 답하는 평가자입니다.
답변은 반드시 제공된 컨텍스트 내용에만 근거해야 합니다.

**중요 지침:**
만약 제공된 컨텍스트 내에서 질문에 대한 답을 찾을 수 없다면, 정보를 알 수 없거나 판단할 수 없음을 나타내는 보기('알 수 없음', '정보 없음' 등)를 반드시 선택해야 합니다.

당신의 응답은 반드시 정답에 해당하는 옵션의 숫자 하나여야 합니다 (예: 1, 2, 3, 4, 또는 5). 다른 설명은 포함하지 마십시오.
"""

USER_PROMPT_TEMPLATE = """
{context_section}
[질문]
{question}

[보기]
{options_text}

[당신의 답변 (숫자만)]
"""

# ======================================================================================
# Model Handling Utilities (RAG 스크립트 참조하여 통합)
# ======================================================================================

def is_openai_model(name: str) -> bool:
    """모델명이 'openai:'로 시작하는지 확인합니다."""
    return name.lower().startswith("openai:")

def ensure_openai_key(key_file: str):
    """OpenAI API 키를 파일 또는 환경 변수에서 로드합니다."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI 라이브러리가 설치되지 않았습니다.")
        
    key = None
    if key_file and Path(key_file).exists():
        key = Path(key_file).read_text().strip()
    if not key:
        key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("❌ OpenAI API 키를 찾을 수 없습니다. --openai_key_file 또는 OPENAI_API_KEY 환경 변수를 확인하세요.")
    
    return OpenAI(api_key=key)

def load_llm(model_name: str):
    """Hugging Face 모델을 로드합니다."""
    print(f"· Loading Hugging Face model: {model_name}")
    
    # 최적의 dtype 설정 (bfloat16 우선)
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32
    
    ADAPTER_DIR = "/home/qqplot/ragfire/outputs/opensloth_cpt_ddp4_bk/checkpoint-400"  # 위 스크린샷의 경로
    BASE_MODEL = "unsloth/Qwen3-1.7B"  # 학습에 쓴 베이스 모델 (예: "meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    # model = AutoModelForCausalLM.from_pretrained(
    #     model_name,
    #     torch_dtype=dtype,
    #     device_map="auto" if torch.cuda.is_available() else "cpu",
    #     trust_remote_code=True,
    # )
    # tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    # Pad 토큰 설정 (필수)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    return model, tokenizer

# ======================================================================================
# Inference Functions (RAG 스크립트 참조하여 안정성 강화)
# ======================================================================================

def run_generation_hf(model, tokenizer, messages, max_new_tokens=10) -> str:
    """Hugging Face 모델 추론 실행."""
    # Chat Template 적용을 위한 인자 설정
    template_args = {
        "conversation": messages,  # <<< 'messages'를 'conversation'으로 수정
        "add_generation_prompt": True,
        "tokenize": True,
        "enable_thinking": False,
        "return_tensors": "pt"
    }

    # Qwen 모델의 경우 'think mode' 비활성화
    # model.name_or_path를 통해 현재 로드된 모델의 ID를 확인합니다.
    if hasattr(model, 'name_or_path') and 'qwen' in model.name_or_path.lower():
        template_args['enable_thinking'] = False
    
    # Chat Template 적용
    input_ids = tokenizer.apply_chat_template(
        **template_args
    ).to(model.device)

    input_len = input_ids.shape[1]

    # temperature=0.7; top_p=0.90; top_k=20
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False, # temperature=0과 동일 (결정론적 출력)
            # do_sample=True,
            # temperature=temperature,
            # top_p=top_p,
            # top_k=top_k,            
        )
    
    # 생성된 부분만 디코딩
    generated_tokens = output[0, input_len:]
    out_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return out_text.strip()

def safe_run_generation_hf(model, tokenizer, messages,
                           max_new_tokens=10,
                           placeholder="[OOM-SKIPPED]") -> str:
    """Hugging Face 모델 추론 래퍼 (OOM 에러 처리)."""
    try:
        return run_generation_hf(model, tokenizer, messages, max_new_tokens)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower():
            print("⚠️ OOM detected - skipping this query", file=sys.stderr)
            # 메모리 정리
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            return placeholder
        raise

def safe_openai_generation(
    client,
    model_name: str,
    messages: List[Dict],
    max_new_tokens: int = 10,
) -> str:
    """OpenAI API 호출 래퍼 (Retry 로직 포함)."""
    if not OPENAI_AVAILABLE:
        return "Error: OpenAI library not available"

    # 재시도 가능 예외 정의 (RAG 스크립트 참조)
    _retryable_names = [
        "RateLimitError", "APIConnectionError", "InternalServerError", 
        "APITimeoutError", "Timeout", "APIError",
    ]
    RETRYABLE_EXC = tuple(
        exc for name in _retryable_names
        if (exc := getattr(openai, name, None))
        if isinstance(exc, type) and issubclass(exc, BaseException)
    )
    
    min_backoff, max_backoff = 1.0, 60.0
    backoff = min_backoff
    attempt = 1

    while True:
        try:
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=0.0,
                max_tokens=max_new_tokens,
            )
            return resp.choices[0].message.content.strip()

        except RETRYABLE_EXC as e:
            wait = random.uniform(backoff * 0.8, backoff * 1.2)
            print(
                f"⚠️ {type(e).__name__} - retry #{attempt} after {wait:.1f}s",
                file=sys.stderr,
            )
            time.sleep(wait)
            backoff = min(backoff * 2, max_backoff)
            attempt += 1

        except Exception as e:
            # 기타 예외 (인증 오류 등)
            print(f"❌ OpenAI 요청 실패 (재시도 불가): {e}", file=sys.stderr)
            return f"API Error: {type(e).__name__}"

def extract_choice_number(text: str) -> int | str:
    """LLM 응답에서 1-5 사이의 숫자만 추출 (Qwen 모델의 <think> 태그 등 복잡한 출력에 대응하도록 수정)"""
    text = text.strip()

    # 1. 가장 먼저, 응답이 숫자 하나로만 구성된 경우 (가장 이상적인 케이스)
    if re.match(r"^[1-5]$", text):
        return int(text)

    # 2. <think>...</think> 와 같은 불필요한 부분을 제거
    # Qwen 모델이 종종 이런 출력을 생성하므로, re.DOTALL 플래그로 여러 줄에 걸친 태그도 제거합니다.
    cleaned_text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 3. 정리된 텍스트에서 1-5 사이의 숫자를 모두 찾습니다.
    matches = re.findall(r"([1-5])", cleaned_text)

    # 4. 숫자가 하나 이상 찾아졌다면, '가장 마지막에 나온 숫자'를 정답으로 간주합니다.
    #    (예: "1번 같지만... 최종 정답은 3번입니다." -> 3을 선택)
    if matches:
        return int(matches[-1])

    # 5. 시스템 에러 메시지인 경우 그대로 반환합니다.
    if text.startswith(("[OOM-SKIPPED]", "API Error:", "Error:")):
        return text

    # 6. 위 모든 경우에 해당하지 않으면 유효하지 않은 형식으로 처리합니다.
    print(f"경고: LLM이 유효하지 않은 형식으로 응답했습니다: '{text}'")
    return text


def call_llm(model_id, system_prompt, user_prompt, hf_model=None, hf_tokenizer=None, openai_client=None):
    """
    통합 LLM 호출 함수 (OpenAI 또는 Hugging Face 모델 디스패치).
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    # <<< 수정된 부분 시작 >>>
    # Qwen 모델의 think mode를 비활성화했으므로, 더 이상 긴 출력을 위한 예외 처리가 필요 없습니다.
    # 모든 모델에 대해 짧고 일관된 max_tokens를 사용합니다.
    max_tokens_for_generation = 20
    # <<< 수정된 부분 끝 >>>

    start_time = time.time()

    if is_openai_model(model_id):
        if openai_client is None:
            result = "Error: OpenAI client not initialized"
        else:
            openai_model_name = model_id.split(":", 1)[1]
            result = safe_openai_generation(openai_client, openai_model_name, messages, max_new_tokens=max_tokens_for_generation)
    else:
        if hf_model is None or hf_tokenizer is None:
            result = "Error: HF model not loaded"
        else:
            result = safe_run_generation_hf(hf_model, hf_tokenizer, messages, max_new_tokens=max_tokens_for_generation)

    latency = time.time() - start_time

    # 결과 검증 및 파싱 (공통)
    parsed_result = extract_choice_number(result)
    return parsed_result, latency
# ======================================================================================
# Utility Functions (LegalQA 스크립트에서 가져옴)
# ======================================================================================

def shuffle_options(row, answer_col_name):
    """옵션 1-5를 셔플하고 정답 인덱스를 조정합니다."""
    options = []
    for i in range(1, 6):
        option_text = row.get(f'option_{i}')
        if pd.notna(option_text) and str(option_text).strip() != "":
            options.append((i, str(option_text)))

    if not options:
        return None, None, None

    original_answer_index = row.get(answer_col_name)
    try:
        # 정답이 float으로 읽힐 수 있으므로 int로 안전하게 변환
        original_answer_index = int(original_answer_index)
    except (ValueError, TypeError):
        print(f"경고: 정답 '{original_answer_index}'이(가) 유효한 숫자가 아닙니다. ({answer_col_name})")
        return None, None, None

    random.shuffle(options)

    shuffled_options_text = ""
    shuffled_options_dict = {}
    new_answer_index = -1

    for i, (original_index, text) in enumerate(options):
        new_index = i + 1
        shuffled_options_text += f"{new_index}. {text}\n"
        shuffled_options_dict[new_index] = text

        if original_index == original_answer_index:
            new_answer_index = new_index

    if new_answer_index == -1:
        print(f"경고: 원본 정답 번호 {original_answer_index}를 옵션 목록에서 찾을 수 없습니다.")
        return None, None, None

    return shuffled_options_text.strip(), new_answer_index, shuffled_options_dict

def save_dataframe(df, path):
    """데이터프레임을 안전하게 저장합니다 (복잡한 타입은 JSON 문자열로 변환)."""
    try:
        df_save = df.copy()
        # 엑셀 저장을 위해 딕셔너리/리스트 타입을 JSON 문자열로 변환
        for col in df_save.columns:
            if df_save[col].apply(lambda x: isinstance(x, (dict, list))).any():
                df_save[col] = df_save[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (dict, list)) else x)

        # openpyxl 엔진 필요
        df_save.to_excel(path, index=False)
        print(f"-> 진행 상황이 '{path}'에 저장되었습니다.")
        return True
    except Exception as e:
        print(f"-> 파일 저장 중 오류 발생: {e}")
        if "No module named 'openpyxl'" in str(e):
             print("-> 'openpyxl'이 필요합니다. (pip install openpyxl)")
        return False

# ======================================================================================
# Main Execution Function
# ======================================================================================

def main(args):
    # 1. 모델 초기화
    hf_model, hf_tokenizer, openai_client = None, None, None
    
    if is_openai_model(args.model_id):
        print("Initializing OpenAI Client...")
        try:
            openai_client = ensure_openai_key(args.openai_key_file)
        except RuntimeError as e:
            print(e)
            return
    else:
        print("Initializing Hugging Face Model...")
        try:
            hf_model, hf_tokenizer = load_llm(args.model_id)
        except Exception as e:
            print(f"모델 로드 실패: {e}")
            return

    # 2. 데이터 로딩 및 준비 (작업 재개 지원)
    if os.path.exists(args.output_file):
        print(f"기존 작업 파일 '{args.output_file}'을(를) 불러와 작업을 재개합니다.")
        df = pd.read_excel(args.output_file)
    elif os.path.exists(args.input_file):
        print(f"새로운 작업 파일 '{args.input_file}'을(를) 불러옵니다.")
        try:
            df = pd.read_excel(args.input_file)
        except Exception as e:
            print(f"파일 로드 오류: {e}")
            return
    else:
        print(f"오류: 입력 파일 경로 '{args.input_file}'를 확인해주세요.")
        return

    # 필수 열 확인
    required_cols = [COL_DOC_A, COL_DOC_B, COL_QUESTION, COL_ANS_A, COL_ANS_B] + [f'option_{i}' for i in range(1, 6)]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"오류: 다음 필수 열이 데이터프레임에 없습니다: {missing_cols}")
        return

    # 시나리오 정의
    scenarios = [
        {
            'name': 'zero_shot',
            'system_prompt': SYSTEM_PROMPT_BASE,
            'answer_col': COL_ANS_B, # Zero-shot은 일반적으로 B의 정답 사용
            'context_cols': []
        },
        {
            'name': 'scenario_A',
            'system_prompt': SYSTEM_PROMPT_CONTEXT,
            'answer_col': COL_ANS_A,
            'context_cols': [(COL_DOC_A, "문서 A")]
        },
        {
            'name': 'scenario_B',
            'system_prompt': SYSTEM_PROMPT_CONTEXT,
            'answer_col': COL_ANS_B,
            'context_cols': [(COL_DOC_A, "문서 A"), (COL_DOC_B, "문서 B")]
        }
    ]

    # 결과 열 초기화
    for scenario in scenarios:
        inf_col = f"{scenario['name']}_inference"
        detail_col = f"{scenario['name']}_details"
        latency_col = f"{scenario['name']}_latency"
        if inf_col not in df.columns:
            df[inf_col] = np.nan
        if detail_col not in df.columns:
            df[detail_col] = np.nan
        if latency_col not in df.columns:
            df[latency_col] = np.nan

    print(f"총 {len(df)}개의 행을 대상으로 작업을 시작합니다. 모델: {args.model_id}")

    # 3. 순회 및 처리
    # 재현성을 위해 시드 고정
    random.seed(42)
    
    for index, row in df.iterrows():
        print(f"\n===== [ 행 {index+1}/{len(df)} ] 처리 시작 =====")

        question_text = row.get(COL_QUESTION)
        if pd.isna(question_text) or not str(question_text).strip():
            print("질문이 없어 이 행은 건너뜁니다.")
            continue

        # 시나리오 처리
        for scenario in scenarios:
            scenario_name = scenario['name']
            inf_col = f"{scenario_name}_inference"
            detail_col = f"{scenario_name}_details"
            latency_col = f"{scenario_name}_latency"

            # 이미 처리된 경우 건너뛰기 (NaN이 아니면 처리된 것으로 간주)
            if pd.notna(df.loc[index, inf_col]):
                print(f"--- [{scenario_name}] 이미 처리됨. 건너뜁니다. ---")
                continue

            print(f"--- [{scenario_name}] 처리 시작 ---")

            # 1. 컨텍스트 준비
            context_parts = []
            for col_name, doc_label in scenario['context_cols']:
                doc_content = row.get(col_name)
                content = str(doc_content) if pd.notna(doc_content) else "(내용 없음)"
                context_parts.append(f"[{doc_label}]\n{content}\n")

            context_section = "\n".join(context_parts) if context_parts else ""

            # 2. 옵션 셔플링
            options_text, correct_answer, shuffled_options_dict = shuffle_options(row, scenario['answer_col'])

            if options_text is None:
                error_msg = "Error: Shuffling Failed / Invalid Answer Info"
                print(f"오류: 옵션 셔플링 실패 또는 정답 정보 문제로 이 시나리오를 건너뜁니다.")
                df.loc[index, inf_col] = error_msg
                save_dataframe(df, args.output_file)
                continue

            # 3. 프롬프트 생성
            user_prompt = USER_PROMPT_TEMPLATE.format(
                context_section=context_section,
                question=question_text,
                options_text=options_text
            )

            # 4. LLM 호출 (통합)
            inference_result, latency = call_llm(
                args.model_id, 
                scenario['system_prompt'], 
                user_prompt, 
                hf_model=hf_model, 
                hf_tokenizer=hf_tokenizer, 
                openai_client=openai_client
            )

            # 5. 결과 처리 및 저장
            is_correct = False
            if isinstance(inference_result, int):
                df.loc[index, inf_col] = inference_result
                is_correct = (inference_result == correct_answer)
            else:
                df.loc[index, inf_col] = inference_result # 에러 메시지 저장

            df.loc[index, latency_col] = latency

            # 상세 정보 저장 (JSON 형식)
            details = {
                'shuffled_options': shuffled_options_dict,
                'correct_answer_shuffled': correct_answer,
                'predicted_answer': inference_result,
                'is_correct': is_correct
            }
            # 엑셀 저장을 위해 JSON 문자열로 저장
            df.loc[index, detail_col] = json.dumps(details, ensure_ascii=False)

            print(f"결과: 예측={inference_result}, 정답(셔플후)={correct_answer}, 정답여부={is_correct}, 지연시간={latency:.2f}s")

            # 6. 중간 저장 (매 단계 후 저장)
            if not save_dataframe(df, args.output_file):
                print("파일 저장 실패로 작업을 중단합니다.")
                return

    print(f"\n모든 처리가 완료되었습니다. 최종 결과는 '{args.output_file}'에 저장되었습니다.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LegalQA Multi-Hop Evaluation Script (HF + OpenAI)")
    parser.add_argument("--input_file", required=True, help="입력 Excel 파일 경로 (예: Legal_QA_Multi-Hop.xlsx)")
    parser.add_argument("--output_file", required=True, help="결과를 저장할 Excel 파일 경로 (예: evaluation_results.xlsx)")
    parser.add_argument("--model_id", required=True, help="사용할 모델 ID (HF 경로 또는 'openai:model-name' 형식)")
    parser.add_argument("--openai_key_file", default=None, help="OpenAI API Key 파일 경로 (OpenAI 모델 사용 시, 환경 변수보다 우선)")
    
    args = parser.parse_args()
    main(args)