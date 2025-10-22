# pip install transformers>=4.41.0 accelerate peft torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

# 경로/모델명만 바꿔주세요
ADAPTER_DIR = "outputs/opensloth_cpt_ddp4_bk/checkpoint-400"  # 위 스크린샷의 경로
BASE_MODEL = "unsloth/Qwen3-1.7B"  # 학습에 쓴 베이스 모델 (예: "meta-llama/Meta-Llama-3-8B-Instruct")

# (팁) 학습 때 어떤 베이스 모델을 썼는지 모르면, ADAPTER_DIR 옆의
#  - training_args.bin을 torch.load로 열어 "model_name_or_path"를 확인하거나
#  - README.md/로그에 기록된 베이스 모델명을 확인하세요.

# tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, use_fast=True)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

# LoRA 어댑터 로드
# model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model.eval()

def chat(messages, max_new_tokens=8192, temperature=0.7, top_p=0.95, top_k=20):
    """
    messages: [{"role":"user","content":"..."} , {"role":"assistant","content":"..."} ...]
    transformers>=4.41 기준 chat_template.jinja가 있으면 자동으로 적용됩니다.
    """
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            pad_token_id=tokenizer.eos_token_id
        )
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generated

# 사용 예시
print(chat([{"role":"user","content":"안녕! 무엇을 도와줄 수 있어?"}]))

print(chat([{"role":"user","content":"스프링클러 설치 기준에 대해서 알려줘."}]))
