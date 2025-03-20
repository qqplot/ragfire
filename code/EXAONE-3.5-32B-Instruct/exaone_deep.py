import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import json
import pandas as pd

model_path = "/home/shared/RAG/model/exaone_deep"  

tokenizer = AutoTokenizer.from_pretrained(model_path)

quantization_config = BitsAndBytesConfig(
    # load_in_8bit=True,  
    # llm_int8_enable_fp32_cpu_offload=True  
    # load_in_4bit=True,  # 1
    # bnb_4bit_compute_dtype=torch.float16 # 1
)

max_memory = {0: "20GB", "cpu": "100GB"}  

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    max_memory=max_memory,
    quantization_config=quantization_config,
    trust_remote_code=True
)


with open("/home/shared/RAG/data/FEtest.json", "r", encoding="utf-8") as file:
    data = json.load(file)

results = []

def format_choices(choices):
    return "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])

for item in data:
    question = item["question"]
    choices = format_choices(item["choices"])
    actual_answer = item["answer"]
    
    input_text = f"{question}\n{choices}\n"
    
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
    
    start_time = time.time()
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=500)
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    results.append({
        "question_choices": input_text,
        "actual_answer": actual_answer,
        "generated_answer": generated_answer
    })
    
    print(f"Processed: {question} (Time: {execution_time:.2f} sec)")

df = pd.DataFrame(results)
df.to_csv("deep_results.csv", index=False, encoding="utf-8-sig")

print("fin")



