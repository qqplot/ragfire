import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct"
# model_name = '/home/shared/RAG/model/exaone'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Choose your prompt
prompt_e = "Explain how wonderful you are"  # English example
prompt_k = "손흥민이 누군지 알아?"       # Korean example

# inputs = tokenizer(prompt_k, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
inputs = tokenizer(prompt_k, return_tensors='pt', truncation=True, max_length=2048)

with torch.no_grad():
        outputs = model.generate(inputs['input_ids'].to('cuda'), eos_token_id=tokenizer.eos_token_id, max_new_tokens=500)

generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_answer)
