import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import time
import json
import pandas as pd
import argparse
from model import load_model
from tqdm import tqdm

def main(args):
    if args.model_name == "exaone-2.4b" or "exaone" :
        model, tokenizer = load_model(args.model_name)

    with open("/home/shared/RAG/data/FEtest.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []

    def format_choices(choices):
        return "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])

    for item in tqdm(data, total=len(data), desc='inference FEtest.json without RAG'):
        question = item["question"]
        choices = format_choices(item["choices"])
        actual_answer = item["answer"]
        
        question_text = f"질문: {question}\n{choices}"
        answer_text = "답변: "
        input_text = question_text + '\n' + answer_text
        
        messages = [
        {"role": "system", 
        "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
        {"role": "user", "content": input_text}
        ]
        inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', tokenize=True, truncation=True, max_length=args.input_length).to("cuda")
        # inputs = tokenizer(input_text, return_tensors="pt").to("cuda")
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(inputs, eos_token_id=tokenizer.eos_token_id, max_new_tokens=500)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        results.append({
            "question_choices": question_text,
            "actual_answer": actual_answer,
            "generated_answer": generated_answer
        })
        
        print(f"Processed: {question} (Time: {execution_time:.2f} sec)")

    df = pd.DataFrame(results)
    df.to_csv(f"results_fe_worag_{args.model_name}_{args.input_length}.csv", index=False, encoding="utf-8-sig")
    # df.to_csv("results.csv", index=False, encoding="utf-8-sig")

    print("fin")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "exaone-2.4b", type=str, help="Name of the model to load")
    parser.add_argument("--input_length", default = 2048, type=int, help="Model input length")

    args = parser.parse_args()
    
    main(args)
