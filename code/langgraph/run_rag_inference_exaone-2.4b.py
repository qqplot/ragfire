from typing import Annotated, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import random_uuid

from retrieve_top1 import retrieve_top1
from model import load_model

import json
import pandas as pd
import argparse
import gc
import multiprocessing as mp
import torch
import argparse
from tqdm import tqdm

######
# GraphState 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "Question"]  # 질문
    context: Annotated[str, "Context"]  # 문서의 검색 결과
    answer: Annotated[str, "Answer"]  # 답변
    messages: Annotated[list, add_messages]  # 메시지(누적되는 list)


# 문서 검색 노드
def retrieve_document(state: GraphState, collection) -> GraphState:
    # 질문을 상태에서 가져옵니다.
    latest_question = state["question"]

    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    retrieved_docs = retrieve_top1("/home/shared/RAG/vector_db", latest_question, collection)

    # 검색된 문서를 context 키에 저장합니다.
    return {**state, "context": retrieved_docs}

# 답변 생성 노드
def llm_answer(state: GraphState, model, tokenizer, input_length) -> GraphState:

    # print(f"***** model.config.max_position_embeddings: {model.config.max_position_embeddings}")
    # 질문을 상태에서 가져옵니다.
    latest_question = state["question"]

    # 검색된 문서를 상태에서 가져옵니다.
    context = state["context"]

    input_text = "\n".join([
        # f"대화 기록: {chat_history}",
        f"참고 문서: {context}",
        f"질문: {latest_question}",
        "답변:"
    ])

    messages = [
    {"role": "system", 
    "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
    {"role": "user", "content": input_text}
    ]
    inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors='pt', tokenize=True, truncation=True, max_length=input_length).to("cuda")
    # inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to("cuda")
    # inputs = tokenizer(input_text, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(inputs, eos_token_id=tokenizer.eos_token_id, max_new_tokens=500)

    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start_index = generated_answer.find("답변:") + len("답변:")
    generated_answer = generated_answer[answer_start_index:].strip()

    updated_messages = state["messages"][:]
    updated_messages.append(("user", latest_question))
    updated_messages.append(("assistant", generated_answer))

    return {**state, "answer": generated_answer, "messages": updated_messages}



def main(args):
    if args.model_name == "exaone" or "exaone-2.4b" :
        model, tokenizer = load_model(args.model_name)

    # elif model_name == "qwen":
    # # model, tokenizer = load_model(model_name)
    #     model, tokenizer = get_ds_model()

    # 그래프 생성
    workflow = StateGraph(GraphState)

    # 노드 정의
    workflow.add_node("retrieve", lambda state: retrieve_document(state, args.collection))
    workflow.add_node("llm_answer", lambda state: llm_answer(state, model, tokenizer, args.input_length))
    
    # 엣지 정의
    workflow.add_edge("retrieve", "llm_answer")  # 검색 -> 답변
    workflow.add_edge("llm_answer", END)  # 답변 -> 종료

    # 그래프 진입점 설정
    workflow.set_entry_point("retrieve")

    # 체크포인터 설정
    memory = MemorySaver()

    # 컴파일
    app = workflow.compile(checkpointer=memory)

    # config 설정(재귀 최대 횟수, thread_id)
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

    ########################
    with open("/home/shared/RAG/data/FEtest.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    results = []

    def format_choices(choices):
        return "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])

    for i, item in tqdm(enumerate(data), total=len(data), desc='inference FEtest.json with RAG'):
        question = item["question"]
        choices = format_choices(item["choices"])
        actual_answer = item["answer"]

        inputs = GraphState(
            question=f"{question}\n{choices}",
            context="",
            answer="",
            messages=[]
        )

        # outputs = app.get_state(config).values
        outputs = app.invoke(inputs, config=config)
        
        results.append({
            "question_choices": question,
            "actual_answer": actual_answer,
            "generated_answer": outputs["answer"],
            "context": outputs["context"]
        })

        del inputs, outputs
        gc.collect()
        torch.cuda.empty_cache()
        
    df = pd.DataFrame(results)
    df.to_csv(f"results_fe_rag_{args.model_name}_{args.input_length}_{args.collection}.csv", index=False, encoding="utf-8-sig")

    ##########################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "exaone-2.4b", type=str, help="Name of the model to load")
    parser.add_argument("--input_length", default = 2048, type=int, help="Model input length")
    parser.add_argument("--collection", default = "law_bme", type=str, help="Vector db collection")

    args = parser.parse_args()
    
    main(args)
