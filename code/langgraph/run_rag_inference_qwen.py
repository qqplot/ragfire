from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

from langchain_teddynote.messages import messages_to_history

from langgraph.graph import END, StateGraph
from langgraph.checkpoint.memory import MemorySaver

from langchain_teddynote.graphs import visualize_graph

from langchain_core.runnables import RunnableConfig
from langchain_teddynote.messages import invoke_graph, stream_graph, random_uuid

from retrieve_top1 import retrieve_top1

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# from model import load_model

#######

import datetime
import argparse
import gc
import multiprocessing as mp
import os
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
import deepspeed.comm as dist
from accelerate import init_empty_weights
from transformers import (AutoConfig, AutoTokenizer, AutoModelForCausalLM, 
                        )
# from transformers.integrations.deepspeed import HfDeepSpeedConfig
from transformers.integrations import HfDeepSpeedConfig

# from utils import (GB, add_model_hooks, cache_bytes,
#                    get_filename, get_quant_config, hidden_bytes, meta_to_cpu,
#                    model_bytes, write_benchmark_log)
from packaging import version

assert version.parse(deepspeed.__version__) >= version.parse("0.10.3"), "ZeRO-Inference with weight quantization and kv cache offloading is available only in DeepSpeed 0.10.3+, please upgrade DeepSpeed"



######
import argparse

# GraphState 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "Question"]  # 질문
    context: Annotated[str, "Context"]  # 문서의 검색 결과
    answer: Annotated[str, "Answer"]  # 답변
    messages: Annotated[list, add_messages]  # 메시지(누적되는 list)

def get_model_config(model_name):
    model_name = "/home/shared/RAG/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31"
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    return config

def get_ds_model(
    model_name="/home/shared/RAG/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31",
    cpu_offload=True,
    batch_size=1,
    bits=None,
    group_size=None,
):

    config = get_model_config(model_name)
    hidden_size = config.hidden_size
    deepspeed.init_distributed("nccl")
    pin_memory = True

    if getattr(config, 'torch_dtype', None) is None:
        dtype = torch.float16
    else:
        dtype = config.torch_dtype

    ds_config = {
        "fp16": {
            "enabled": dtype == torch.float16,
        },
        "bf16": {
            "enabled": dtype == torch.bfloat16,
        },
        "zero_optimization": {
            "stage": 3,
            "stage3_prefetch_bucket_size": 2 * hidden_size * hidden_size, 
            "stage3_param_persistence_threshold": hidden_size, #### 이걸 조정해야함!! (값이 클수록 오프로드 크기가 작아집니다)

            "stage3_max_live_parameters": 2 * hidden_size * hidden_size,
        },
        "steps_per_print": 2000,
        "train_batch_size": batch_size,
        "wall_clock_breakdown": False,
    }

    if bits == 4:
        quant_config = get_quant_config(config, bits=bits, group_size=group_size)
        ds_config.update(quant_config)
    if cpu_offload:
        ds_config["zero_optimization"]["offload_param"] = dict(
            device="cpu", pin_memory=pin_memory
        )

    dschf = HfDeepSpeedConfig(
        ds_config
    )  # this tells from_pretrained to instantiate directly on gpus

    # clear cache / free memory
    get_accelerator().empty_cache()
    gc.collect()

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype
    )

    model = model.eval()


    ds_engine = deepspeed.initialize(model=model, config_params=ds_config)[0]
    ds_engine.module.eval()
    model = ds_engine.module
    print(f"model.config = {model.config}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    return model, tokenizer


# 문서 검색 노드
def retrieve_document(state: GraphState) -> GraphState:
    # 질문을 상태에서 가져옵니다.
    latest_question = state["question"]

    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    # retrieved_docs = pdf_retriever.invoke(latest_question)
    retrieved_docs = retrieve_top1("/home/shared/RAG/vector_db", latest_question)

    # 검색된 문서를 형식화합니다.(프롬프트 입력으로 넣어주기 위함)
    # retrieved_docs = format_docs(retrieved_docs)

    # 검색된 문서를 context 키에 저장합니다.
    # return {"context": retrieved_docs}
    return {**state, "context": retrieved_docs}


# 답변 생성 노드
def llm_answer(state: GraphState, model, tokenizer) -> GraphState:
    # 질문을 상태에서 가져옵니다.
    latest_question = state["question"]

    # 검색된 문서를 상태에서 가져옵니다.
    context = state["context"]
    # chat_history = messages_to_history(state["messages"])

    print("# 검색된 문서를 상태에서 가져옵니다.")
    print(f"context= {context}")

    input_text = "\n".join([
        # f"대화 기록: {chat_history}",
        f"참고 문서: {context}",
        f"질문: {latest_question}",
        "답변:"
    ])
    inputs = tokenizer(input_text, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=500)

    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer_start_index = generated_answer.find("답변:") + len("답변:")
    generated_answer = generated_answer[answer_start_index:].strip()

    updated_messages = state["messages"][:]
    updated_messages.append(("user", latest_question))
    updated_messages.append(("assistant", generated_answer))

    return {**state, "answer": generated_answer, "messages": updated_messages}



def main(model_name: str):

    if model_name == "exaone" :
        model, tokenizer = load_model(model_name)
    # 모델과 토크나이저 로드
    if model_name == "qwen":
    # model, tokenizer = load_model(model_name)
        model, tokenizer = get_ds_model()

    # 그래프 생성
    workflow = StateGraph(GraphState)

    # 노드 정의
    workflow.add_node("retrieve", retrieve_document)
    workflow.add_node("llm_answer", lambda state: llm_answer(state, model, tokenizer))
    
    # 엣지 정의
    workflow.add_edge("retrieve", "llm_answer")  # 검색 -> 답변
    workflow.add_edge("llm_answer", END)  # 답변 -> 종료

    # 그래프 진입점 설정
    workflow.set_entry_point("retrieve")

    # 체크포인터 설정
    memory = MemorySaver()

    # 컴파일
    app = workflow.compile(checkpointer=memory)

    # 그래프 시각화
    visualize_graph(app)

    # config 설정(재귀 최대 횟수, thread_id)
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": random_uuid()})

    # 질문 입력
    inputs = GraphState(question="과태료의 부과")

    # 그래프 실행
    invoke_graph(app, inputs, config)

    # 그래프를 스트리밍 출력
    stream_graph(app, inputs, config)

    outputs = app.get_state(config).values

    print(f'Question: {outputs["question"]}')
    print("===" * 20)
    print(f'Answer:\n{outputs["answer"]}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "exaone", type=str, help="Name of the model to load.")
    parser.add_argument('--local_rank', type=int, default=0)
    
    args = parser.parse_args()
    model_name = args.model_name
    
    main(model_name)
