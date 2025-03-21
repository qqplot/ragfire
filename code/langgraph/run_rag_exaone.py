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


# Load EXAONE
model_path = "/home/shared/RAG/model/exaone"

tokenizer = AutoTokenizer.from_pretrained(model_path)

quantization_config = BitsAndBytesConfig(  
    # load_in_4bit=True, 
    # bnb_4bit_compute_dtype=torch.float16 
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True,
    quantization_config=quantization_config,
).eval() 

# GraphState 상태 정의
class GraphState(TypedDict):
    question: Annotated[str, "Question"]  # 질문
    context: Annotated[str, "Context"]  # 문서의 검색 결과
    answer: Annotated[str, "Answer"]  # 답변
    messages: Annotated[list, add_messages]  # 메시지(누적되는 list)


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
def llm_answer(state: GraphState) -> GraphState:
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



if __name__ == "__main__":

    # 그래프 생성
    workflow = StateGraph(GraphState)

    # 노드 정의
    workflow.add_node("retrieve", retrieve_document)
    workflow.add_node("llm_answer", llm_answer)

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