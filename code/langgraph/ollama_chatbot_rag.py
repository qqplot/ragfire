import argparse
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from retrieve_top1 import retrieve_top1

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "대화 기록"]
    context: Annotated[str, "context"]
    next: Annotated[str, "다음 노드"]

# 노드 정의
def user(state: AgentState) -> AgentState:
    # 사용자 입력을 받아 메시지에 추가
    user_input = input("User: ")
    state["messages"].append(HumanMessage(content=user_input))
    state["next"] = "retrieve_document"
    return state

def retrieve_document(state: AgentState, collection) -> AgentState:
    # 질문을 상태에서 가져옵니다.
    latest_question = state["messages"][-1].content

    # 문서에서 검색하여 관련성 있는 문서를 찾습니다.
    retrieved_docs = retrieve_top1("/home/shared/RAG/vector_db", latest_question, collection)

    # 검색된 문서를 context 키에 저장합니다.
    return {**state, "context": retrieved_docs, "next": "chatbot"}

def chatbot(state: AgentState, model_name, input_length) -> AgentState:
    # LLM를 사용하여 AI 응답 생성
    messages = state["messages"]
    last_message = messages[-1].content
    
    template = ChatPromptTemplate([
    ("system", "You are {model_name} model, a helpful assistant."),
    ("human","질문: {user_input}" + "\n참고 문서: {context}"),
    ])
    llm = OllamaLLM(model = model_name, num_ctx = input_length)
    chain = template | llm

    response = chain.invoke({"user_input": last_message, "context": state["context"]})
    state["messages"].append(AIMessage(content=response))
    print(f"ChatBot: {response}")
    
    state["next"] = "user"
    return state

# 그래프 구성
def main(args):
    workflow = StateGraph(AgentState)

    workflow.add_node("user", user)
    workflow.add_node("retrieve", lambda state: retrieve_document(state, args.collection))
    workflow.add_node("chatbot", lambda state: chatbot(state, args.model_name, args.input_length))

    workflow.set_entry_point("user")

    workflow.add_edge("user", "retrieve")
    workflow.add_edge("retrieve", "chatbot")
    workflow.add_edge("chatbot", "user")

    # 그래프 컴파일
    app = workflow.compile()

    # 실행
    for output in app.stream({
        "messages": [],
        "next": "user"
    }):
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default = "exaone3.5:2.4b", type=str, help="Name of the model to load")
    parser.add_argument("--input_length", default = 2048, type=int, help="Model input length")
    parser.add_argument("--collection", default = "law_bme", type=str, help="Vector db collection")

    args = parser.parse_args()
    
    main(args)