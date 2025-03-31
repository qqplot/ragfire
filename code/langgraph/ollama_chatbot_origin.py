import argparse
from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
# from ollama import chat
# from ollama import ChatResponse

# 상태 정의
class AgentState(TypedDict):
    messages: Annotated[Sequence[HumanMessage | AIMessage], "대화 기록"]
    next: Annotated[str, "다음 노드"]

# 노드 정의
def user(state: AgentState) -> AgentState:
    # 사용자 입력을 받아 메시지에 추가
    user_input = input("Human: ")
    state["messages"].append(HumanMessage(content=user_input))
    state["next"] = "chatbot"
    return state

def chatbot(state: AgentState, model_name, input_length) -> AgentState:
    # Llama를 사용하여 AI 응답 생성
    messages = state["messages"]
    last_message = messages[-1].content
    

    template = ChatPromptTemplate([
        ("system", "You are {model_name} model, a helpful assistant."),
        ("human", "{user_input}"),
        ])
    
    llm = OllamaLLM(model = model_name, num_ctx = input_length)
    chain = template | llm
    response = chain.invoke({"user_input": last_message, "model_name": model_name})

    # response: ChatResponse = chat(model='exaon3.5:2.4b', messages=[
    # {"role": "system", "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
    # {"role": "user", "content": last_message}
    # ])
    state["messages"].append(AIMessage(content=response))
    print(f"ChatBot: {response}")
    state["next"] = "user"
    return state

def main(args):
        
    # 그래프 구성
    workflow = StateGraph(AgentState)

    workflow.add_node("user", user)
    workflow.add_node("chatbot", lambda state: chatbot(state, args.model_name, args.input_length))

    workflow.set_entry_point("user")

    workflow.add_edge("user", "chatbot")
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

    args = parser.parse_args()
    
    main(args)