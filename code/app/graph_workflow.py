from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from retrieve import retrieve_topk


PATH_DB = "/home/qqplot/ragfire/code/app/vector_db/lawDB_chroma"

class AgentState(TypedDict):
    user_id: Annotated[str, "User identifier"]
    messages: Annotated[Sequence[HumanMessage | AIMessage], "Chat history"]
    context: Annotated[str, "Retrieved context"]
    retrieved_docs: Annotated[Sequence, "List of retrieved documents"]
    next: Annotated[str, "Next node"]

PROMPT = ChatPromptTemplate.from_messages([
    ("system", "You are {model_name} model, a helpful assistant."),
    ("human", "질문: {user_input}\n참고 문서: {context}")
])

def user(state: AgentState) -> AgentState:
    return state

def retrieve_document(state: AgentState, collection=None) -> AgentState:
    question = state["messages"][-1].content
    retrieved_docs = retrieve_topk(PATH_DB, question, k=3, return_documents=True)

    # context 문자열 생성
    formatted_context = "\n\n---\n\n".join(
        f"[{doc.metadata.get('law_name', '')} {doc.metadata.get('chapter', '')}]\n{doc.page_content}"
        for doc in retrieved_docs
    )

    return {**state, "context": formatted_context, "retrieved_docs": retrieved_docs, "next": "chatbot"}

def chatbot(state: AgentState, model_name: str, input_length: int) -> AgentState:
    user_input = state["messages"][-1].content
    context = state["context"]
    retrieved_docs = state.get("retrieved_docs", [])

    llm = OllamaLLM(model=model_name, num_ctx=input_length)
    chain = PROMPT | llm

    response = chain.invoke({
        "user_input": user_input,
        "context": context,
        "model_name": model_name
    })
    
    ai_msg = AIMessage(content=response) if isinstance(response, str) else response
    new_messages = state["messages"] + [ai_msg]
    return {**state, "messages": new_messages, "next": "user"}

def build_workflow(model_name="qwen2.5:32b-instruct", input_length=2048, collection=None):
    workflow = StateGraph(AgentState)
    workflow.add_node("user", user)
    workflow.add_node("retrieve", lambda s: retrieve_document(s, collection))
    workflow.add_node("chatbot", lambda s: chatbot(s, model_name, input_length))

    workflow.set_entry_point("user")
    workflow.add_edge("user", "retrieve")
    workflow.add_edge("retrieve", "chatbot")
    workflow.set_finish_point("chatbot")
    return workflow
