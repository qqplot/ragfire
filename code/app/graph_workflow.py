import http.client
import json
from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from langchain_core.documents import Document
from pydantic import BaseModel, Field
from retrieve import retrieve_topk
from functools import partial

from prompts import RAG_TEXT, BUILDING_TEXT, NAIVE_TEXT, NAIVE_WEB_TEXT, ROUTE_TEXT

PATH_DB = "/home/qqplot/ragfire/code/app/vector_db/lawDB_chroma"
API_KEY = "" # Set your Serper API key here


class AgentState(TypedDict):
    user_id: Annotated[str, "User identifier"]
    messages: Annotated[Sequence[HumanMessage | AIMessage], "Chat history"]
    context: Annotated[str, "Retrieved context"]
    retrieved_docs: Annotated[Sequence, "List of retrieved documents"]
    next: Annotated[str, "Next node"]


class RouteQuery(BaseModel):
    """Route a user query to the most relevant method."""
    # 데이터 소스 선택을 위한 리터럴 타입 필드
    method: Literal["retrieve_law", "retrieve_building", "naive_web", "naive_no_web"] = Field(
        ...,
        description="Given a user question, choose one of: retrieve_law, retrieve_building, naive_web, naive_no_web."
    )


RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_TEXT),
    ("human", "질문: {user_input}\n참조 문서: {context}\n\n답변:\n")
])

BUILDING_PROMPT = ChatPromptTemplate.from_messages([
    ("system", BUILDING_TEXT),
    ("human", "질문: {user_input}\n참조 문서: {context}\n\n답변:\n")
])

NAIVE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", NAIVE_TEXT),
    ("human", "질문: {user_input}\n\n검색 결과: {context}\n\n답변:\n")
])

NAIVE_WEB_PROMPT = ChatPromptTemplate.from_messages([
    ("system", NAIVE_WEB_TEXT),
    ("human", "질문: {user_input}\n\n검색 결과: {context}\n\n답변:\n")
])


def user(state: AgentState) -> AgentState:
    return state


def retrieve(state: AgentState, partition=None) -> AgentState:
    question = state["messages"][-1].content
    K = 5 if partition == "law" else 3
    retrieved_docs = retrieve_topk(PATH_DB, question, k=K, partition_filter=partition)
    # print("Retrieved documents:", retrieved_docs)

    # context 문자열 생성
    formatted_contexts = []
    for doc in retrieved_docs:
        meta = doc.metadata
        law_name = meta.get("law_name", "").strip()
        chapter = meta.get("chapter", "").strip()
        source = f"[{law_name} {chapter}]\n" if partition == "law" else ""
        formatted_contexts.append(f"{source}{doc.page_content}")

    formatted_context = "\n\n---\n\n".join(formatted_contexts)

    return {**state, "context": formatted_context, "retrieved_docs": retrieved_docs}


def web_search(q: str, api_key: str) -> list:
    
    conn = http.client.HTTPSConnection("google.serper.dev")
    payload = json.dumps({
        "q": q,
        "gl": "kr",
        "hl": "ko",
    })
    headers = {
        'X-API-KEY': api_key,
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/search", payload, headers)
    res = conn.getresponse()
    data = res.read()
    json_str = data.decode("utf-8")
    data = json.loads(json_str)

    docs = data.get("organic", [])
    context = ""
    retrieved_docs = []    
    for i, item in enumerate(docs, start=1):
        title = item.get("title", "제목 없음")
        link = item.get("link", "링크 없음")
        snippet = item.get("snippet", "내용 없음")
        date = item.get("date", "날짜 없음")
        page_content = f"[{i}]\n제목: {title}\n링크: {link}\n내용: {snippet}\n날짜: {date}\n"
        context += f"{page_content}\n\n---\n\n"
        retrieved_docs.append(Document(
            page_content=page_content,
            metadata={
                "law_name": title,
                "link": link or None,
                "date": date or None,
                "source": "web",
            }
        ))
    return retrieved_docs, context


def decide_to_retrieve(state: AgentState, model_name: str, input_length: int) -> str:
    print("==== [ROUTE QUESTION] ====")
    user_input = state["messages"][-1].content

    llm = ChatOllama(model=model_name, num_ctx=input_length)
    structured_llm_router = llm.with_structured_output(RouteQuery)

    route_prompt = ChatPromptTemplate.from_messages(
        [("system", ROUTE_TEXT), ("human", user_input)]
    )
    question_router = route_prompt | structured_llm_router

    source = question_router.invoke({"question": user_input})
    print(f"==== [ROUTE DECISION: {source.method}] ====")
    return source.method


def generation_node(state: AgentState, llm: ChatOllama, prompt: ChatPromptTemplate, is_web: bool = False, is_naive: bool = False) -> AgentState:

    user_input = state["messages"][-1].content
    context = state.get("context", "") # context가 없는 경우 대비
    retrieved_docs = state.get("retrieved_docs", [])
    if is_web:
        retrieved_docs, context = web_search(user_input, API_KEY)
    
    chain = prompt | llm
    response = chain.invoke({
        "user_input": user_input,
        "context": context,
        "model_name": llm.model
    })

    ai_msg = response if isinstance(response, AIMessage) else AIMessage(content=str(response))
    new_messages = state["messages"] + [ai_msg]

    return {
        **state,
        "messages": new_messages,
        "retrieved_docs": retrieved_docs if not is_naive else [],  # naive 경우 빈 리스트로 설정
    }


def build_workflow(model_name="qwen2.5:32b-instruct", input_length=4096):

    llm = ChatOllama(model=model_name, num_ctx=input_length)

    # partial을 이용해 각기 다른 프롬프트를 사용하는 노드를 생성
    rag_chatbot_node = partial(generation_node, llm=llm, prompt=RAG_PROMPT)
    building_chatbot_node = partial(generation_node, llm=llm, prompt=BUILDING_PROMPT)
    naive_web_chatbot_node = partial(generation_node, llm=llm, prompt=NAIVE_WEB_PROMPT, is_web=True)
    naive_chatbot_node = partial(generation_node, llm=llm, prompt=NAIVE_PROMPT, is_naive=True)


    workflow = StateGraph(AgentState)
    workflow.add_node("user", user)
    workflow.add_node("retrieve_law", lambda s: retrieve(s, partition="law"))
    workflow.add_node("rag_chatbot", rag_chatbot_node)  
    workflow.add_node("retrieve_building", lambda s: retrieve(s, partition="naedam"))
    workflow.add_node("building_chatbot", building_chatbot_node)
    workflow.add_node("naive_web_chatbot", naive_web_chatbot_node)
    workflow.add_node("naive_chatbot", naive_chatbot_node)

    workflow.set_entry_point("user")

    workflow.add_conditional_edges(
        "user",
        lambda s: decide_to_retrieve(s, model_name, input_length),
        {
            "retrieve_law": "retrieve_law",
            "retrieve_building": "retrieve_building",
            "naive_web": "naive_web_chatbot",
            "naive_no_web": "naive_chatbot"
        },
    )
    workflow.add_edge("retrieve_law", "rag_chatbot")
    workflow.add_edge("rag_chatbot", END)
    
    workflow.add_edge("retrieve_building", "building_chatbot")
    workflow.add_edge("building_chatbot", END)
    
    workflow.add_edge("naive_chatbot", END)
    workflow.add_edge("naive_web_chatbot", END)
    # workflow.set_finish_point("chatbot")
    return workflow