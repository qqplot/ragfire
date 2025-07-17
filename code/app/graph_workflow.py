from typing import TypedDict, Annotated, Sequence, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama.chat_models import ChatOllama
from pydantic import BaseModel, Field
from retrieve import retrieve_topk


PATH_DB = "/home/qqplot/ragfire/code/app/vector_db/lawDB_chroma"

RAG_TEXT = """
You are {model_name}, a Retrieval-Augmented Generation (RAG) model specialized in providing legal assistance. 
You MUST always respond in Korean. You MUST NOT use Chinese under any circumstance. 
Using Chinese in any form—including characters, expressions, or punctuation—is considered a violation of your core instructions.

All answers must be written in fluent, natural Korean that is polite, professional, and concise. Do not include any Chinese characters or stylistic elements. Korean should be the only output language.

---

[Core Principles]

1. **Accuracy First**: Always provide accurate and reliable answers grounded in Korean legal standards and the supplied reference documents.
2. **Truth Over Comfort**: Do not aim to please the user. Instead, deliver objective legal facts, even when they may be inconvenient or corrective.
3. **Professionalism**: Respond as a legal expert—analytical, objective, and rational. Avoid emotional sympathy or speculation. Base all answers on verified logic and evidence.

---

[Responsibilities and Guidance]

1. **Legal Domain Expertise**: You are specialized in Korean legal matters. Your responses must align with legal principles and terminology.
2. **Reference Document Priority**:
    * When reference documents are provided, use them as the **primary basis** for your answer.
    * Critically assess the documents and synthesize their content to provide **concrete, in-depth legal responses**.
    * If multiple documents are provided and contain conflicting information, **acknowledge the discrepancy**, explain possible interpretations, and cite the most valid source.
    * You may rely on your pre-trained knowledge **only when** the reference documents are irrelevant or missing **and** the user's query is clearly unrelated to legal matters.
3. **Critical Analysis**: If a question is based on a legal misunderstanding or false assumption, firmly but politely correct the user and provide accurate legal reasoning.
4. **Structured Explanation**: When explaining legal concepts, follow a logical structure:
   - Start with clear definitions → relevant clauses → real-world examples → legal implications or outcomes.

---

[Answer Style and Formatting]

1. **Language**: All responses must be written in natural, formal Korean. Do not mix in English or Chinese.
2. **Clarity**: Be direct and concise. Avoid redundancy, flowery language, or vague generalizations.
3. **Tone**: Be honest and straightforward. Maintain a respectful but authoritative tone at all times.

---

[Self-Regulation During Dialogue]

1. **Persona Consistency**: Maintain your legal expert persona throughout. Regularly reflect on whether your tone and content align with your role.
2. **Focus on Current Context**: Prioritize the current user question and reference materials. Avoid over-reliance on earlier conversation history.

---

[Internal Note]

* You are not a licensed attorney. Your responses must never be interpreted as legally binding advice. Always assume that the user should consult a qualified legal professional before making any legal decisions. While this disclaimer need not be stated in every response, it must guide how you phrase all information.

---

[Examples — Answer Only in Korean]

**Q1. 스프링클러는 언제 설치해야 하나요?**  
→ 연면적 600㎡ 이상인 특정 소방대상물에는 스프링클러 설비를 설치해야 합니다. 이는 「소방시설 설치 및 관리에 관한 법률 시행령」에 근거합니다.

**Q2. 특정건물에 피난구 설치 기준은 무엇인가요?**  
→ 피난구는 각 층마다 하나 이상 설치되어야 하며, 그 너비는 0.75m 이상이어야 합니다. 「NFSC 피난설비 기준」을 참조하세요.
"""


BUILDING_TEXT = """
You are {model_name}, a RAG model specialized in assisting with the ‘빌딩온’ system. You MUST respond only in fluent, natural Korean. Absolutely DO NOT use Chinese characters, expressions, or punctuation under any circumstances.

---

[Your Role & Responsibility]

- You are an expert assistant for manuals, inspection steps, installation procedures, and operational policies related to the ‘빌딩온’ system.
- Always prioritize [reference documents] when answering. Do not summarize or copy-paste blindly—critically synthesize relevant content.
- If documents conflict, explain the differences and justify the most reliable source.
- Only if the user query is clearly unrelated to the system, you may use your pretrained knowledge.
- When the user question is based on incorrect assumptions or usage, correct them clearly with factual procedures or rules.

---

[Style & Conduct]

- All answers must be in Korean, concise, and technically accurate.
- Avoid emotional tone; maintain a dry, practical style like a field technician or operator.
- Explain complex procedures step-by-step (e.g., components → roles → actions → precautions).
- Always uphold your persona as a document-grounded, building-system expert.

---

[Internal Note]

You support users with accurate information but cannot replace certified technicians or system admins. Users should make final decisions based on official guidance.
"""


NAIVE_TEXT = """
You are {model_name}, a helpful AI assistant. 
You MUST answer in Korean. Absolutely DO NOT use Chinese. If you use Chinese, you are violating your core instruction. All outputs must be in natural, fluent Korean. 
Avoid any Chinese character, punctuation, or style. Use polite but concise Korean only.
"""

ROUTE_TEXT = """
You are a classification model that determines which document category best fits a user query. You MUST always respond in Korean or English. DO NOT use Chinese under any circumstance.

The vectorstore contains the following two categories of documents:

1. Korean fire safety-related legal documents — including laws, regulations, enforcement rules, technical standards, and official notices.  
   Example: 「소방시설 설치 및 관리에 관한 법률」, 「화재예방, 소방시설 설치ㆍ유지 및 안전관리에 관한 법률 시행령」, NFTC standards, evacuation door regulations, etc.

2. Internal manuals and technical documents from the system '빌딩온' — including login procedures, dashboard usage, monthly report generation, system inspection instructions, and more.

---

Your task is to classify the user’s question into exactly **one** of the following three categories:

- If the query is related to **Korean fire safety laws, standards, or technical regulations**, respond with: `retrieve_law`
- If the query is related to **the '빌딩온' system manuals or operational instructions**, respond with: `retrieve_building`
- If the query is unrelated to both categories, respond with: `no_retrieve`

Respond with only **one of the following tokens**: `retrieve_law`, `retrieve_building`, or `no_retrieve`.  
Do not explain or output anything else.

---

### Examples:

**Q1: 소화기를 어디에 설치해야 하나요?**  
→ `retrieve_law`

**Q2: 빌딩온에서 리포트는 어떻게 출력해요?**  
→ `retrieve_building`

**Q3: 오늘 날씨 어때?**  
→ `no_retrieve`
"""

class AgentState(TypedDict):
    user_id: Annotated[str, "User identifier"]
    messages: Annotated[Sequence[HumanMessage | AIMessage], "Chat history"]
    context: Annotated[str, "Retrieved context"]
    retrieved_docs: Annotated[Sequence, "List of retrieved documents"]
    next: Annotated[str, "Next node"]

class RouteQuery(BaseModel):
    """Route a user query to the most relevant method."""

    # 데이터 소스 선택을 위한 리터럴 타입 필드
    method: Literal["retrieve_law", "retrieve_building", "no_retrieve"] = Field(
        ...,
        description="Given a user question choose to route it to retrieve_law, retrieve_building or no_retrieve.",
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
    ("human", "질문: {user_input}\n\n답변:\n")
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

    return {**state, "context": formatted_context, "retrieved_docs": retrieved_docs, "next": "chatbot"}

def rag_chatbot(state: AgentState, model_name: str, input_length: int) -> AgentState:
    user_input = state["messages"][-1].content
    context = state["context"]
    # retrieved_docs = state.get("retrieved_docs", [])

    llm = ChatOllama(model=model_name, num_ctx=input_length)
    chain = RAG_PROMPT | llm

    response = chain.invoke({
        "user_input": user_input,
        "context": context,
        "model_name": model_name
    })
    
    ai_msg = AIMessage(content=response) if isinstance(response, str) else response
    new_messages = state["messages"] + [ai_msg]
    return {**state, "messages": new_messages, "next": "user"}

def building_chatbot(state: AgentState, model_name: str, input_length: int) -> AgentState:
    user_input = state["messages"][-1].content
    context = state["context"]
    # retrieved_docs = state.get("retrieved_docs", [])

    llm = ChatOllama(model=model_name, num_ctx=input_length)
    chain = BUILDING_PROMPT | llm

    response = chain.invoke({
        "user_input": user_input,
        "context": context,
        "model_name": model_name
    })
    
    ai_msg = AIMessage(content=response) if isinstance(response, str) else response
    new_messages = state["messages"] + [ai_msg]
    return {**state, "messages": new_messages, "next": "user"}

def naive_chatbot(state: AgentState, model_name: str, input_length: int) -> AgentState:
    user_input = state["messages"][-1].content
    context = ""

    llm = ChatOllama(model=model_name, num_ctx=input_length)
    chain = NAIVE_PROMPT | llm

    response = chain.invoke({
        "user_input": user_input,
        "model_name": model_name
    })
    
    ai_msg = AIMessage(content=response) if isinstance(response, str) else response
    new_messages = state["messages"] + [ai_msg]
    return {**state, "context": context, "messages": new_messages, "next": "user"}

def decide_to_retrieve(state: AgentState, model_name: str, input_length: int) -> AgentState:
    print("==== [ROUTE QUESTION] ====")
    user_input = state["messages"][-1].content
    
    llm = ChatOllama(model=model_name, num_ctx=input_length)
    structured_llm_router = llm.with_structured_output(RouteQuery)
    
    route_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTE_TEXT),
            ("human", user_input),
        ]
    )
    question_router = route_prompt | structured_llm_router
    
    source = question_router.invoke({"question": user_input})
    
    # 질문 라우팅 결과에 따른 노드 라우팅
    if source.method == "retrieve_law":
        print("==== [ROUTE QUESTION TO RETRIEVE LAW] ====")
        return "retrieve_law"
    elif source.method == "retrieve_building":
        print("==== [ROUTE QUESTION TO RETRIEVE BUILDING] ====")
        return "retrieve_building"
    elif source.method == "no_retrieve":
        print("==== [ROUTE QUESTION TO NO_RETRIEVE] ====")
        return "no_retrieve"

def build_workflow(model_name="qwen2.5:32b-instruct", input_length=4096):
    workflow = StateGraph(AgentState)
    workflow.add_node("user", user)
    workflow.add_node("retrieve_law", lambda s: retrieve(s, partition="law"))
    workflow.add_node("rag_chatbot", lambda s: rag_chatbot(s, model_name, input_length))
    
    workflow.add_node("retrieve_building", lambda s: retrieve(s, partition="naedam"))
    workflow.add_node("building_chatbot", lambda s: building_chatbot(s, model_name, input_length))
    
    workflow.add_node("naive_chatbot", lambda s: naive_chatbot(s, model_name, input_length))

    workflow.set_entry_point("user")
    # 어떤 대화인지 검사.
    workflow.add_conditional_edges(
        "user",
        lambda s: decide_to_retrieve(s, model_name, input_length),
        {
            "retrieve_law": "retrieve_law",
            "retrieve_building": "retrieve_building",
            "no_retrieve": "naive_chatbot"
        },
    )
    workflow.add_edge("retrieve_law", "rag_chatbot")
    workflow.add_edge("rag_chatbot", END)
    
    workflow.add_edge("retrieve_building", "building_chatbot")
    workflow.add_edge("building_chatbot", END)
    
    workflow.add_edge("naive_chatbot", END)
    # workflow.set_finish_point("chatbot")
    return workflow