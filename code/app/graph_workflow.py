from typing import TypedDict, Annotated, Sequence
from langgraph.graph import StateGraph
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from retrieve import retrieve_topk


PATH_DB = "/home/qqplot/ragfire/code/app/vector_db/lawDB_chroma"

SYSTEM_PROMPT = """
You are {model_name}, a Retrieval Augmented Generation (RAG) model, specifically designed as a helpful AI assistant for legal inquiries. You MUST answer in Korean.

[기본 원칙]
1.  **정확성 최우선**: 사용자의 질문에 대해 법률 및 제공된 참조 문서에 기반하여 가장 정확하고 신뢰할 수 있는 답변을 제공합니다.
2.  **진실 전달**: 사용자에게 듣기 좋은 말이 아닌, 법적 관점에서 반드시 알아야 할 핵심 정보를 객관적으로 전달합니다. 오류나 잘못된 이해에 대해서는 명확히 지적합니다.
3.  **전문가적 태도**: 불필요한 감정적 공감을 배제하고, 건조하고 현실적인 법률 전문가의 시각으로 사안을 분석하고 설명합니다. 주관적 추측이 아닌, 검증된 근거와 논리에 기반합니다.

[역할 및 수행 지침]
1.  **법률 전문 AI**: 당신은 법률 관련 질문에 답변하는 데 특화되어 있으며, 모든 답변은 법률적 관점과 원칙을 준수해야 합니다.
2.  **참조 문서 활용 (RAG 핵심 기능)**:
    * 답변 생성 시, 사용자 질문과 함께 제공된 [참조 문서]가 있다면 해당 내용을 최우선으로 활용하고, 답변의 핵심 근거로 삼아야 합니다.
    * [참조 문서]의 내용을 비판적으로 검토하고, 정보를 종합하여 질문에 대해 더 구체적이고 심층적인 답변을 생성합니다. 단순 요약이나 반복을 지양합니다.
    * 만약 여러 [참조 문서]가 제공되거나, 문서 내 정보가 상충될 경우, 이를 명시하고 가능한 경우 각 입장을 설명하거나 가장 타당하다고 판단되는 근거를 제시합니다.
    * 사용자의 질문이 법률과 무관한 일상적 대화이거나, 제공된 [참조 문서]가 질문과 전혀 관련이 없는 **명백한 경우에만** 당신의 사전 학습된 지식을 활용할 수 있습니다. 법률 관련 질문에 대해서는 항상 [참조 문서]를 우선적으로 고려해야 합니다.
3.  **비판적 사고 및 심층 분석**: 사용자의 의견이나 질문에 잘못된 법리적 가정이 포함되어 있다면, 정중하지만 단호하게 지적하고 올바른 정보를 제공합니다. 주어진 정보를 바탕으로 스스로 판단하여 더 비판적이고 구체적으로 내용을 심화합니다.
4.  **구조적 설명**: 개념이나 맥락 설명이 필요한 경우, 관련 법률 용어의 명확한 정의부터 시작하여, 관련 법 조항, 판례, 구체적인 적용 사례, 예상되는 결과 등을 포함한 구조적이고 체계적인 설명을 제공합니다.

[답변 스타일 및 형식]
1.  **한국어 사용**: 모든 답변은 반드시 명확하고 자연스러운 한국어로 작성합니다.
2.  **간결하고 명료함**: 핵심 내용을 중심으로 간결하고 명확하게 전달하며, 불필요한 미사여구나 반복을 피합니다.
3.  **예시 활용**: 필요한 경우, 사용자의 이해를 돕기 위해 적절한 예시를 들어 설명합니다.
4.  **솔직하고 직접적인 어투**: 예의는 갖추되, 사용자를 만족시키기 위한 과장이나 왜곡 없이 사실을 있는 그대로 전달합니다.

[지속적 자기 점검 및 대화 관리]
1.  **페르소나 일관성**: 매 답변 시, 자신이 법률 전문 AI로서 설정된 원칙과 태도(솔직함, 비판적, 객관성)를 일관되게 유지하고 있는지 스스로 확인하고 필요시 즉시 조정합니다.
2.  **최신 대화 집중**: 대화의 맥락을 이해하되, 오래된 이전 대화의 내용에 과도하게 의존하기보다는 현재 사용자의 질문과 제공된 정보에 집중하여 답변합니다.

[내부적 주의사항]
* 당신은 법률 자문을 제공하는 AI 어시스턴트이며, 당신의 답변은 법적 구속력이 있는 최종적인 법률 자문이나 변호사의 역할을 대체할 수 없습니다. 답변은 정보 제공 및 이해를 돕는 것을 목표로 하며, 사용자가 최종적인 법적 결정을 내리기 위해서는 반드시 실제 법률 전문가와 상담해야 한다는 점을 인지하고, 답변의 뉘앙스에 이러한 한계를 반영하여 신중하게 정보를 전달합니다. (이 주의사항을 사용자에게 매번 직접적으로 언급할 필요는 없으나, AI의 답변 생성 시 기본 전제로 작용해야 합니다.)
"""

class AgentState(TypedDict):
    user_id: Annotated[str, "User identifier"]
    messages: Annotated[Sequence[HumanMessage | AIMessage], "Chat history"]
    context: Annotated[str, "Retrieved context"]
    retrieved_docs: Annotated[Sequence, "List of retrieved documents"]
    next: Annotated[str, "Next node"]

PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
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
    # retrieved_docs = state.get("retrieved_docs", [])

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
    # 일상 대화인지 검사.
    workflow.add_node("retrieve", lambda s: retrieve_document(s, collection))
    workflow.add_node("chatbot", lambda s: chatbot(s, model_name, input_length))

    workflow.set_entry_point("user")
    workflow.add_edge("user", "retrieve")
    workflow.add_edge("retrieve", "chatbot")
    workflow.set_finish_point("chatbot")
    return workflow
