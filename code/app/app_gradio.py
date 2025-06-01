import gradio as gr
import json
import atexit
from langchain_core.messages import HumanMessage
from graph_workflow import build_workflow
from log_to_db import init_db, log_chat_with_docs

# LangGraph 앱 빌드
graph_app = build_workflow().compile()

# 사용자별 히스토리 저장소
user_histories = {}

# 초기 상태 생성 함수
def init_state(user_id="user_default"):
    return {
        "user_id": user_id,
        "messages": [],
        "context": "",
        "retrieved_docs": [],
        "next": "user"
    }

# 대화 처리 함수
def chat_interface(history, message, state, user_id):
    if not user_id.strip():
        return history, "❗ Please enter your user ID.", state

    # state에 user_id 저장
    state["user_id"] = user_id

    # 사용자 입력 메시지 추가
    state["messages"].append(HumanMessage(content=message))

    # LangGraph 실행 (한 턴)
    state = graph_app.invoke(state)
    answer = state["messages"][-1].content
    context = state.get("context", "")
    retrieved_docs = state.get("retrieved_docs", [])

    # 사용자별 히스토리 누적
    if user_id not in user_histories:
        user_histories[user_id] = []
    user_histories[user_id].append({
        "question": message,
        "retrieved_docs": context,
        "response": answer
    })

    # ✅ SQLite 로그 기록
    log_chat_with_docs(user_id, message, answer, retrieved_docs)

    # 채팅 UI 업데이트
    history.append((message, answer))
    return history, "", state

# 종료 시 JSON으로 백업 저장 (보조용)
def save_user_logs():
    with open("user_histories.json", "w", encoding="utf-8") as f:
        json.dump(user_histories, f, indent=2, ensure_ascii=False)

atexit.register(save_user_logs)

# Gradio UI 구성
with gr.Blocks() as demo:
    gr.Markdown("### 💬 LangGraph + Ollama Multi-turn RAG Chatbot")

    with gr.Row():
        user_id_input = gr.Textbox(label="🆔 User ID", placeholder="Enter your name or ID")
        clear_btn = gr.Button("🧹 Clear")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask something...", label="Your Message")
    state = gr.State(init_state())

    # 메시지 입력 시 처리
    msg.submit(chat_interface, [chatbot, msg, state, user_id_input], [chatbot, msg, state])

    # 초기화 버튼 클릭 시 상태 재설정
    clear_btn.click(
        lambda user_id: ([], "", init_state(user_id)),
        inputs=[user_id_input],
        outputs=[chatbot, msg, state]
    )

# 실행
if __name__ == "__main__":
    init_db()  # ✅ DB 테이블 없으면 생성
    demo.launch(server_name="localhost", server_port=3000, share=False)
