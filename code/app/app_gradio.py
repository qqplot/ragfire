import html
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


def highlight_text(text: str, keyword: str) -> str:
    # 단순 하이라이팅: 대소문자 구분 없이 keyword 감싸기
    safe_text = html.escape(text)  # XSS 방지
    return safe_text.replace(
        keyword,
        f"<mark>{keyword}</mark>"
    )


# 초기 상태 생성 함수
def init_state(user_id="user_default"):
    return {
        "user_id": user_id,
        "messages": [],
        "context": "",
        "retrieved_docs": [],
        "next": "user"
    }

def chat_interface(history, message, state, user_id):
    if not user_id.strip():
        return history, "❗ Please enter your user ID.", state, ""

    state["user_id"] = user_id
    state["messages"].append(HumanMessage(content=message))

    state = graph_app.invoke(state)
    answer = state["messages"][-1].content
    retrieved_docs = state.get("retrieved_docs", [])

    # 히스토리 저장
    user_histories.setdefault(user_id, []).append({
        "question": message,
        "retrieved_docs": [doc.page_content for doc in retrieved_docs],
        "response": answer
    })

    # SQLite 기록
    log_chat_with_docs(user_id, message, answer, retrieved_docs)

    # ✅ 하이라이팅 포함 Accordion HTML
    references_html = ""
    if retrieved_docs:
        references_html += "<h4>📄 참고 문서</h4>"
        for i, doc in enumerate(retrieved_docs, start=1):
            title = f"{doc.metadata.get('law_name', '')} {doc.metadata.get('chapter', '')}" or f"문서 {i}"

            snippet = highlight_text(doc.page_content.strip(), message)
            references_html += f"""
            <details style="margin-bottom: 10px;">
                <summary><b>[{i}] {html.escape(title)}</b></summary>
                <div style="margin-left: 15px; white-space: pre-wrap; font-size: 0.9em;">{snippet}</div>
            </details>
            """

    # 채팅 UI 업데이트
    history.append((message, answer))
    return history, "", state, references_html



# 종료 시 JSON으로 백업 저장 (보조용)
def save_user_logs():
    with open("user_histories.json", "w", encoding="utf-8") as f:
        json.dump(user_histories, f, indent=2, ensure_ascii=False)

atexit.register(save_user_logs)

# Gradio UI 구성
with gr.Blocks() as demo:
    gr.Markdown("### 💬 내담-서울대 RAG Chatbot")

    with gr.Row():
        user_id_input = gr.Textbox(label="🆔 User ID", placeholder="Enter your name or ID")
        clear_btn = gr.Button("🧹 Clear")

    chatbot = gr.Chatbot()
    msg = gr.Textbox(placeholder="Ask something...", label="Your Message")

    # ✅ 새로 추가될 Accordion 출력 영역
    references_output = gr.HTML(label="참고 문서")  # 출력은 HTML로

    state = gr.State(init_state())

    # 메시지 입력 시 처리
    msg.submit(
        chat_interface,
        [chatbot, msg, state, user_id_input],
        [chatbot, msg, state, references_output]  # ✅ references_output 추가
    )

    # 초기화 버튼 클릭 시 상태 재설정
    clear_btn.click(
        lambda user_id: ([], "", init_state(user_id), ""),
        inputs=[user_id_input],
        outputs=[chatbot, msg, state, references_output]
    )

# 실행
if __name__ == "__main__":
    init_db()  # ✅ DB 테이블 없으면 생성
    demo.launch(server_name="localhost", server_port=3000, share=False)
