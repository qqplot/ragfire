import html
import json
import atexit
import argparse
import gradio as gr
from langchain_core.messages import HumanMessage
from graph_workflow import build_workflow
from log_to_db import ChatLogger


# 사용자별 히스토리 저장소
user_histories = {}


def highlight_text(text: str, keyword: str) -> str:
    safe_text = html.escape(text)
    return safe_text.replace(keyword, f"<mark>{keyword}</mark>")


def init_state(user_id="user_default"):
    return {
        "user_id": user_id,
        "messages": [],
        "context": "",
        "retrieved_docs": [],
        "next": "user"
    }


def chat_interface(graph_app, history, message, state, user_id, logger=None):
    if not user_id.strip():
        return history, "❗ Please enter your user ID.", state, ""

    state["user_id"] = user_id
    state["messages"].append(HumanMessage(content=message))

    state = graph_app.invoke(state)
    answer = state["messages"][-1].content
    retrieved_docs = state.get("retrieved_docs", [])

    user_histories.setdefault(user_id, []).append({
        "question": message,
        "retrieved_docs": [doc.page_content for doc in retrieved_docs],
        "response": answer
    })

    logger.log_chat_with_docs(user_id, message, answer, retrieved_docs)

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

    history.append((message, answer))
    return history, "", state, references_html


def save_user_logs():
    with open("user_histories.json", "w", encoding="utf-8") as f:
        json.dump(user_histories, f, indent=2, ensure_ascii=False)


def launch_ui(graph_app, logger=None):
    with gr.Blocks() as demo:
        gr.Markdown("### 💬 내담-서울대 RAG Chatbot")

        with gr.Row():
            user_id_input = gr.Textbox(label="🆔 User ID", placeholder="Enter your name or ID")
            clear_btn = gr.Button("🧹 Clear")

        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Ask something...", label="Your Message")
        references_output = gr.HTML(label="참고 문서")
        state = gr.State(init_state())

        # submit handler
        msg.submit(
            lambda h, m, s, uid: chat_interface(graph_app, h, m, s, uid, logger),
            inputs=[chatbot, msg, state, user_id_input],
            outputs=[chatbot, msg, state, references_output]
        )

        # clear handler
        clear_btn.click(
            lambda uid: ([], "", init_state(uid), ""),
            inputs=[user_id_input],
            outputs=[chatbot, msg, state, references_output]
        )

    demo.launch(server_name="localhost", server_port=3000, share=False)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5:32b-instruct", help="Ollama 모델 이름")
    parser.add_argument("--ctx", type=int, default=4096, help="LLM context window 크기")
    return parser.parse_args()


def main():
    args = parse_args()

    logger = ChatLogger(db_path="./chat_logs.db")
    logger.init_db()

    graph_app = build_workflow(model_name=args.model, input_length=args.ctx).compile()
    atexit.register(save_user_logs)
    launch_ui(graph_app, logger=logger)


if __name__ == "__main__":
    main()
