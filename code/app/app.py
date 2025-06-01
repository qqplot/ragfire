from flask import Flask, request, jsonify, send_from_directory
from langchain_core.messages import HumanMessage
from graph_workflow import build_workflow
from uuid import uuid4

# 세션별 상태 저장소 (메모리 기반)
session_store = {}

# Flask 앱 초기화
app = Flask(__name__, static_folder="static")

# LangGraph 워크플로우 준비
workflow = build_workflow()
runner = workflow.compile()

# ✅ HTML UI 제공
@app.route("/")
def serve_ui():
    return app.send_static_file("index.html")

# ✅ Chat API (세션 기반 멀티턴)
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message", "")
    session_id = data.get("session_id")

    # 세션이 없으면 새로 생성
    if not session_id or session_id not in session_store:
        session_id = str(uuid4())
        state = {
            "messages": [],
            "context": "",
            "next": "retrieve"
        }
    else:
        state = session_store[session_id]
        state["next"] = "retrieve"  # 항상 next 초기화

    # 사용자 메시지 추가
    state["messages"].append(HumanMessage(content=user_input))

    # LangGraph 한 턴 실행
    state = runner.invoke(state)

    # 세션 상태 저장
    session_store[session_id] = state

    # 마지막 응답 반환
    last_response = state["messages"][-1].content
    return jsonify({
        "response": last_response,
        "session_id": session_id
    })

# 앱 실행
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
