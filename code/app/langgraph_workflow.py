def run_langgraph_workflow(user_message: str) -> str:
    # 예시: 간단한 연결 흐름. 실제로는 LangGraph로 구성 가능
    retrieved_docs = fake_rag_retrieve(user_message)
    prompt = f"Context: {retrieved_docs}\n\nQuestion: {user_message}"
    
    # Ollama inference 예시
    import subprocess
    result = subprocess.run(["ollama", "run", "llama3", prompt], capture_output=True, text=True)
    return result.stdout.strip()

def fake_rag_retrieve(query):
    return "This is a placeholder document relevant to: " + query
