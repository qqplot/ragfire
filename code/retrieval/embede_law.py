import json
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_db_from_json(
    json_path: str,
    persist_dir: str = "db_chroma",
    embedding_model_name: str = "BAAI/bge-m3"
):
    """
    로컬 JSON 파일을 읽어와서 원하는 필드를 기반으로
    텍스트를 chunk split → 임베딩 → Chroma DB에 저장하는 함수.
    """
    # JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 텍스트 청크(분할) 세팅
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,     # 청크 분할 크기(토큰 또는 문자 기준)
        chunk_overlap=100,  # 청크 간 오버랩
        separators=["\n"]   # 줄바꿈 등
    )

    # Document 리스트 만들기
    docs = []
    for d in data:
        content = f"[{d.get('law_name','')} / {d.get('chapter','')} / {d.get('article','')}]\n{d.get('text','')}"

        # chunk 나누기
        chunks = text_splitter.split_text(content)

        # Document 객체로 변환
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "law_name": d.get("law_name", ""),
                        "chapter": d.get("chapter", ""),
                        "article": d.get("article", ""),
                        "chunk_index": idx
                    }
                )
            )

    # 로컬 임베딩 모델
    # HuggingFaceEmbeddings는 토치(PyTorch)가 설치되어 있어야 합니다.
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    # Chroma DB 만들기
    # persist_directory: 임베딩 된 DB가 저장될 로컬 폴더
    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=YOUR_COLLECTION_NAME,
        persist_directory=persist_dir
    )

    # DB 영구 저장
    vectorstore.persist()
    print(f"벡터DB 생성 완료! 저장 위치: {persist_dir}")

if __name__ == "__main__":
    JSON_PATHS = [
    # JSON PATH 경로
    ]

    # 벡터 DB가 저장될 폴더 경로
    DB_DIR = "/home/shared/RAG/vector_db"

    # 함수 호출
    for JSON_PATH in JSON_PATHS:
        print(JSON_PATH)
        build_vector_db_from_json(json_path=JSON_PATH, persist_dir=DB_DIR)
