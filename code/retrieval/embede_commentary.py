import json
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def build_vector_db_for_new_structure(
    json_path: str,
    persist_dir: str = "db_chroma",
    embedding_model_name: str = "BM-K/KoSimCSE-roberta-multitask" #BAAI/bge-m3
):
    """
    다음과 같은 구조의 JSON 만 처리하는 함수:
    {
        "Intro": "...문자열...",
        "공통사항(수계소화설비)": {
            "수원": "...문자열..."
        },
        "가압송수장치": "...문자열..."
    }
    
    - 1차 key가 문자열이면 → 바로 chunk
    - 1차 key가 dict면 → 내부의 2차 key 문자열을 chunk
    - metadata에는 'level1', 'level2' 등을 기록
    """

    # JSON 로드
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 텍스트 청크(분할) 세팅
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,       # 청크 분할 크기(토큰 또는 문자 기준)
        chunk_overlap=100,    # 청크 간 오버랩
        separators=["\n"]     # 줄바꿈 등
    )

    docs = []

    # 1) 최상위 key 순회
    for level1_key, level1_value in data.items():
        # (1) value가 문자열이면 → 바로 chunk
        if isinstance(level1_value, str):
            chunks = text_splitter.split_text(level1_value)
            for idx, chunk in enumerate(chunks):
                docs.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "level1": level1_key,
                            "chunk_index": idx
                        }
                    )
                )

        # (2) value가 dict면 → 내부 2차 key 순회
        elif isinstance(level1_value, dict):
            for level2_key, level2_value in level1_value.items():
                if isinstance(level2_value, str):
                    # 문자열이면 chunk
                    chunks = text_splitter.split_text(level2_value)
                    for idx, chunk in enumerate(chunks):
                        docs.append(
                            Document(
                                page_content=chunk,
                                metadata={
                                    "level1": level1_key,
                                    "level2": level2_key,
                                    "chunk_index": idx
                                }
                            )
                        )
                else:
                    # 2차 key가 또 dict거나 list이면, 여기서는 깊이 안 들어간다고 했으니 pass
                    pass
        else:
            # 1차 key가 문자열/dict가 아니라면 pass
            pass

    # 2) 문서를 Embedding 후, Chroma에 저장
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)

    if not os.path.exists(persist_dir):
        os.makedirs(persist_dir, exist_ok=True)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=YOUR_COLLECTION_NAME,
        persist_directory=persist_dir
    )

    vectorstore.persist()
    print(f"벡터DB 생성 완료! 저장 위치: {persist_dir}")


if __name__ == "__main__":
    JSON_PATHS = [
    # JSON PATH
    ]

    # 벡터 DB가 저장될 폴더 경로
    DB_DIR = "/home/shared/RAG/vector_db"

    # 함수 호출
    for JSON_PATH in JSON_PATHS:
        print(JSON_PATH)
        build_vector_db_for_new_structure(json_path=JSON_PATH, persist_dir=DB_DIR)
