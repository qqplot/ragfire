import json
import os
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEmbeddings


def load_documents(jsonl_path):
    docs = []
    metadatas = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line.strip())
            body = record.get("chapter_body", "").strip()
            if not body:
                continue  # 내용이 비어 있으면 스킵

            meta = {
                "doc_id": record.get("doc_id"),
                "semantic_id": record.get("semantic_id"),
                "collection_name": record.get("collection_name"),
                "law_name": record.get("law_name"),
                "chapter": record.get("chapter"),
            }

            docs.append(body)
            metadatas.append(meta)

    print(f"✅ Loaded {len(docs)} documents.")
    return docs, metadatas


def build_chroma_index(docs, metadatas, save_path, context_encoder_path):
    if os.path.exists(save_path):
        print(f"⚠️ Warning: {save_path} already exists. Existing index may be overwritten.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 4
        }
    )

    langchain_docs = [
        Document(page_content=doc, metadata=meta)
        for doc, meta in zip(docs, metadatas)
    ]

    db = Chroma.from_documents(
        langchain_docs,
        embedding_model,
        persist_directory=save_path
    )
    db.persist()
    print(f"✅ Chroma DB built and saved to: {save_path}")


def retrieve_topk(db_path, query, collection_filter=None, k=3):
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 4
        }
    )

    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)

    print(f"🔍 Query: {query}")
    if collection_filter:
        print(f"📂 Filter: collection_name = '{collection_filter}'")
        results = db.similarity_search(query, k=k, filter={"collection_name": collection_filter})
    else:
        results = db.similarity_search(query, k=k)

    if not results:
        return "❗ 검색 결과가 없습니다."

    formatted_contexts = []
    for doc in results:
        meta = doc.metadata
        source = f"[{meta.get('law_name', '')} {meta.get('chapter', '')}]"
        formatted_contexts.append(f"{source}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_contexts)


if __name__ == "__main__":
    # 절대 경로 설정
    jsonl_path = "/home/qqplot/ragfire/data/doc_ver2.jsonl"
    vector_db_path = "./vector_db/lawDB_chroma"

    # 1. 문서 로드 및 DB 구축
    docs, metas = load_documents(jsonl_path)
    build_chroma_index(docs, metas, save_path=vector_db_path, context_encoder_path="")

    # 2. 검색 테스트
    query = "스프링클러 설치 기준은?"
    collection = None
    context = retrieve_topk(vector_db_path, query, collection_filter=collection)

    print("\n📘 검색 결과:\n")
    print(context)
