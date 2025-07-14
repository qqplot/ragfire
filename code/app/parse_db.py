import json
import os
import pandas as pd
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
            source = f"[{record.get('law_name', '')} {record.get('chapter', '')}]"
            body = source + "\n" + body
            if not body:
                continue

            meta = {
                "doc_id": record.get("doc_id"),
                "semantic_id": record.get("semantic_id"),
                "collection_name": record.get("collection_name"),
                "law_name": record.get("law_name"),
                "chapter": record.get("chapter"),
                "partition": "law",
            }

            docs.append(body)
            metadatas.append(meta)

    print(f"✅ Loaded {len(docs)} documents from JSONL.")
    return docs, metadatas


def load_documents_from_csv(csv_path, collection_name="내담"):
    df = pd.read_csv(csv_path)
    docs = []
    metadatas = []

    for _, row in df.iterrows():
        body = str(row.get("content", "")).strip()
        title = str(row.get("title", "")).strip()
        full_text = f"[{title}]\n{body}"

        if not body:
            continue

        meta = {
            "doc_id": row.get("id"),
            "semantic_id": None,
            "collection_name": collection_name,
            "law_name": None,
            "chapter": title,
            "partition": "naedam",
        }

        docs.append(full_text)
        metadatas.append(meta)

    print(f"✅ Loaded {len(docs)} documents from CSV.")
    return docs, metadatas


def build_chroma_index(docs, metadatas, save_path, context_encoder_path):
    if os.path.exists(save_path):
        print(f"⚠️ Warning: {save_path} already exists. Existing index may be overwritten.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={
            "normalize_embeddings": True,
            "batch_size": 1
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
            "batch_size": 1
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
        print("doc_id:", meta.get("doc_id"))
        formatted_contexts.append(f"{doc.page_content}")

    return "\n\n---\n\n".join(formatted_contexts)


if __name__ == "__main__":
    # 경로
    jsonl_path = "/home/qqplot/ragfire/data/doc_ver2.jsonl"
    csv_path = "/home/qqplot/ragfire/code/app/naedam.csv"
    vector_db_path = "./vector_db/lawDB_chroma"

    # 1. JSONL + CSV 로드
    docs1, metas1 = load_documents(jsonl_path)
    docs2, metas2 = load_documents_from_csv(csv_path, collection_name="naedam")

    # 2. 병합 및 DB 구축
    all_docs = docs1 + docs2
    all_metas = metas1 + metas2
    build_chroma_index(all_docs, all_metas, save_path=vector_db_path, context_encoder_path="")

    # 3. 검색 테스트
    query = "스프링클러 설치 기준은?"
    collection = "naedam"  # or None
    context = retrieve_topk(vector_db_path, query, collection_filter=collection)

    print("\n📘 검색 결과:\n")
    print(context)
