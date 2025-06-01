from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def retrieve_topk(db_path, query, k=3, return_documents=False, collection_filter=None):

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={
        "normalize_embeddings": True,
        }
    )

    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    if collection_filter:
        results = db.similarity_search(query, k=k, filter={"collection_name": collection_filter})
    else:
        # print("문서 수:", len(db.get()["documents"]))
        results = db.similarity_search(query, k=k)

    if return_documents:
        return results

    # default: 반환은 string context
    formatted_contexts = []
    for doc in results:
        meta = doc.metadata
        source = f"[{meta.get('law_name', '')} {meta.get('chapter', '')}]"
        formatted_contexts.append(f"{source}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_contexts)

