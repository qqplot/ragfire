from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def retrieve_topk(db_path, query, k=3, partition_filter=None):

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-m3",
        model_kwargs={"device": "cuda"},
        encode_kwargs={
        "normalize_embeddings": True,
        }
    )

    db = Chroma(persist_directory=db_path, embedding_function=embedding_model)
    if partition_filter:
        results = db.similarity_search(query, k=k, filter={"partition": partition_filter})
    else:
        results = db.similarity_search(query, k=k)

    return results

