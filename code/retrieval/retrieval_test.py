from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def test_search(db_dir, query):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Chroma(
        collection_name=YOUR_COLLECTION_NAME, 
        persist_directory=db_dir,
        embedding_function=embeddings
    )

    results = vectorstore.similarity_search(query, k=10)
    for idx, doc in enumerate(results, start=1):
        print(f"[ Top {idx} ]")
        print("문서 내용:", doc.page_content)
        print("메타데이터:", doc.metadata)
        print("--------")

if __name__ == "__main__":
    test_search("/home/shared/RAG/vector_db", "출입구로부터 보행거리 5m 이내에 설치해야 하는 것은 무엇인가요?")
