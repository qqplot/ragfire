from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

YOUR_COLLECTION_NAME = "law_bme"

def retrieve_top1(db_dir, query):
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    vectorstore = Chroma(
        collection_name=YOUR_COLLECTION_NAME, 
        persist_directory=db_dir,
        embedding_function=embeddings
    )

    # 우선 사용자가 입력한 query로 Top-3까지 검색
    results = vectorstore.similarity_search(query, k=3)
    
    # 검색된 결과가 하나 이상 있으면
    if results:
        # Top-1 문서
        top_doc = results[0]
        
        # print("[ Top 1 ]")
        # print("문서 내용:", top_doc.page_content)
        # print("메타데이터:", top_doc.metadata)
        # print("--------")

        top_chapter = top_doc.metadata.get("chapter", None)
        top_article = top_doc.metadata.get("article", None)
        
        top_level1 = top_doc.metadata.get("level1", None)
        top_level2 = top_doc.metadata.get("level2", None)

        # 1) chapter, article이 있는 경우
        if top_chapter is not None and top_article is not None:
            same_chapter_article_docs = vectorstore.similarity_search(
                "",
                k=50,  # 충분히 큰 값으로 설정
                filter={
                    "$and": [
                        {"chapter": {"$eq": top_chapter}},
                        {"article": {"$eq": top_article}}
                    ]
                }
            )

            print(f"\n>>> Top-1과 동일한 chapter('{top_chapter}')와 article('{top_article}')을 가진 모든 문서 <<<")

            context_docs = same_chapter_article_docs


        # 2) chapter, article 중 하나라도 없다면, level1, level2 확인
        else:
            if top_level1 is not None and top_level2 is not None:
                same_level_docs = vectorstore.similarity_search(
                    "",
                    k=50,  # 충분히 큰 값으로 설정
                    filter={
                        "$and": [
                            {"level1": {"$eq": top_level1}},
                            {"level2": {"$eq": top_level2}}
                        ]
                    }
                )

                print(f"\n>>> Top-1과 동일한 level1('{top_level1}')와 level2('{top_level2}')를 가진 모든 문서 <<<")

                context_docs = same_level_docs

            else:
                print("메타데이터에 'chapter'/'article' 또는 'level1'/'level2' 정보가 충분하지 않습니다.")

        context = ""
        for idx, doc in enumerate(context_docs, start=1):
            # print(f"[ 문서 {idx} ]")
            # print("문서 내용:", doc.page_content)
            # print("메타데이터:", doc.metadata)
            # print("--------")
            context = context + " " + doc.page_content
        
        return context
    else:
        print("검색 결과가 없습니다.")
        return None
    
    


if __name__ == "__main__":
    test_search("/home/shared/RAG/vector_db", "과태료의 부과")
