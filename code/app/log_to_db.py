import sqlite3
from datetime import datetime

DB_PATH = "/home/qqplot/ragfire/code/app/chat_logs.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_id TEXT,              
            user_input TEXT,
            model_response TEXT
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS retrieved_docs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_log_id INTEGER,
            doc_rank INTEGER,
            doc_id INTEGER,
            semantic_id TEXT,
            collection_name TEXT,
            law_name TEXT,
            chapter TEXT,
            page_content TEXT,
            FOREIGN KEY(chat_log_id) REFERENCES chat_logs(id)
        )
    ''')
    conn.commit()
    conn.close()

def log_chat_with_docs(user_id, user_input, model_response, retrieved_docs):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1. 채팅 로그 저장
    timestamp = datetime.utcnow().isoformat()
    c.execute('''
        INSERT INTO chat_logs (timestamp, user_id, user_input, model_response)
        VALUES (?, ?, ?, ?)
    ''', (timestamp, user_id, user_input, model_response))
    chat_log_id = c.lastrowid

    # 2. 검색된 문서들 저장
    for rank, doc in enumerate(retrieved_docs):
        meta = doc.metadata
        c.execute('''
            INSERT INTO retrieved_docs (
                chat_log_id, doc_rank, doc_id, semantic_id,
                collection_name, law_name, chapter, page_content
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            chat_log_id,
            rank,
            meta.get("doc_id"),
            meta.get("semantic_id"),
            meta.get("collection_name"),
            meta.get("law_name"),
            meta.get("chapter"),
            doc.page_content
        ))

    conn.commit()
    conn.close()
