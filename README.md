# 🏛️ 내담 - 서울대학교 빌딩온 챗봇 과제

## 개요

서울대학교 **LAAL 연구실**에서는 **내담 C&C**와 함께,  
소방법 관련 **민원 질의응답 데이터**와 **소방법령 문서**를 기반으로 한  
**Retrieval-Augmented Generation (RAG)** 기반 챗봇 시스템을 개발하였습니다.  

해당 시스템은 다음과 같은 주요 구성 요소를 포함합니다:

- Gradio 인터페이스  
- Chroma 기반 VectorDB  
- Ollama LLM 연동  
- 대화 로그 저장 기능 (SQLite 기반)  

---


### 🔧 주요 기능

- 민원 질문 및 법령 문서 기반 RAG 응답  
- LangGraph 기반 챗봇 워크플로우 처리  
- Ollama를 통한 로컬 LLM 추론  
- Gradio 기반 사용자 인터페이스  
- 대화 로그 자동 저장 (`chat_logs.db`)  


### 🧩 시스템 구성도


---

### 📁 프로젝트 구조

<pre>
ragfire/
├── code
│   └── app
│       ├── app_gradio.py        ← 메인 실행 파일 (Gradio UI)
│       ├── graph_workflow.py    ← LangGraph 챗봇 흐름 정의
│       ├── retrieve.py          ← VectorDB 검색 모듈
│       ├── parse_db.py          ← Chroma DB 구축 스크립트
│       ├── log_to_db.py         ← 대화 로그 저장기
│       ├── chat_logs.db         ← SQLite DB (자동 생성)
│       ├── static/index.html    ← Gradio 커스텀 템플릿 (선택)
│       └── vector_db/lawDB_chroma/
│           ├── chroma.sqlite3
│           └── (FAISS 인덱스 파일들)
├── data/
│   ├── doc_ver2.jsonl           ← 법령 문서 데이터
│   ├── naedam.csv               ← 빌딩온 메뉴얼 요약
│   └── qna_ver1.jsonl           ← 민원 질의응답 데이터
└── vector_db/
    └── lawDB_chroma/            ← VectorDB 저장 위치
</pre>

---

---

## 설치

### 1️⃣ Ollama 설치 (LLM 실행 환경)

[https://ollama.com](https://ollama.com) 에서 운영체제에 맞는 설치파일을 내려받아 설치하세요.

백그라운드에서 서버 실행:

```bash
ollama serve &
```


```bash
ollama run qwen2.5:32b-instruct
```

더 큰 모데로 
```bash
ollama run qwen2.5:32b-instruct
```


#### ✅ 모델 실행 확인

```bash
curl http://localhost:11434/api/generate   -d '{
    "model": "qwen2.5-32b-instruct",
    "prompt": "한국의 수도는 어디인가요?",
    "stream": false
  }'
```

---


### 2️⃣ Python 환경 설치

#### 방법 A) Conda 환경 (권장)

```bash
conda create -n ragfire python=3.10
conda activate ragfire
pip install -r requirements.txt
```

#### 방법 B) pip만 사용하는 경우

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```



---

### 3️⃣ VectorDB (Chroma) 구성

아직 `vector_db/lawDB_chroma/` 폴더가 없다면 다음을 실행하세요:

```bash
cd code/app
python parse_db.py
```

---

## 실행

### 1️⃣ Ollama 서버 실행 확인

```bash
ollama serve &
```

Ollama가 제대로 실행 중인지 확인하세요. (`localhost:11434`)

---

### 2️⃣ Gradio 인터페이스 실행

```bash
cd code/app
python app_gradio.py
```

브라우저에서 자동으로 `http://localhost:3000`에 접속됩니다.

---

## ⚠️ 주의사항

- `graph_workflow.py` 내부에서 사용 모델명(e.g., `mistral`, `qwen2.5`)이 명시되어 있으므로 **Ollama에 동일한 모델을 로드**해야 합니다.
- **VectorDB가 비어 있을 경우**, 먼저 `parse_db.py`를 실행해 Chroma DB를 생성해야 합니다.
- `chat_logs.db`는 앱 실행 중 자동 생성되며, 로그 저장이 불필요할 경우 `log_to_db.py` 내 관련 코드를 주석 처리하면 됩니다.

---