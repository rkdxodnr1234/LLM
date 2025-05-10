conda 환경을 만들면서 python 3.11 버전을 설치

# Flask (웹 서버)
pip install flask

# PDF 읽기
pip install PyPDF2

# LangChain (RAG 구성)
pip install langchain langchain-community

# FAISS (문서 검색용 벡터 DB)
pip install faiss-cpu

# HuggingFace 임베딩
pip install sentence-transformers

# Ollama (로컬 LLM API 연결용)
pip install requests  # Ollama는 REST API로 호출됨

# 위 모두 설치

pip install -r requirements.txt

# ollama llama3와 gemma3:12b를 사용
ollama run llama3
ollama run gemma3:12b


# faster-whisper 설치
pip install faster-whisper

# edge_tts 설치

# jdk 설치
# 환경변수 추가
# jpype 설치
conda install -c conda-forge jpype1
# konlpy 설치
