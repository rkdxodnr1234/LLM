from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

# 1. 문서 로드 및 분할
loader = PyPDFLoader("document.pdf")
docs = loader.load()

# 2. Embedding
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
vector_db = FAISS.from_documents(docs, embedding_model)

# 3. LLM 파이프라인 구성 (Gemma도 가능)
qa_llm = HuggingFacePipeline(pipeline=pipeline("text-generation", model="google/gemma-2b"))

# 4. Retrieval QA 체인
qa_chain = RetrievalQA.from_chain_type(llm=qa_llm, retriever=vector_db.as_retriever())

# 5. 질문에 응답
response = qa_chain.run("이 문서에서 농약에 대한 정보를 알려줘")
print(response)
