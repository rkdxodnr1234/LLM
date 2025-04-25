import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔹 임베딩 모델 준비
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")

# 🔹 텍스트 쪼개기
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 🔹 공개 LLM: Mistral (접근 제한 없음)
llm_pipeline = pipeline("text-generation", model="mistralai/Mistral-7B-v0.1", max_new_tokens=512)
llm = HuggingFacePipeline(pipeline=llm_pipeline)

@app.route('/')
def index():
    return render_template("llm_index.html")

@app.route('/ask_llm', methods=['POST'])
def ask_llm():
    file = request.files['file']
    question = request.form['question']

    # PDF 저장 및 읽기
    pdf_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(pdf_path)
    reader = PdfReader(pdf_path)
    text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

    # 텍스트 → Chunk → 임베딩 → Vector DB
    docs = text_splitter.create_documents([text])
    db = FAISS.from_documents(docs, embedding_model)

    # RAG QA 체인 구성
    retriever = db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

    # 질문 처리
    answer = qa_chain.run(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
