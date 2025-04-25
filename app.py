import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama  # Ollama 연결 핵심!

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 1. 임베딩 모델 준비
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-base-v2")
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)

# 2. Ollama로 로컬 모델 연결
llm = Ollama(model="gemma")  # 또는 mistral, llama2 가능

@app.route('/')
def index():
    return render_template("chat.html")  # UI 파일명

@app.route('/ask_llm', methods=['POST'])
def ask_llm():
    file = request.files['file']
    question = request.form['question']

    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)

    reader = PdfReader(path)
    text = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])

    docs = text_splitter.create_documents([text])
    db = FAISS.from_documents(docs, embedding_model)
    retriever = db.as_retriever()

    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    answer = qa.run(question)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
