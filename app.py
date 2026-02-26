import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# Setup API Key dari Secrets Streamlit
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

def get_pdf_text_from_folder():
    text = ""
    # Membaca semua file PDF yang ada di repository (folder utama)
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Gagal membaca {pdf}: {e}")
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Anda adalah asisten AI resmi untuk Inspektorat Daerah Provinsi Sulawesi Tenggara (SCAN). 
    Tugas Anda adalah membantu memberikan informasi berdasarkan dokumen yang disediakan.
    Jawablah dengan sopan, formal, dan akurat. 
    Jika jawaban tidak ada dalam dokumen, katakan: "Maaf, informasi tersebut tidak ditemukan dalam dokumen Renstra atau Perjanjian Kinerja saat ini."

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    # Memuat index pencarian dokumen
    if os.path.exists("faiss_index"):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("🤖 Asisten SCAN: ", response["output_text"])
    else:
        st.error("Data belum diproses. Harap tunggu sebentar atau hubungi admin.")

def main():
    st.set_page_config(page_title="Asisten AI SCAN", page_icon="🏛️")
    st.header("Asisten AI Inspektorat Sultra 🏛️")
    st.write("Gunakan chatbot ini untuk bertanya seputar Renstra 2025-2029 dan dokumen resmi lainnya.")

    # PROSES OTOMATIS: Bot langsung belajar saat dijalankan
    if "processed" not in st.session_state:
        with st.spinner("Sedang mempelajari dokumen SCAN Sultra..."):
            raw_text = get_pdf_text_from_folder()
            if raw_text:
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.session_state.processed = True
                st.success("Selesai! Saya sudah memahami dokumen Renstra & Perjanjian Kinerja.")
            else:
                st.warning("Belum ada file PDF yang ditemukan di server.")

    user_question = st.text_input("Ketik pertanyaan Anda di sini:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
