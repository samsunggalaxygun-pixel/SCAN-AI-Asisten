import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# Kita ubah cara panggil chain agar lebih kompatibel
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Konfigurasi API Key
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

def get_pdf_text():
    text = ""
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except:
            continue
    return text

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_qa_chain(vector_store):
    prompt_template = """
    Anda adalah asisten AI resmi Inspektorat Sultra (SCAN). 
    Gunakan konteks berikut untuk menjawab pertanyaan. Jika tidak ada di dokumen, katakan tidak tahu.
    
    Context: {context}
    Question: {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Menggunakan RetrievalQA (lebih stabil di versi baru)
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(),
        chain_type_kwargs={"prompt": prompt}
    )
    return chain

def main():
    st.set_page_config(page_title="SCAN AI Sultra", page_icon="🏛️")
    st.header("Asisten AI Inspektorat Sultra 🏛️")

    # Pastikan faiss_index tersedia
    if "processed" not in st.session_state:
        with st.spinner("Mempelajari dokumen Renstra & Perjanjian Kinerja..."):
            raw_text = get_pdf_text()
            if raw_text:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(raw_text)
                st.session_state.vector_store = get_vector_store(chunks)
                st.session_state.processed = True
                st.success("Selesai! Saya siap menjawab pertanyaan Anda.")
            else:
                st.error("File PDF belum diunggah ke repository.")

    question = st.text_input("Tanyakan sesuatu tentang dokumen Inspektorat:")
    if question and "vector_store" in st.session_state:
        with st.spinner("Berpikir..."):
            qa_chain = get_qa_chain(st.session_state.vector_store)
            response = qa_chain.invoke({"query": question})
            st.write("🤖 **Asisten SCAN:**")
            st.write(response["result"])

if __name__ == "__main__":
    main()
