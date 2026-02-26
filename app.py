import streamlit as st
import os
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Konfigurasi API Key
api_key = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=api_key)

def get_pdf_text():
    text = ""
    pdf_files = [f for f in os.listdir('.') if f.endswith('.pdf')]
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    Anda adalah asisten AI resmi Inspektorat Sultra. Jawablah berdasarkan dokumen yang ada.
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("🤖 Asisten SCAN: ", response["output_text"])

def main():
    st.set_page_config("SCAN AI")
    st.header("Asisten AI Inspektorat Sultra 🏛️")

    if "processed" not in st.session_state:
        with st.spinner("Mempelajari dokumen..."):
            raw_text = get_pdf_text()
            if raw_text:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                chunks = text_splitter.split_text(raw_text)
                get_vector_store(chunks)
                st.session_state.processed = True
                st.success("Siap menjawab!")
            else:
                st.error("File PDF tidak ditemukan.")

    question = st.text_input("Tanyakan isi Renstra/Dokumen:")
    if question:
        user_input(question)

if __name__ == "__main__":
    main()
