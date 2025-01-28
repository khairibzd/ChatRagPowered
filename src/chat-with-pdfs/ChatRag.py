
import streamlit as st

"""
This script provides a Streamlit application for uploading PDF files, processing their content, and performing question-answering tasks using a language model.
Modules:
    - streamlit: For creating the web interface.
    - langchain_community.document_loaders: For loading PDF documents.
    - langchain_text_splitters: For splitting text into manageable chunks.
    - langchain_core.vectorstores: For storing and retrieving document vectors.
    - langchain_ollama: For embeddings and language model.
    - langchain_core.prompts: For creating chat prompts.
Functions:
    - upload_pdf(file): Saves the uploaded PDF file to the specified directory.
    - load_pdf(file_path): Loads the PDF file and extracts its content.
    - split_text(documents): Splits the loaded documents into smaller chunks.
    - index_docs(documents): Indexes the document chunks into the vector store.
    - retrieve_docs(query): Retrieves documents from the vector store based on the query.
    - answer_question(question, documents): Generates an answer to the question using the retrieved documents and the language model.
Streamlit Interface:
    - Allows users to upload a PDF file.
    - Processes the uploaded file and indexes its content.
    - Accepts user questions and provides answers based on the indexed content.

"""
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.prompts import ChatPromptTemplate



template = """
You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
Question: {question} 
Context: {context} 
Answer:
"""


pdfs_directory = 'chat-with-pdfs/pdfs/'
embeddings = OllamaEmbeddings(model="deepseek-r1:7b")
vector_store = InMemoryVectorStore(embeddings)

model = OllamaLLM(model="deepseek-r1:7b")

def upload_pdf(file):
    with open(pdfs_directory + file.name, "wb") as f:
        f.write(file.getbuffer())

def load_pdf(file_path):
    loader = PDFPlumberLoader(file_path)
    documents = loader.load()

    return documents

def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )

    return text_splitter.split_documents(documents)

def index_docs(documents):
    vector_store.add_documents(documents)

def retrieve_docs(query):
    return vector_store.similarity_search(query)

# The function will return the model's answer to the question based on the provided documents.
def answer_question(question, documents):
    context = "\n\n".join([doc.page_content for doc in documents])
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model

    return chain.invoke({"question": question, "context": context})

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf",
    accept_multiple_files=False
)

if uploaded_file:
    upload_pdf(uploaded_file)
    documents = load_pdf(pdfs_directory + uploaded_file.name)
    chunked_documents = split_text(documents)
    index_docs(chunked_documents)

    question = st.chat_input()

    if question:
        st.chat_message("user").write(question)
        related_documents = retrieve_docs(question)
        answer = answer_question(question, related_documents)
        st.chat_message("assistant").write(answer)  