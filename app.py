import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API key is not configured. Please check your environment variables.")
    st.stop()

# Configure Google Generative AI
genai.configure(api_key=api_key)

# Function to extract text from PDF
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        if pdf.name.endswith(".pdf"):
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        else:
            st.warning(f"Skipping non-PDF file: {pdf.name}")
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save the FAISS vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local("faiss_index")
    except Exception as e:
        st.error(f"Error saving FAISS index: {str(e)}")
        st.stop()

# Function to set up the conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say, "answer is not available in the context." 
    Do not provide incorrect answers.\n\n
    Context:\n {context}\n
    Question:\n {question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to process user input and generate a response
def user_input(user_question):
    if not user_question.strip():
        st.error("Question cannot be empty!")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    try:
        # Load the FAISS vector store
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Error loading FAISS index: {str(e)}")
        return

    try:
        # Perform similarity search
        docs = new_db.similarity_search(user_question)
    except Exception as e:
        st.error(f"Error during similarity search: {str(e)}")
        return

    # Get the conversational chain and generate a response
    chain = get_conversational_chain()
    try:
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        st.write("Reply: ", response.get("output_text", "No response available"))
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")

# Main Streamlit app
def main():
    st.set_page_config(page_title="Chat with PDFs", layout="wide")
    st.header("Chat with Multiple PDFs Using Gemini 💁")

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Upload PDF Files:")
        pdf_docs = st.file_uploader(
            "Upload multiple PDFs to build your knowledge base.",
            accept_multiple_files=True,
            type=["pdf"]
        )

        if st.button("Submit & Process"):
            with st.spinner("Processing files..."):
                if not pdf_docs:
                    st.warning("No PDF files uploaded. Please upload at least one PDF.")
                else:
                    raw_text = get_pdf_text(pdf_docs)
                    if not raw_text.strip():
                        st.error("No readable text found in the uploaded files.")
                    else:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("Files processed successfully! You can now ask questions.")

    # Text input for user query
    user_question = st.text_input(
        "Ask a question based on the uploaded PDFs:",
        placeholder="Example: Who has 1 year of Python experience?"
    )
    if user_question:
        user_input(user_question)  # Pass the text input value to the function

# Entry point
if __name__ == "__main__":
    main()
