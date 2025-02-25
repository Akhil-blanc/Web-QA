import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

st.set_page_config(
    page_title="URL Content Q&A Tool",
    page_icon="🔍",
    layout="centered"
)

# Initialize session state
if 'urls' not in st.session_state:
    st.session_state.urls = []
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'loaded_urls' not in st.session_state:
    st.session_state.loaded_urls = []

# Function to load and process URLs
def process_urls(urls):
    # Show loading message
    with st.spinner('Loading and processing URLs...'):
        try:
            # Load content from URLs
            loader = WebBaseLoader(urls)
            documents = loader.load()
            
            # Split documents into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            splits = text_splitter.split_documents(documents)
            
            # Create vector store with Gemini embeddings
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vectorstore = FAISS.from_documents(splits, embeddings)
            
            # Create QA chain with Gemini model
            llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=0.2,
                convert_system_message_to_human=True
            )

            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

            combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
            rag_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)
            
            # qa_chain = RetrievalQA.from_chain_type(
            #     llm=llm,
            #     chain_type="stuff",
            #     retriever=vectorstore.as_retriever(),
            #     return_source_documents=True
            # )
            
            return rag_chain
        
        except Exception as e:
            st.error(f"Error processing URLs: {str(e)}")
            return None

# App header
st.title("URL Content Q&A Tool")
st.markdown("Ask questions about the content of specific URLs without relying on general knowledge.")

# URL input section
with st.expander("Add URLs", expanded=True):
    col1, col2 = st.columns([3, 1])
    
    with col1:
        url_input = st.text_input("Enter a URL:", placeholder="https://example.com")
    
    with col2:
        add_url = st.button("Add URL", use_container_width=True)
    
    if add_url and url_input:
        if url_input not in st.session_state.urls:
            st.session_state.urls.append(url_input)
            st.session_state.qa_chain = None  # Reset QA chain when URLs change
        url_input = ""

# Display added URLs
if st.session_state.urls:
    st.subheader("Added URLs:")
    
    for i, url in enumerate(st.session_state.urls):
        col1, col2 = st.columns([5, 1])
        with col1:
            st.text(url)
        with col2:
            if st.button("Remove", key=f"remove_{i}"):
                st.session_state.urls.pop(i)
                st.session_state.qa_chain = None  # Reset QA chain when URLs change
                st.rerun()
    
    # Process URLs button
    if st.button("Process URLs", use_container_width=True):
        st.session_state.qa_chain = process_urls(st.session_state.urls)
        if st.session_state.qa_chain:
            st.session_state.loaded_urls = st.session_state.urls.copy()
            st.success("URLs processed successfully!")
else:
    st.info("Add URLs to get started.")

# Q&A section
st.divider()
st.subheader("Ask Questions")

if st.session_state.qa_chain:
    # Display which URLs are currently loaded
    st.markdown("**Currently loaded URLs:**")
    for url in st.session_state.loaded_urls:
        st.markdown(f"- {url}")
    
    # Question input
    question = st.text_input("Enter your question about the content:")
    
    if st.button("Get Answer") and question:
        with st.spinner('Searching for answer...'):
            try:
                result = st.session_state.qa_chain.invoke({"input": question})
                
                st.markdown("### Answer")
                st.markdown(result["answer"])
                
                # Display sources
                st.markdown("### Sources")
                for i, doc in enumerate(result["context"]):
                    with st.expander(f"Source {i+1}"):
                        st.markdown(f"**Content:**\n{doc.page_content}")
                        st.markdown(f"**Source URL:** {doc.metadata.get('source', 'Unknown')}")
            
            except Exception as e:
                st.error(f"Error generating answer: {str(e)}")
else:
    st.info("Process URLs to ask questions about their content.")

# API key warning
if not os.getenv("GOOGLE_API_KEY"):
    st.warning(
        "This app requires a Google API key. Add it to a .env file with the key GOOGLE_API_KEY."
    )