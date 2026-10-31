import os
import shutil
from dotenv import load_dotenv
import streamlit as st

# Import your modular processing blocks
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.chunker import create_chunks
from src.vectorstore.pinecone_store import store_chunks 
from src.rag.pipeline import rag_pipeline

load_dotenv()

# App Configuration
st.set_page_config(
    page_title="Medical RAG Assistant", 
    page_icon="🩺", 
    layout="wide"
)

st.title("🩺 Medical RAG Assistant")

# Ensure a local directory exists for processing uploads safely
TEMP_DIR = "./docs"
os.makedirs(TEMP_DIR, exist_ok=True)

# --- SIDEBAR: DOCUMENT INGESTION MANAGEMENT ---
with st.sidebar:
    st.header("📄 Document Ingestion Panel")
    st.write("Upload medical literature PDFs to update your vector knowledge base.")
    
    uploaded_file = st.file_uploader(
        "Choose a PDF file", 
        type=["pdf"], 
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # Prevent redundant reprocessing if the file name hasn't changed
        if "last_uploaded" not in st.session_state or st.session_state.last_uploaded != uploaded_file.name:
            
            temp_file_path = os.path.join(TEMP_DIR, uploaded_file.name)
            
            with st.status("Processing your document...", expanded=True) as status:
                try:
                    # Step 1: Write file stream bytes to temporary disk storage
                    status.write("Saving file locally...")
                    with open(temp_file_path, "wb") as buffer:
                        shutil.copyfileobj(uploaded_file, buffer)
                    
                    # Step 2: Extract text representations using PyMuPDF wrapper
                    status.write("Extracting text layers from PDF pages...")
                    documents = load_pdf(temp_file_path)
                    
                    # Step 3: Run recursive boundary sliding split partitioner
                    status.write("Partitioning text bodies into semantic blocks...")
                    chunks = create_chunks(documents)
                    
                    # Step 4: Stream vectorized token items down to Pinecone index
                    status.write("Pushing token fragments up to Pinecone cluster...")
                    store_chunks (chunks)
                    
                    status.update(label="✅ Ingestion Success!", state="complete", expanded=False)
                    st.sidebar.success(f"Successfully processed and indexed: **{uploaded_file.name}**")
                    
                    # Track operational history state token
                    st.session_state.last_uploaded = uploaded_file.name
                    
                except Exception as error:
                    status.update(label="❌ Ingestion Failed", state="error", expanded=True)
                    st.sidebar.error(f"Error lifecycle details: {str(error)}")
                finally:
                    # Clean up file disk cache trace when finished
                    if os.path.exists(temp_file_path):
                        os.remove(temp_file_path)
        else:
            st.sidebar.info(f"💡 **{uploaded_file.name}** is already fully parsed and active inside your index.")

# --- MAIN PAGE: Q&A QUERY CHAT INTERFACE ---
st.write("Ask questions based on the uploaded medical document.")

question = st.chat_input("Enter your question...")

if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching document and generating answer..."):
            try:
                # Query pipeline orchestration engine
                result = rag_pipeline(question)

                # Extract and unpack values using your custom structural format
                st.write(result["answer"])

                with st.expander("Retrieved sources"):
                    for source in result["sources"]:
                        st.write(f"📄 {source['title']}")
                        st.write(f"Relevance score: {source['score']:.4f}")
                        st.write("---")

            except Exception as e:
                st.error(f"Something went wrong: {str(e)}")

st.caption(
    "This application provides document-based information, "
    "not a medical diagnosis or substitute for professional care."
)
