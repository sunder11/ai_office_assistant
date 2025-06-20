import os
from dotenv import load_dotenv
import streamlit as st
from docling.document_converter import DocumentConverter
from langchain.schema import Document
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# load the environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document(file_path):
    """Load a single document"""
    # Initialize the DocumentConverter
    converter = DocumentConverter()
    # Convert the HTML document
    result = converter.convert(file_path)
    # Extract text content and create LangChain Document objects
    documents = []
    # Method 1: Try to get the full text/markdown export
    try:
        # Get the full document text as markdown
        full_text = result.document.export_to_markdown()
        if full_text and full_text.strip():
            # Create a single document with the full text
            doc = Document(
                page_content=full_text,
                metadata={
                    "source": file_path,
                    "filename": os.path.basename(file_path),
                    "format": "markdown",
                },
            )
            documents.append(doc)
    except AttributeError:
        # If markdown export is not available, try alternative methods
        pass
    # Method 2: If no markdown export, try to access the document content directly
    if not documents:
        try:
            # Check if document has a text attribute or method
            if hasattr(result.document, "text"):
                text_content = result.document.text
            elif hasattr(result.document, "get_text"):
                text_content = result.document.get_text()
            elif hasattr(result, "text"):
                text_content = result.text
            else:
                # Try to convert the document to string
                text_content = str(result.document)
            if text_content and text_content.strip():
                doc = Document(
                    page_content=text_content,
                    metadata={
                        "source": file_path,
                        "filename": os.path.basename(file_path),
                    },
                )
                documents.append(doc)
        except Exception as e:
            st.warning(
                f"Error extracting text from {os.path.basename(file_path)}: {str(e)}"
            )
    # Method 3: If still no documents, try to access raw content
    if not documents:
        try:
            # Check if there's a content attribute
            if hasattr(result, "content"):
                content = result.content
                if content and str(content).strip():
                    doc = Document(
                        page_content=str(content),
                        metadata={
                            "source": file_path,
                            "filename": os.path.basename(file_path),
                        },
                    )
                    documents.append(doc)
        except Exception:
            pass
    # If still no documents were created, raise an error
    if not documents:
        raise ValueError(f"Could not extract text from HTML file: {file_path}")
    return documents

def load_documents_from_folder(folder_path):
    """Load all HTML documents from a folder"""
    all_documents = []
    html_files = []
    # Find all HTML files in the folder
    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".html", ".htm")):
            html_files.append(os.path.join(folder_path, filename))
    if not html_files:
        raise ValueError(f"No HTML files found in folder: {folder_path}")
    # Process each HTML file
    progress_bar = st.progress(0)
    status_text = st.empty()
    for idx, file_path in enumerate(html_files):
        try:
            status_text.text(f"Processing {os.path.basename(file_path)}...")
            docs = load_document(file_path)
            all_documents.extend(docs)
            progress_bar.progress((idx + 1) / len(html_files))
        except Exception as e:
            st.warning(f"Failed to process {os.path.basename(file_path)}: {str(e)}")
            continue
    progress_bar.empty()
    status_text.empty()
    if not all_documents:
        raise ValueError("No documents could be loaded from the folder")
    return all_documents, html_files

def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
    )
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

def create_chain(vectorstore):
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(
        llm=llm, output_key="answer", memory_key="chat_history", return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",
        memory=memory,
        verbose=True,
    )
    return chain

st.set_page_config(page_title="Chat with Docs", page_icon="", layout="centered")
st.title("Chat with Docs - LLAMA 3.1")
# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "loaded_files" not in st.session_state:
    st.session_state.loaded_files = []
# Document loading options
st.subheader("Load Documents")
upload_option = st.radio(
    "Choose upload option:", ["Upload single HTML", "Process folder of HTML"]
)
if upload_option == "Upload single HTML":
    uploaded_file = st.file_uploader(label="Upload your HTML file", type=["html", "htm"])
    if uploaded_file:
        file_path = f"{working_dir}/{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        # Check if file already processed
        if uploaded_file.name not in st.session_state.loaded_files:
            if "vectorstore" not in st.session_state:
                try:
                    with st.spinner("Processing HTML..."):
                        documents = load_document(file_path)
                        st.success(f"Successfully loaded {len(documents)} document(s)")
                        st.session_state.vectorstore = setup_vectorstore(documents)
                        st.session_state.loaded_files = [uploaded_file.name]
                except Exception as e:
                    st.error(f"Error processing HTML: {str(e)}")
        else:
            # Add to existing vectorstore
            try:
                with st.spinner("Adding HTML to existing collection..."):
                    documents = load_document(file_path)
                    st.session_state.vectorstore.add_documents(documents)
                    st.session_state.loaded_files.append(uploaded_file.name)
                    st.success(f"Added {uploaded_file.name} to collection")
            except Exception as e:
                st.error(f"Error adding HTML: {str(e)}")
else:  # Process folder of HTMLs
    col1, col2 = st.columns([3, 1])
    with col1:
        folder_path = st.text_input(
            "Enter folder path containing HTML files:",
            placeholder="e.g., /mnt/c/html or /home/user/html",
        )
    with col2:
        process_button = st.button("Process Folder", type="primary")
    if process_button and folder_path:
        # Validate folder
        if not os.path.exists(folder_path):
            st.error(f"Folder not found: {folder_path}")
        elif not os.path.isdir(folder_path):
            st.error(f"Path is not a folder: {folder_path}")
        else:
            try:
                with st.spinner("Processing HTML files in folder..."):
                    documents, html_files = load_documents_from_folder(folder_path)
                    st.success(
                        f"Successfully loaded {len(documents)} document(s) from {len(html_files)} HTML files"
                    )
                    # Show processed files
                    with st.expander("Processed files", expanded=True):
                        for html_file in html_files:
                            st.write(f"{os.path.basename(html_file)}")
                    # Create or update vectorstore
                    if "vectorstore" not in st.session_state:
                        st.session_state.vectorstore = setup_vectorstore(documents)
                    else:
                        st.session_state.vectorstore.add_documents(documents)
                    st.session_state.loaded_files.extend(
                        [os.path.basename(f) for f in html_files]
                    )
            except Exception as e:
                st.error(f"Error processing folder: {str(e)}")
# Show loaded documents
if st.session_state.loaded_files:
    with st.sidebar:
        st.subheader("Loaded Documents")
        for file in st.session_state.loaded_files:
            st.write(f"{file}")
        if st.button("Clear All Documents"):
            for key in [
                "vectorstore",
                "conversation_chain",
                "loaded_files",
                "chat_history",
            ]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
# Initialize conversation chain if vectorstore exists
if "vectorstore" in st.session_state and "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)
# Chat interface
if "vectorstore" in st.session_state:
    st.divider()
    st.subheader("Chat with your documents")
    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Chat input
    user_input = st.chat_input("Ask about your documents...")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            response = st.session_state.conversation_chain({"question": user_input})
            assistant_response = response["answer"]
            st.markdown(assistant_response)
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_response}
            )
else:
    st.info("Please upload an HTML file or select a folder to start chatting")