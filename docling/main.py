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
    # Initialize the DocumentConverter
    converter = DocumentConverter()

    # Convert the PDF document
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
                metadata={"source": file_path, "format": "markdown"},
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
                    page_content=text_content, metadata={"source": file_path}
                )
                documents.append(doc)
        except Exception as e:
            st.warning(f"Error extracting text: {str(e)}")

    # Method 3: If still no documents, try to access raw content
    if not documents:
        try:
            # Check if there's a content attribute
            if hasattr(result, "content"):
                content = result.content
                if content and str(content).strip():
                    doc = Document(
                        page_content=str(content), metadata={"source": file_path}
                    )
                    documents.append(doc)
        except Exception:
            pass

    # If still no documents were created, raise an error
    if not documents:
        raise ValueError(f"Could not extract text from PDF file: {file_path}")

    return documents


def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Fixed: was "/n", should be "\n"
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


st.set_page_config(page_title="Chat with Doc", page_icon="📄", layout="centered")
st.title("🦙 Chat with Doc - LLAMA 3.1")

# initialize the chat history in streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader(label="Upload your pdf file", type=["pdf"])

if uploaded_file:
    file_path = f"{working_dir}/{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vectorstore" not in st.session_state:
        try:
            with st.spinner("Processing PDF..."):
                documents = load_document(file_path)
                st.success(f"Successfully loaded {len(documents)} document(s)")
                st.session_state.vectorstore = setup_vectorstore(documents)
        except Exception as e:
            st.error(f"Error processing PDF: {str(e)}")
            st.stop()

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Ask Llama...")

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
