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
import xml.etree.ElementTree as ET
from typing import List
from urllib.parse import urljoin
import requests
import time
import random
import sqlite3
import datetime
import threading

# Load the environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))

# Define data directory for persistent storage
DATA_DIR = os.path.join(working_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "website_data.db")

# Thread-local storage for database connections
_local = threading.local()


def get_db_connection():
    """Get a thread-specific database connection"""
    if not hasattr(_local, "conn"):
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        # Enable foreign keys
        _local.conn.execute("PRAGMA foreign_keys = ON")
    return _local.conn


def close_db_connection():
    """Close the thread-specific database connection if it exists"""
    if hasattr(_local, "conn"):
        _local.conn.close()
        delattr(_local, "conn")


def setup_database():
    """Set up SQLite database for document storage"""
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create tables for processed URLs and document content
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS websites (
        id INTEGER PRIMARY KEY,
        sitemap_url TEXT UNIQUE,
        processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS pages (
        id INTEGER PRIMARY KEY,
        website_id INTEGER,
        url TEXT UNIQUE,
        title TEXT,
        content TEXT,
        processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (website_id) REFERENCES websites(id)
    )
    """)

    # Create table for chat history
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS chat_history (
        id INTEGER PRIMARY KEY,
        role TEXT,
        content TEXT,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    return True


def save_vectorstore(vectorstore, save_path=DATA_DIR):
    """Save FAISS vectorstore to disk"""
    os.makedirs(save_path, exist_ok=True)
    vectorstore.save_local(f"{save_path}/faiss_index")


def load_vectorstore(load_path=DATA_DIR):
    """Load FAISS vectorstore from disk"""
    if os.path.exists(f"{load_path}/faiss_index"):
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.load_local(
            f"{load_path}/faiss_index",
            embeddings,
            allow_dangerous_deserialization=True,  # Add this parameter
        )
        return vectorstore
    return None


def save_processed_content(sitemap_url, processed_urls, documents):
    """Save processed content to SQLite database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Insert or update website record
        cursor.execute(
            "INSERT OR REPLACE INTO websites (sitemap_url, processed_date) VALUES (?, ?)",
            (sitemap_url, datetime.datetime.now()),
        )
        website_id = (
            cursor.lastrowid
            or cursor.execute(
                "SELECT id FROM websites WHERE sitemap_url = ?", (sitemap_url,)
            ).fetchone()[0]
        )

        # Insert document records
        for doc, url in zip(documents, processed_urls):
            # Extract title if available in metadata
            title = doc.metadata.get("title", os.path.basename(url))

            cursor.execute(
                "INSERT OR REPLACE INTO pages (website_id, url, title, content, processed_date) VALUES (?, ?, ?, ?, ?)",
                (website_id, url, title, doc.page_content, datetime.datetime.now()),
            )

        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Database error: {str(e)}")


def load_processed_content(sitemap_url=None):
    """Load processed content from SQLite database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    if sitemap_url:
        cursor.execute(
            """
            SELECT p.url, p.title, p.content
            FROM pages p
            JOIN websites w ON p.website_id = w.id
            WHERE w.sitemap_url = ?
            """,
            (sitemap_url,),
        )
    else:
        cursor.execute("SELECT url, title, content FROM pages")

    results = cursor.fetchall()

    documents = []
    processed_urls = []

    for url, title, content in results:
        doc = Document(
            page_content=content, metadata={"source": url, "url": url, "title": title}
        )
        documents.append(doc)
        processed_urls.append(url)

    return documents, processed_urls


def save_chat_history(chat_history):
    """Save chat history to database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        # Clear existing history
        cursor.execute("DELETE FROM chat_history")

        # Insert new history
        for idx, message in enumerate(chat_history):
            cursor.execute(
                "INSERT INTO chat_history (id, role, content) VALUES (?, ?, ?)",
                (idx, message["role"], message["content"]),
            )

        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Error saving chat history: {str(e)}")


def load_chat_history():
    """Load chat history from database"""
    conn = get_db_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("SELECT role, content FROM chat_history ORDER BY id")
        results = cursor.fetchall()

        chat_history = []
        for role, content in results:
            chat_history.append({"role": role, "content": content})

        return chat_history
    except:
        return []


def get_sitemap_urls(sitemap_url: str) -> List[str]:
    """
    Extracts all URLs from a sitemap XML file.
    Handles both regular sitemaps and sitemap indexes.
    """
    try:
        response = requests.get(sitemap_url, timeout=30)
        response.raise_for_status()

        root = ET.fromstring(response.content)

        # Define XML namespaces
        namespaces = {
            "sm": "http://www.sitemaps.org/schemas/sitemap/0.9",
            "xhtml": "http://www.w3.org/1999/xhtml",
        }

        urls = []

        # Check if this is a sitemap index (contains other sitemaps)
        sitemap_tags = root.findall(".//sm:sitemap", namespaces)
        if sitemap_tags:
            # This is a sitemap index
            for sitemap_tag in sitemap_tags:
                loc_tag = sitemap_tag.find("sm:loc", namespaces)
                if loc_tag is not None and loc_tag.text:
                    # Recursively get URLs from each child sitemap
                    child_sitemap_url = loc_tag.text.strip()
                    child_urls = get_sitemap_urls(child_sitemap_url)
                    urls.extend(child_urls)
        else:
            # This is a regular sitemap
            url_tags = root.findall(".//sm:url", namespaces)
            for url_tag in url_tags:
                loc_tag = url_tag.find("sm:loc", namespaces)
                if loc_tag is not None and loc_tag.text:
                    urls.append(loc_tag.text.strip())

        return urls
    except Exception as e:
        st.error(f"Error processing sitemap: {str(e)}")
        return []


def fetch_html_content(url: str) -> str:
    """Fetch HTML content from a URL"""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        st.warning(f"Error fetching HTML from {url}: {str(e)}")
        return ""


def process_url_with_docling(url: str, temp_dir: str) -> Document:
    """Process a URL's HTML content with Docling"""
    html_content = fetch_html_content(url)
    if not html_content:
        return None

    # Create a temporary HTML file
    url_filename = url.replace(":", "_").replace("/", "_").replace(".", "_") + ".html"
    temp_file_path = os.path.join(temp_dir, url_filename)

    with open(temp_file_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    # Process with Docling
    try:
        converter = DocumentConverter()
        result = converter.convert(temp_file_path)

        # Extract page title if possible
        title = url
        try:
            import re

            title_match = re.search("<title>(.*?)</title>", html_content, re.IGNORECASE)
            if title_match:
                title = title_match.group(1)
        except:
            pass

        # Try to get the full text as markdown
        try:
            full_text = result.document.export_to_markdown()
            if full_text and full_text.strip():
                return Document(
                    page_content=full_text,
                    metadata={
                        "source": url,
                        "url": url,
                        "title": title,
                        "format": "markdown",
                    },
                )
        except AttributeError:
            pass

        # If no markdown export, try to access content directly
        if hasattr(result.document, "text"):
            text_content = result.document.text
        elif hasattr(result.document, "get_text"):
            text_content = result.document.get_text()
        elif hasattr(result, "text"):
            text_content = result.text
        else:
            text_content = str(result.document)

        if text_content and text_content.strip():
            return Document(
                page_content=text_content,
                metadata={
                    "source": url,
                    "url": url,
                    "title": title,
                },
            )

        # If still no content, try to access raw content
        if hasattr(result, "content"):
            content = result.content
            if content and str(content).strip():
                return Document(
                    page_content=str(content),
                    metadata={
                        "source": url,
                        "url": url,
                        "title": title,
                    },
                )

        return None
    except Exception as e:
        st.warning(f"Error processing URL with Docling: {url} - {str(e)}")
        return None
    finally:
        # Clean up the temp file
        if os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


def process_sitemap(sitemap_url: str, max_urls: int = None):
    """Process a sitemap and extract content from all URLs"""
    with st.spinner("Fetching sitemap URLs..."):
        urls = get_sitemap_urls(sitemap_url)
        if not urls:
            st.error("No URLs found in the sitemap.")
            return [], []

        st.success(f"Found {len(urls)} URLs in the sitemap.")

        if max_urls and max_urls < len(urls):
            st.info(f"Processing only {max_urls} URLs out of {len(urls)}.")
            urls = urls[:max_urls]

    # Create a temporary directory for HTML files
    temp_dir = os.path.join(working_dir, "temp_html")
    os.makedirs(temp_dir, exist_ok=True)

    all_documents = []
    processed_urls = []

    # Process each URL
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, url in enumerate(urls):
        status_text.text(f"Processing URL {idx + 1}/{len(urls)}: {url}")

        try:
            # Add some delay to avoid overloading the server
            time.sleep(random.uniform(0.5, 2.0))

            document = process_url_with_docling(url, temp_dir)
            if document:
                all_documents.append(document)
                processed_urls.append(url)

            progress_bar.progress((idx + 1) / len(urls))
        except Exception as e:
            st.warning(f"Failed to process {url}: {str(e)}")
            continue

    progress_bar.empty()
    status_text.empty()

    # Clean up the temp directory
    try:
        import shutil

        shutil.rmtree(temp_dir)
    except:
        pass

    if not all_documents:
        st.error("No documents could be extracted from the sitemap URLs.")

    return all_documents, processed_urls


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


def initialize_storage():
    """Initialize storage and load existing data"""
    # Set up SQLite database
    setup_database()

    # Try to load existing vectorstore
    vectorstore = load_vectorstore(DATA_DIR)

    if vectorstore:
        st.session_state.vectorstore = vectorstore

        # Load URLs from database to session state
        documents, urls = load_processed_content()
        st.session_state.processed_urls = urls

        # Load chat history if available
        chat_history = load_chat_history()
        if chat_history:
            st.session_state.chat_history = chat_history

        st.sidebar.success(f"Loaded {len(urls)} previously processed URLs")

    return True


# Streamlit app configuration
st.set_page_config(page_title="Chat with Website", page_icon="", layout="wide")
st.title("Chat with Website - LLAMA 3.3")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = []

# Initialize storage when app starts
if "storage_initialized" not in st.session_state:
    initialize_storage()
    st.session_state.storage_initialized = True

# Document loading options
st.subheader("Load Website Content")

sitemap_col, options_col = st.columns([3, 1])

with sitemap_col:
    sitemap_url = st.text_input(
        "Enter website sitemap URL:",
        placeholder="e.g., https://example.com/sitemap.xml",
    )

with options_col:
    max_urls = st.slider(
        "Max URLs",
        min_value=1,
        max_value=100,
        value=50,
        help="Limit the number of URLs to process",
    )

process_button = st.button("Process Sitemap", type="primary")

if process_button and sitemap_url:
    try:
        # Process the sitemap
        documents, processed_urls = process_sitemap(sitemap_url, max_urls)

        if documents:
            st.success(
                f"Successfully processed {len(documents)} pages from {len(processed_urls)} URLs"
            )

            # Create or update vectorstore
            if "vectorstore" not in st.session_state:
                st.session_state.vectorstore = setup_vectorstore(documents)
            else:
                st.session_state.vectorstore.add_documents(documents)

            # Save data to persistent storage
            save_vectorstore(st.session_state.vectorstore)
            save_processed_content(sitemap_url, processed_urls, documents)

            st.session_state.processed_urls.extend(processed_urls)

            # Show processed URLs
            with st.expander("Processed URLs", expanded=True):
                for url in processed_urls:
                    st.write(url)
    except Exception as e:
        st.error(f"Error processing sitemap: {str(e)}")

# Show loaded documents/URLs in sidebar
with st.sidebar:
    st.subheader("Processed Content")

    if st.session_state.processed_urls:
        st.write(f"Total URLs: {len(st.session_state.processed_urls)}")

        with st.expander("View all URLs", expanded=False):
            for url in st.session_state.processed_urls:
                st.write(f"- {url}")

        # Initialize confirmation state
        if "db_delete_confirmation" not in st.session_state:
            st.session_state.db_delete_confirmation = False

        col1, col2 = st.columns(2)

        with col1:
            # Display different buttons based on confirmation state
            if not st.session_state.db_delete_confirmation:
                # Initial button
                if st.button("Clear Database", type="primary"):
                    st.session_state.db_delete_confirmation = True
                    st.rerun()
            else:
                # Show warning and confirmation buttons
                st.warning(
                    "⚠️ Are you sure you want to delete all data? This cannot be undone."
                )

                # Yes button
                if st.button("Yes, Delete", type="primary"):
                    # Clear tables but keep structure
                    conn = get_db_connection()
                    cursor = conn.cursor()

                    # Delete in the correct order to respect foreign key constraints
                    # First delete from pages (child table)
                    cursor.execute("DELETE FROM pages")
                    # Then delete from websites (parent table)
                    cursor.execute("DELETE FROM websites")

                    conn.commit()

                    # Remove vector store files
                    import shutil

                    if os.path.exists(os.path.join(DATA_DIR, "faiss_index")):
                        shutil.rmtree(os.path.join(DATA_DIR, "faiss_index"))

                    # Clear session state but keep chat history
                    for key in [
                        "vectorstore",
                        "conversation_chain",
                        "processed_urls",
                        "db_delete_confirmation",
                    ]:
                        if key in st.session_state:
                            del st.session_state[key]

                    st.success("Database cleared successfully!")
                    st.rerun()

                # No button
                if st.button("No, Cancel"):
                    st.session_state.db_delete_confirmation = False
                    st.rerun()

        with col2:
            if st.button("Clear Chat History", type="secondary"):
                # Clear only chat history
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("DELETE FROM chat_history")
                conn.commit()

                # Clear chat history from session state
                st.session_state.chat_history = []

                # If conversation chain exists, reset its memory
                if "conversation_chain" in st.session_state:
                    # Create a new memory object
                    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
                    memory = ConversationBufferMemory(
                        llm=llm,
                        output_key="answer",
                        memory_key="chat_history",
                        return_messages=True,
                    )
                    # Update the chain with the new memory
                    st.session_state.conversation_chain.memory = memory

                st.success("Chat history cleared!")
                st.rerun()
    else:
        st.info("No content processed yet. Enter a sitemap URL to begin.")


# Add this after the sidebar section and before the chat interface
if "vectorstore" in st.session_state:
    st.divider()
    export_container = st.container()

    with export_container:
        st.subheader("Data Export")

        # Add export button for FAISS vectors
        if st.button("Export Vectors to CSV", type="primary"):
            with st.spinner("Exporting vectors..."):
                # Define the export function
                def export_faiss_vectors(filename="faiss_vectors.csv"):
                    """Export FAISS vectors to CSV for external visualization"""
                    try:
                        vs = st.session_state.vectorstore

                        # Get all the vectors and their IDs
                        vectors = []
                        ids = []
                        metadata = []

                        # Different vectorstores have different structures
                        if hasattr(vs, "index_to_docstore_id") and hasattr(vs, "index"):
                            for i in range(len(vs.index_to_docstore_id)):
                                doc_id = vs.index_to_docstore_id.get(i)
                                if doc_id and doc_id in vs.docstore._dict:
                                    # Get the document
                                    doc = vs.docstore._dict[doc_id]
                                    # Get the vector
                                    vector = vs.index.reconstruct(i)

                                    vectors.append(vector)
                                    ids.append(doc_id)
                                    metadata.append(
                                        {
                                            "source": doc.metadata.get("source", ""),
                                            "title": doc.metadata.get("title", ""),
                                            "content_preview": doc.page_content[:100],
                                        }
                                    )

                        # Export to CSV
                        import pandas as pd
                        import numpy as np

                        # Create a dataframe with metadata
                        meta_df = pd.DataFrame(metadata)

                        # Create a dataframe with vectors
                        vector_df = pd.DataFrame(np.array(vectors))

                        # Combine the dataframes
                        result_df = pd.concat([meta_df, vector_df], axis=1)

                        # Save to CSV
                        result_df.to_csv(filename, index=False)

                        return True, f"Exported {len(vectors)} vectors to {filename}"
                    except Exception as e:
                        return False, f"Error exporting vectors: {str(e)}"

                # Export the vectors to a file in the data directory
                export_path = os.path.join(DATA_DIR, "faiss_vectors.csv")
                success, message = export_faiss_vectors(export_path)

                if success:
                    st.success(message)
                    st.info(f"File saved to: {os.path.abspath(export_path)}")
                    # Create a download link for the exported file
                    with open(export_path, "rb") as file:
                        st.download_button(
                            label="Download CSV File",
                            data=file,
                            file_name="faiss_vectors.csv",
                            mime="text/csv",
                        )
                else:
                    st.error(message)

        # Add a note about SQLite database location
        st.info(
            f"SQLite database location: {os.path.abspath(DB_PATH)}\nUse your preferred SQLite browser to view and modify the database."
        )


# Initialize conversation chain if vectorstore exists
if "vectorstore" in st.session_state and "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

# Chat interface
if "vectorstore" in st.session_state:
    st.divider()
    st.subheader("Chat with your website content")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask about the website content...")
    if user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state.conversation_chain({"question": user_input})
                assistant_response = response["answer"]
                st.markdown(assistant_response)

                # Add assistant response to chat history
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": assistant_response}
                )

                # Save chat history to database
                save_chat_history(st.session_state.chat_history)
else:
    st.info("Please process a website sitemap to start chatting")


# Close database connection when app is done
def on_shutdown():
    close_db_connection()


# Register shutdown handler
import atexit

atexit.register(on_shutdown)
