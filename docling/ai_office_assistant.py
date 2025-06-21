# Add these imports with the existing ones
from pptx import Presentation
import pandas as pd
import openpyxl
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
from pathlib import Path
import glob
import re

# Load the environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))
# Define data directory for persistent storage
DATA_DIR = os.path.join(working_dir, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "website_data.db")
# Thread-local storage for database connections
_local = threading.local()


def get_all_names_from_vectorstore_improved(vectorstore):
    """Extract ONLY real names from the vectorstore - FIXED for real names with punctuation"""
    try:
        all_names = set()

        if hasattr(vectorstore, "docstore") and hasattr(
            vectorstore, "index_to_docstore_id"
        ):
            for i in range(len(vectorstore.index_to_docstore_id)):
                doc_id = vectorstore.index_to_docstore_id.get(i)
                if doc_id and doc_id in vectorstore.docstore._dict:
                    doc = vectorstore.docstore._dict[doc_id]

                    # Only process Excel chunks
                    if doc.metadata.get("file_type") in ["xlsx", "xls"]:
                        content = doc.page_content

                        # ONLY use the most precise pattern for your data structure
                        name_pattern = (
                            r"FIRST_NAME:\s*([^|]+?)\s*\|\s*LAST_NAME:\s*([^|]+?)\s*\|"
                        )
                        matches = re.findall(name_pattern, content, re.IGNORECASE)

                        for first, last in matches:
                            first = first.strip()
                            last = last.strip()

                            # More realistic filtering for real names
                            if (
                                first
                                and last
                                and len(first) >= 2
                                and len(last) >= 2
                                and
                                # Allow letters, spaces, periods, commas, apostrophes, hyphens
                                re.match(r"^[A-Za-z\s\.\,\'\-]+$", first)
                                and re.match(
                                    r"^[A-Za-z\s\.\,\'\-IVX]+$", last
                                )  # Allow Roman numerals
                                and
                                # Exclude obvious non-names
                                not any(
                                    word.lower() in first.lower()
                                    for word in [
                                        "record",
                                        "structured",
                                        "data",
                                        "sheet",
                                        "column",
                                        "total",
                                        "fee",
                                    ]
                                )
                                and not any(
                                    word.lower() in last.lower()
                                    for word in [
                                        "record",
                                        "structured",
                                        "data",
                                        "sheet",
                                        "column",
                                        "total",
                                        "fee",
                                    ]
                                )
                                and
                                # Must start with a letter
                                first[0].isalpha()
                                and last[0].isalpha()
                            ):
                                full_name = f"{first} {last}"
                                all_names.add(full_name)

        return sorted(list(all_names))

    except Exception as e:
        st.error(f"Error extracting names: {str(e)}")
        return []


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
    # Create enhanced tables for all document types
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
    # New table for local documents
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY,
        file_path TEXT UNIQUE,
        file_name TEXT,
        file_type TEXT,
        source_type TEXT,
        content TEXT,
        processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            allow_dangerous_deserialization=True,
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


def save_local_documents(documents, file_paths, file_type):
    """Save locally processed documents to SQLite database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        for doc, file_path in zip(documents, file_paths):
            file_name = os.path.basename(file_path)
            cursor.execute(
                "INSERT OR REPLACE INTO documents (file_path, file_name, file_type, source_type, content, processed_date) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    file_path,
                    file_name,
                    file_type,
                    "local_directory",
                    doc.page_content,
                    datetime.datetime.now(),
                ),
            )
        conn.commit()
    except Exception as e:
        conn.rollback()
        st.error(f"Database error: {str(e)}")


def load_processed_content(sitemap_url=None):
    """Load processed content from SQLite database"""
    conn = get_db_connection()
    cursor = conn.cursor()
    documents = []
    processed_urls = []
    # Load web pages
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
    for url, title, content in results:
        doc = Document(
            page_content=content,
            metadata={"source": url, "url": url, "title": title, "type": "web"},
        )
        documents.append(doc)
        processed_urls.append(url)
    # Load local documents
    cursor.execute("SELECT file_path, file_name, content, file_type FROM documents")
    results = cursor.fetchall()
    for file_path, file_name, content, file_type in results:
        doc = Document(
            page_content=content,
            metadata={
                "source": file_path,
                "file_name": file_name,
                "file_type": file_type,
                "type": "local",
            },
        )
        documents.append(doc)
        processed_urls.append(file_path)
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


def process_powerpoint_file(file_path: str) -> Document:
    """Process a PowerPoint file and extract text content"""
    try:
        # First try with Docling (it might support PPTX)
        try:
            converter = DocumentConverter()
            result = converter.convert(file_path)
            title = os.path.basename(file_path)
            # Try to get markdown format first
            try:
                full_text = result.document.export_to_markdown()
                if full_text and full_text.strip():
                    return Document(
                        page_content=full_text,
                        metadata={
                            "source": file_path,
                            "file_name": title,
                            "file_type": "pptx",
                            "format": "markdown",
                        },
                    )
            except AttributeError:
                pass
            # Try other content extraction methods
            if hasattr(result.document, "text"):
                text_content = result.document.text
            elif hasattr(result.document, "get_text"):
                text_content = result.document.get_text()
            else:
                text_content = str(result.document)
            if text_content and text_content.strip():
                return Document(
                    page_content=text_content,
                    metadata={
                        "source": file_path,
                        "file_name": title,
                        "file_type": "pptx",
                    },
                )
        except Exception:
            # If Docling fails, fall back to python-pptx
            pass
        # Fallback to manual extraction using python-pptx
        prs = Presentation(file_path)
        text_content = []
        slide_count = 0
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_text = []
            slide_text.append(f"\n--- Slide {slide_num} ---\n")
            # Extract text from all shapes in the slide
            for shape in slide.shapes:
                if hasattr(shape, "text") and shape.text.strip():
                    slide_text.append(shape.text.strip())
                # Handle tables in slides
                if shape.shape_type == 19:  # Table
                    try:
                        table_text = []
                        for row in shape.table.rows:
                            row_text = []
                            for cell in row.cells:
                                row_text.append(cell.text.strip())
                            table_text.append(" | ".join(row_text))
                        slide_text.append("\n".join(table_text))
                    except:
                        pass
            if len(slide_text) > 1:  # More than just the slide header
                text_content.extend(slide_text)
                slide_count += 1
        if text_content:
            full_text = "\n".join(text_content)
            title = os.path.basename(file_path)
            return Document(
                page_content=full_text,
                metadata={
                    "source": file_path,
                    "file_name": title,
                    "file_type": "pptx",
                    "slide_count": slide_count,
                    "extraction_method": "python-pptx",
                },
            )
        return None
    except Exception as e:
        st.warning(f"Error processing PowerPoint file {file_path}: {str(e)}")
        return None


def process_excel_file(file_path: str) -> Document:
    """Process an Excel file with better structure preservation"""
    try:
        text_content = []
        sheet_count = 0
        try:
            excel_file = pd.ExcelFile(file_path, engine="openpyxl")
        except:
            try:
                excel_file = pd.ExcelFile(file_path, engine="xlrd")
            except:
                excel_file = pd.ExcelFile(file_path)
        for sheet_name in excel_file.sheet_names:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if df.empty:
                    continue
                sheet_text = [f"\n=== SHEET: {sheet_name} ===\n"]
                # Add structured data representation
                df_clean = df.fillna("")
                # For name-related data, create a more structured format
                if any(
                    col.lower() in ["name", "first_name", "last_name", "full_name"]
                    for col in df_clean.columns
                ):
                    sheet_text.append("STRUCTURED DATA - PEOPLE RECORDS:")
                    sheet_text.append("-" * 40)
                    for idx, row in df_clean.iterrows():
                        row_parts = []
                        for col, val in row.items():
                            if str(val).strip():
                                row_parts.append(f"{col}: {val}")
                        if row_parts:
                            sheet_text.append(
                                f"Record {idx + 1}: {' | '.join(row_parts)}"
                            )
                else:
                    # Standard table format for non-name data
                    if not df_clean.columns.empty:
                        headers = " | ".join(str(col) for col in df_clean.columns)
                        sheet_text.append(f"COLUMNS: {headers}")
                        sheet_text.append("-" * len(headers))
                    for idx, row in df_clean.iterrows():
                        row_text = " | ".join(
                            str(val) for val in row.values if str(val).strip()
                        )
                        if row_text.strip():
                            sheet_text.append(f"Row {idx + 1}: {row_text}")
                sheet_text.append(
                    f"\n[SHEET SUMMARY: {len(df)} total rows, {len(df.columns)} columns]"
                )
                sheet_text.append("=" * 50)
                text_content.extend(sheet_text)
                sheet_count += 1
            except Exception as e:
                st.warning(f"Error reading sheet '{sheet_name}': {str(e)}")
                continue
        if text_content:
            full_text = "\n".join(text_content)
            title = os.path.basename(file_path)
            return Document(
                page_content=full_text,
                metadata={
                    "source": file_path,
                    "file_name": title,
                    "file_type": "xlsx",
                    "sheet_count": sheet_count,
                    "extraction_method": "structured_pandas",
                    "total_content_length": len(full_text),
                },
            )
        return None
    except Exception as e:
        st.warning(f"Error processing Excel file {file_path}: {str(e)}")
        return None


def process_document_file(file_path: str, file_type: str) -> Document:
    """Process a single document file using appropriate method"""
    try:
        # Handle different file types with specialized functions
        if file_type.lower() == "pptx":
            return process_powerpoint_file(file_path)
        elif file_type.lower() in ["xlsx", "xls"]:
            return process_excel_file(file_path)
        else:
            # Use original Docling approach for other formats
            converter = DocumentConverter()
            result = converter.convert(file_path)
            title = os.path.basename(file_path)
            # Try to get the full text as markdown
            try:
                full_text = result.document.export_to_markdown()
                if full_text and full_text.strip():
                    return Document(
                        page_content=full_text,
                        metadata={
                            "source": file_path,
                            "file_name": title,
                            "file_type": file_type,
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
                        "source": file_path,
                        "file_name": title,
                        "file_type": file_type,
                    },
                )
            return None
    except Exception as e:
        st.warning(f"Error processing file {file_path}: {str(e)}")
        return None


def process_local_directory(
    directory_path: str, file_extension: str, max_files: int = None
):
    """Process all files of a specific type in a directory"""
    if not os.path.exists(directory_path):
        st.error(f"Directory not found: {directory_path}")
        return [], []
    # Get all files with the specified extension
    pattern = os.path.join(directory_path, f"*.{file_extension}")
    file_paths = glob.glob(pattern)
    if not file_paths:
        st.warning(f"No .{file_extension} files found in {directory_path}")
        return [], []
    st.success(f"Found {len(file_paths)} .{file_extension} files")
    # Limit number of files if specified
    if max_files and max_files < len(file_paths):
        st.info(f"Processing only {max_files} files out of {len(file_paths)}")
        file_paths = file_paths[:max_files]
    all_documents = []
    processed_files = []
    # Process each file
    progress_bar = st.progress(0)
    status_text = st.empty()
    for idx, file_path in enumerate(file_paths):
        status_text.text(
            f"Processing file {idx + 1}/{len(file_paths)}: {os.path.basename(file_path)}"
        )
        try:
            document = process_document_file(file_path, file_extension)
            if document:
                all_documents.append(document)
                processed_files.append(file_path)
            progress_bar.progress((idx + 1) / len(file_paths))
            # Add small delay to prevent overwhelming the system
            time.sleep(0.1)
        except Exception as e:
            st.warning(f"Failed to process {file_path}: {str(e)}")
            continue
    progress_bar.empty()
    status_text.empty()
    if not all_documents:
        st.error(f"No documents could be extracted from .{file_extension} files")
    return all_documents, processed_files


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
    # Different splitting strategy for different document types
    doc_chunks = []
    for doc in documents:
        if doc.metadata.get("file_type") in ["xlsx", "xls"]:
            # For Excel files, use larger chunks to preserve record integrity
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=2000,  # Larger chunks for tabular data
                chunk_overlap=100,  # Smaller overlap
            )
        else:
            # Standard splitting for other documents
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200,
            )
        chunks = text_splitter.split_documents([doc])
        doc_chunks.extend(chunks)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore


def create_chain(vectorstore):
    """Original create_chain function"""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=4096,
    )
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


def create_chain_with_custom_retriever(vectorstore, k=20):
    """Create a chain with a custom retriever that returns more results"""
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0,
        max_tokens=4096,
    )
    # Create a custom retriever that returns more documents
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},  # Return top k most similar chunks
    )
    memory = ConversationBufferMemory(
        llm=llm, output_key="answer", memory_key="chat_history", return_messages=True
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        chain_type="map_reduce",  # Better for handling multiple documents
        memory=memory,
        verbose=True,
        return_source_documents=True,  # This helps with debugging
    )
    return chain


def handle_comprehensive_query(query: str, vectorstore) -> str:
    """Handle queries that need to access all data, like 'list all names'"""
    # Keywords that indicate a comprehensive query
    comprehensive_keywords = [
        "list all",
        "all names",
        "all people",
        "everyone",
        "complete list",
        "full list",
        "entire",
        "total",
        "every",
    ]
    is_comprehensive = any(
        keyword in query.lower() for keyword in comprehensive_keywords
    )
    if is_comprehensive:
        # Get ALL documents from the vectorstore
        all_docs = []
        # Access the vectorstore's document store directly
        if hasattr(vectorstore, "docstore") and hasattr(
            vectorstore, "index_to_docstore_id"
        ):
            for i in range(len(vectorstore.index_to_docstore_id)):
                doc_id = vectorstore.index_to_docstore_id.get(i)
                if doc_id and doc_id in vectorstore.docstore._dict:
                    doc = vectorstore.docstore._dict[doc_id]
                    all_docs.append(doc.page_content)
        # Combine all document content
        combined_content = "\n".join(all_docs)
        # Create a specialized prompt for comprehensive queries
        llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            max_tokens=4096,
        )
        prompt = f"""
        Based on the following complete dataset, please answer this query: {query}
        
        Dataset:
        {combined_content[:15000]}  # Limit to avoid token limits
        
        Please provide a comprehensive answer based on ALL the data provided.
        If the data is too large to process completely, please mention this and provide
        what you can extract from the available portion.
        """
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error processing comprehensive query: {str(e)}"
    return None  # Not a comprehensive query


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
        st.sidebar.success(f"Loaded {len(urls)} previously processed items")
    return True


def get_content_statistics():
    """Get statistics about processed content"""
    conn = get_db_connection()
    cursor = conn.cursor()
    stats = {}
    # Count web pages
    cursor.execute("SELECT COUNT(*) FROM pages")
    stats["web_pages"] = cursor.fetchone()[0]
    # Count documents by type
    cursor.execute("SELECT file_type, COUNT(*) FROM documents GROUP BY file_type")
    doc_counts = cursor.fetchall()
    for file_type, count in doc_counts:
        stats[f"{file_type}_files"] = count
    return stats


def diagnose_excel_processing(file_path: str):
    """Diagnostic function to show how Excel data is being processed"""
    try:
        st.write("## 📊 Excel Processing Diagnostics")
        st.write(f"**Processing file: {os.path.basename(file_path)}**")

        # Read the Excel file directly
        try:
            df = pd.read_excel(file_path, engine="openpyxl")
        except Exception as e:
            st.error(f"Error reading Excel file: {str(e)}")
            return None, None

        st.write(f"**Original Excel File:**")
        st.write(f"- Total rows: {len(df)}")
        st.write(f"- Total columns: {len(df.columns)}")
        st.write(f"- Column names: {list(df.columns)}")

        # Show first few rows
        st.write("**First 5 rows of Excel data:**")
        st.dataframe(df.head())

        # Check for name-related columns
        name_columns = [
            col
            for col in df.columns
            if any(
                name_word in col.lower()
                for name_word in ["name", "first", "last", "client", "person"]
            )
        ]
        if name_columns:
            st.write(f"**Name-related columns found: {name_columns}**")
        else:
            st.write("**No obvious name columns found**")

        # Process the file and show the document
        st.write("**Processing file with process_excel_file function...**")
        doc = process_excel_file(file_path)

        if doc:
            st.write(f"**Processed Document:**")
            st.write(f"- Total content length: {len(doc.page_content)} characters")
            st.write(f"- Metadata: {doc.metadata}")

            # Show first part of processed content
            st.write("**First 1000 characters of processed content:**")
            st.text_area(
                "Processed Content Preview", doc.page_content[:1000], height=200
            )

            # Show how it gets chunked
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=2000,
                chunk_overlap=100,
            )
            chunks = text_splitter.split_documents([doc])
            st.write(f"**After Text Splitting:**")
            st.write(f"- Number of chunks created: {len(chunks)}")

            # Show chunk details
            for i, chunk in enumerate(chunks[:3]):  # Show first 3 chunks
                st.write(f"**Chunk {i + 1}:**")
                st.write(f"- Length: {len(chunk.page_content)} characters")
                st.text_area(
                    f"Chunk {i + 1} Content",
                    chunk.page_content[:500],
                    height=150,
                    key=f"chunk_{i}",
                )

            if len(chunks) > 3:
                st.write(f"... and {len(chunks) - 3} more chunks")
        else:
            st.error("Failed to process the Excel file into a document")

        return df, doc

    except Exception as e:
        st.error(f"Error in Excel diagnostics: {str(e)}")
        st.exception(e)  # This will show the full traceback
        return None, None


def diagnose_vectorstore_content():
    """Show what's actually in the vector store"""
    if "vectorstore" not in st.session_state:
        st.error("No vectorstore found!")
        return

    st.write("## 🔍 Vector Store Diagnostics")

    vs = st.session_state.vectorstore
    total_docs = 0
    excel_chunks = 0

    if hasattr(vs, "docstore") and hasattr(vs, "index_to_docstore_id"):
        total_docs = len(vs.index_to_docstore_id)
        st.write(f"**Total chunks in vector store: {total_docs}**")

        # Analyze chunks by type
        chunk_types = {}
        excel_content_lengths = []

        for i in range(len(vs.index_to_docstore_id)):
            doc_id = vs.index_to_docstore_id.get(i)
            if doc_id and doc_id in vs.docstore._dict:
                doc = vs.docstore._dict[doc_id]
                file_type = doc.metadata.get("file_type", "unknown")

                if file_type not in chunk_types:
                    chunk_types[file_type] = 0
                chunk_types[file_type] += 1

                if file_type in ["xlsx", "xls"]:
                    excel_chunks += 1
                    excel_content_lengths.append(len(doc.page_content))

        st.write("**Chunks by file type:**")
        for file_type, count in chunk_types.items():
            st.write(f"- {file_type}: {count} chunks")

        if excel_chunks > 0:
            st.write(f"**Excel chunk details:**")
            st.write(f"- Total Excel chunks: {excel_chunks}")
            st.write(
                f"- Average chunk size: {sum(excel_content_lengths) / len(excel_content_lengths):.0f} characters"
            )
            st.write(f"- Min chunk size: {min(excel_content_lengths)} characters")
            st.write(f"- Max chunk size: {max(excel_content_lengths)} characters")

            # Show some sample Excel chunks
            st.write("**Sample Excel chunks:**")
            count = 0
            for i in range(len(vs.index_to_docstore_id)):
                if count >= 3:  # Show first 3 Excel chunks
                    break
                doc_id = vs.index_to_docstore_id.get(i)
                if doc_id and doc_id in vs.docstore._dict:
                    doc = vs.docstore._dict[doc_id]
                    if doc.metadata.get("file_type") in ["xlsx", "xls"]:
                        count += 1
                        st.write(f"**Excel Chunk {count}:**")
                        st.text_area(
                            f"Content {count}",
                            doc.page_content[:800],
                            height=200,
                            key=f"excel_chunk_{count}",
                        )


def improved_get_all_names_from_vectorstore_with_diagnostics(vectorstore):
    """Enhanced name extraction with diagnostics - FIXED VERSION"""
    try:
        all_names = set()
        all_content = []
        processed_chunks = 0
        pattern_matches = {}

        st.write("## 🔎 Name Extraction Diagnostics")

        if hasattr(vectorstore, "docstore") and hasattr(
            vectorstore, "index_to_docstore_id"
        ):
            total_chunks = len(vectorstore.index_to_docstore_id)
            st.write(f"**Total chunks to process: {total_chunks}**")

            for i in range(total_chunks):
                doc_id = vectorstore.index_to_docstore_id.get(i)
                if doc_id and doc_id in vectorstore.docstore._dict:
                    doc = vectorstore.docstore._dict[doc_id]

                    # Only process Excel chunks
                    if doc.metadata.get("file_type") in ["xlsx", "xls"]:
                        content = doc.page_content
                        all_content.append(content)
                        processed_chunks += 1

                        # Define patterns with their names for tracking
                        patterns = {
                            "first_last_pattern": r"FIRST_NAME:\s*([^|]+)\s*\|\s*LAST_NAME:\s*([^|]+)",
                            "name_colon_pattern": r"(?:Name|FULL_NAME|Client):\s*([^|]+?)(?:\s*\||$)",
                            "record_name_pattern": r"Record \d+:.*?(?:Name|FIRST_NAME|LAST_NAME|Client):\s*([^|]+?)(?:\s*\||$)",
                            "title_name_pattern": r"(?:Mr\.|Mrs\.|Ms\.|Dr\.)\s+([A-Z][a-z]+\s+[A-Z][a-z]+)",
                            "capitalized_names": r"\b([A-Z][a-z]{2,}\s+[A-Z][a-z]{2,})\b",
                        }

                        for pattern_name, pattern in patterns.items():
                            if pattern_name not in pattern_matches:
                                pattern_matches[pattern_name] = 0

                            matches = re.findall(pattern, content, re.IGNORECASE)
                            pattern_matches[pattern_name] += len(matches)

                            # Handle different match types
                            for match in matches:
                                if pattern_name == "first_last_pattern":
                                    # This returns tuples (first, last)
                                    if isinstance(match, tuple) and len(match) == 2:
                                        first, last = match
                                        if first.strip() and last.strip():
                                            full_name = (
                                                f"{first.strip()} {last.strip()}"
                                            )
                                            if len(full_name) > 3 and not any(
                                                word in full_name.lower()
                                                for word in [
                                                    "name",
                                                    "first",
                                                    "last",
                                                    "column",
                                                    "row",
                                                ]
                                            ):
                                                all_names.add(full_name)
                                else:
                                    # These return single strings
                                    name = (
                                        match.strip()
                                        if isinstance(match, str)
                                        else str(match).strip()
                                    )
                                    if (
                                        name
                                        and len(name) > 3
                                        and not any(
                                            word in name.lower()
                                            for word in [
                                                "name",
                                                "first",
                                                "last",
                                                "column",
                                                "row",
                                                "record",
                                            ]
                                        )
                                    ):
                                        all_names.add(name)

            st.write(f"**Excel chunks processed: {processed_chunks}**")
            st.write(f"**Total unique names found: {len(all_names)}**")

            # Show pattern matching statistics
            st.write("**Pattern matching results:**")
            for pattern_name, count in pattern_matches.items():
                st.write(f"- {pattern_name}: {count} matches")

            # Show sample content for debugging
            if all_content:
                st.write("**Sample content being processed:**")
                sample_content = all_content[0][:1500] if all_content else "No content"
                st.text_area(
                    "Sample Content",
                    sample_content,
                    height=300,
                    key="sample_content_debug",
                )

                # Test patterns on sample content
                st.write("**Testing patterns on sample content:**")
                test_patterns = {
                    "FIRST_NAME|LAST_NAME": r"FIRST_NAME:\s*([^|]+)\s*\|\s*LAST_NAME:\s*([^|]+)",
                    "Name: Something": r"(?:Name|FULL_NAME):\s*([^|]+?)(?:\s*\||$)",
                    "Record with names": r"Record \d+:.*?(?:Name|FIRST_NAME|LAST_NAME):\s*([^|]+?)(?:\s*\||$)",
                }

                for pattern_name, pattern in test_patterns.items():
                    matches = re.findall(pattern, sample_content, re.IGNORECASE)
                    st.write(f"**{pattern_name}**: Found {len(matches)} matches")
                    if matches:
                        st.write(f"Examples: {matches[:5]}")  # Show first 5 matches

        return sorted(list(all_names))

    except Exception as e:
        st.error(f"Error in name extraction: {str(e)}")
        st.exception(e)  # Show full traceback
        return []


def compare_counting_methods(vectorstore):
    """Compare different ways of counting names"""
    st.write("## 🔢 Name Counting Comparison")

    # Method 1: Direct pattern matching on all content (like comprehensive query)
    all_content = []
    if hasattr(vectorstore, "docstore") and hasattr(
        vectorstore, "index_to_docstore_id"
    ):
        for i in range(len(vectorstore.index_to_docstore_id)):
            doc_id = vectorstore.index_to_docstore_id.get(i)
            if doc_id and doc_id in vectorstore.docstore._dict:
                doc = vectorstore.docstore._dict[doc_id]
                if doc.metadata.get("file_type") in ["xlsx", "xls"]:
                    all_content.append(doc.page_content)

    combined_content = "\n".join(all_content)

    # Count using the comprehensive query approach
    import re

    pattern = r"FIRST_NAME:\s*([^|]+?)\s*\|\s*LAST_NAME:\s*([^|]+?)\s*\|"
    comprehensive_matches = re.findall(pattern, combined_content, re.IGNORECASE)

    comprehensive_names = set()
    for first, last in comprehensive_matches:
        first, last = first.strip(), last.strip()
        if first and last:
            comprehensive_names.add(f"{first} {last}")

    # Method 2: Using your improved function
    improved_names = set(get_all_names_from_vectorstore_improved(vectorstore))

    # Method 3: Raw pattern matching without filtering
    raw_names = set()
    for first, last in comprehensive_matches:
        first, last = first.strip(), last.strip()
        if first and last:
            raw_names.add(f"{first} {last}")

    st.write(
        f"**Method 1 - Comprehensive Query Style**: {len(comprehensive_names)} names"
    )
    st.write(f"**Method 2 - Improved Function**: {len(improved_names)} names")
    st.write(f"**Method 3 - Raw Pattern Matches**: {len(raw_names)} names")

    # Show the difference
    missing_from_improved = comprehensive_names - improved_names
    if missing_from_improved:
        st.write(
            f"**Names in comprehensive but missing from improved ({len(missing_from_improved)}):**"
        )
        for name in sorted(list(missing_from_improved))[:10]:
            st.write(f"- {name}")
        if len(missing_from_improved) > 10:
            st.write(f"... and {len(missing_from_improved) - 10} more")

    return len(comprehensive_names), len(improved_names), len(raw_names)


def on_shutdown():
    close_db_connection()


# Register shutdown handler
import atexit

atexit.register(on_shutdown)


def handle_specific_person_query(query: str, vectorstore) -> str:
    """Handle queries about specific people - GUARANTEED WORKING VERSION"""
    try:
        # Extract the name from the query first
        import re

        # Simple name extraction
        words = query.split()
        potential_names = []

        # Look for pairs of capitalized words
        for i in range(len(words) - 1):
            if (
                words[i].replace(",", "").replace(".", "").isalpha()
                and words[i + 1].replace(",", "").replace(".", "").isalpha()
                and len(words[i]) > 1
                and len(words[i + 1]) > 1
            ):
                first = words[i].capitalize()
                last = words[i + 1].capitalize()

                # Skip obvious non-names
                if first.lower() not in [
                    "give",
                    "show",
                    "tell",
                    "what",
                    "have",
                    "all",
                    "the",
                    "information",
                ] and last.lower() not in [
                    "give",
                    "show",
                    "tell",
                    "what",
                    "have",
                    "all",
                    "the",
                    "information",
                ]:
                    potential_names.append(f"{first} {last}")

        # Also try single words that might be names
        for word in words:
            clean_word = word.replace(",", "").replace(".", "").capitalize()
            if (
                len(clean_word) > 2
                and clean_word.isalpha()
                and clean_word.lower()
                not in [
                    "give",
                    "show",
                    "tell",
                    "what",
                    "have",
                    "all",
                    "the",
                    "information",
                    "you",
                    "me",
                    "on",
                    "about",
                ]
            ):
                potential_names.append(clean_word)

        st.write(f"**Debug: Looking for these names: {potential_names}**")

        if not potential_names:
            return "I couldn't identify any names in your query. Please try asking like 'What is John Smith's phone number?'"

        # Get ALL the names from the database using our working function
        all_names = get_all_names_from_vectorstore_improved(vectorstore)

        st.write(f"**Debug: Total names in database: {len(all_names)}**")

        # Find exact matches (case insensitive)
        matched_names = []
        for search_name in potential_names:
            st.write(f"**Debug: Searching for '{search_name}'**")

        for db_name in all_names:
            # Exact match
            if search_name.lower() == db_name.lower():
                matched_names.append(db_name)
                st.write(f"**Debug: EXACT MATCH FOUND: {db_name}**")
            # First name match
            elif search_name.lower() == db_name.split()[0].lower():
                matched_names.append(db_name)
                st.write(f"**Debug: FIRST NAME MATCH: {db_name}**")
            # Last name match (check if search_name matches any part after first name)
            elif len(db_name.split()) > 1 and search_name.lower() in [
                part.lower() for part in db_name.split()[1:]
            ]:
                matched_names.append(db_name)
                st.write(f"**Debug: LAST NAME MATCH: {db_name}**")
            # Partial match (search_name is part of db_name)
            elif search_name.lower() in db_name.lower():
                matched_names.append(db_name)
                st.write(f"**Debug: PARTIAL MATCH FOUND: {db_name}**")
        # Remove duplicates
        matched_names = list(set(matched_names))

        if not matched_names:
            # Show similar names for debugging
            similar_names = [
                name
                for name in all_names
                if any(part.lower() in name.lower() for part in potential_names)
            ]
            return f"I couldn't find exact matches for: {potential_names}. Similar names in database: {similar_names[:10]}"

        st.write(f"**Debug: Found these matching names: {matched_names}**")

        # Now get the full records for these names
        all_content = []
        if hasattr(vectorstore, "docstore") and hasattr(
            vectorstore, "index_to_docstore_id"
        ):
            for i in range(len(vectorstore.index_to_docstore_id)):
                doc_id = vectorstore.index_to_docstore_id.get(i)
                if doc_id and doc_id in vectorstore.docstore._dict:
                    doc = vectorstore.docstore._dict[doc_id]
                    if doc.metadata.get("file_type") in ["xlsx", "xls"]:
                        all_content.append(doc.page_content)

        combined_content = "\n".join(all_content)

        # Find the full records for the matched names
        found_records = []
        for name in matched_names:
            # Split name into first and last
            name_parts = name.split()
            if len(name_parts) >= 2:
                first_name = name_parts[0]
                last_name = " ".join(name_parts[1:])  # Handle names like "Van Der Berg"

                # Search for the exact pattern in the content
                pattern = f"FIRST_NAME: {first_name} | LAST_NAME: {last_name}"

                # Find the record containing this pattern
                lines = combined_content.split("\n")
                for i, line in enumerate(lines):
                    if pattern in line:
                        # Get the full record (this line plus several after it)
                        record_lines = []
                        for j in range(i, min(i + 10, len(lines))):
                            record_lines.append(lines[j])
                            # Stop when we hit the next record or end
                            if j > i and (
                                "Record " in lines[j] and "FIRST_NAME:" in lines[j]
                            ):
                                break

                        found_records.append("\n".join(record_lines))
                        break

        if found_records:
            st.write(f"**Debug: Found {len(found_records)} complete records**")

            # Use LLM to format the response
            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                temperature=0,
                max_tokens=4096,
            )

            records_text = "\n\n".join(found_records)

            prompt = f"""
            Question: {query}
            
            Here are the complete database records for the person(s) you asked about:
            
            {records_text}
            
            Please provide a clear, organized response with all the information for the person requested.
            """

            response = llm.invoke(prompt)
            return response.content
        else:
            return f"Found matching names {matched_names} but couldn't retrieve their full records."

    except Exception as e:
        st.write(f"**Debug: Error: {str(e)}**")
        return f"Error: {str(e)}"


# Streamlit app configuration
st.set_page_config(page_title="Chat with Website", page_icon="", layout="wide")
st.title("CLAUDIA  🦙(LLAMA 3.3)")

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
st.subheader("Load Content")

# Sitemap input
sitemap_col, sitemap_options_col = st.columns([3, 1])
with sitemap_col:
    sitemap_url = st.text_input(
        "Enter website sitemap URL:",
        placeholder="e.g., https://example.com/sitemap.xml",
    )
with sitemap_options_col:
    max_urls = st.slider(
        "Max URLs",
        min_value=1,
        max_value=100,
        value=50,
        help="Limit the number of URLs to process",
    )

# Local HTML directory input
html_col, html_options_col = st.columns([3, 1])
with html_col:
    html_dir = st.text_input(
        "Enter path to HTML files directory:",
        placeholder="e.g., C:/Documents/html_files or /home/user/html_files",
    )
with html_options_col:
    max_html_files = st.slider(
        "Max HTML files",
        min_value=1,
        max_value=100,
        value=50,
        help="Limit the number of HTML files to process",
    )

# Local DOCX directory input
docx_col, docx_options_col = st.columns([3, 1])
with docx_col:
    docx_dir = st.text_input(
        "Enter path to DOCX files directory:",
        placeholder="e.g., C:/Documents/docx_files or /home/user/docx_files",
    )
with docx_options_col:
    max_docx_files = st.slider(
        "Max DOCX files",
        min_value=1,
        max_value=100,
        value=50,
        help="Limit the number of DOCX files to process",
    )

# Local PDF directory input
pdf_col, pdf_options_col = st.columns([3, 1])
with pdf_col:
    pdf_dir = st.text_input(
        "Enter path to PDF files directory:",
        placeholder="e.g., C:/Documents/pdf_files or /home/user/pdf_files",
    )
with pdf_options_col:
    max_pdf_files = st.slider(
        "Max PDF files",
        min_value=1,
        max_value=100,
        value=50,
        help="Limit the number of PDF files to process",
    )

# Local PowerPoint directory input
pptx_col, pptx_options_col = st.columns([3, 1])
with pptx_col:
    pptx_dir = st.text_input(
        "Enter path to PowerPoint files directory:",
        placeholder="e.g., C:/Documents/pptx_files or /home/user/pptx_files",
    )
with pptx_options_col:
    max_pptx_files = st.slider(
        "Max PPTX files",
        min_value=1,
        max_value=100,
        value=50,
        help="Limit the number of PowerPoint files to process",
    )

# Local Excel directory input
xlsx_col, xlsx_options_col = st.columns([3, 1])
with xlsx_col:
    xlsx_dir = st.text_input(
        "Enter path to Excel files directory:",
        placeholder="e.g., C:/Documents/xlsx_files or /home/user/xlsx_files",
    )
with xlsx_options_col:
    max_xlsx_files = st.slider(
        "Max Excel files",
        min_value=1,
        max_value=100,
        value=50,
        help="Limit the number of Excel files to process",
    )

# Process buttons
col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
    process_sitemap_button = st.button(
        "Process Sitemap", type="primary", use_container_width=True
    )
with col2:
    process_html_button = st.button(
        "Process HTML Files", type="primary", use_container_width=True
    )
with col3:
    process_docx_button = st.button(
        "Process DOCX Files", type="primary", use_container_width=True
    )
with col4:
    process_pdf_button = st.button(
        "Process PDF Files", type="primary", use_container_width=True
    )
with col5:
    process_pptx_button = st.button(
        "Process PowerPoint Files", type="primary", use_container_width=True
    )
with col6:
    process_xlsx_button = st.button(
        "Process Excel Files", type="primary", use_container_width=True
    )

# Process sitemap
if process_sitemap_button and sitemap_url:
    try:
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

# Process HTML files
if process_html_button and html_dir:
    try:
        documents, processed_files = process_local_directory(
            html_dir, "html", max_html_files
        )
        if documents:
            st.success(f"Successfully processed {len(documents)} HTML files")
            # Create or update vectorstore
            if "vectorstore" not in st.session_state:
                st.session_state.vectorstore = setup_vectorstore(documents)
            else:
                st.session_state.vectorstore.add_documents(documents)
            # Save data to persistent storage
            save_vectorstore(st.session_state.vectorstore)
            save_local_documents(documents, processed_files, "html")
            st.session_state.processed_urls.extend(processed_files)
            # Show processed files
            with st.expander("Processed HTML Files", expanded=True):
                for file_path in processed_files:
                    st.write(os.path.basename(file_path))
    except Exception as e:
        st.error(f"Error processing HTML files: {str(e)}")

# Process DOCX files
if process_docx_button and docx_dir:
    try:
        documents, processed_files = process_local_directory(
            docx_dir, "docx", max_docx_files
        )
        if documents:
            st.success(f"Successfully processed {len(documents)} DOCX files")
            # Create or update vectorstore
            if "vectorstore" not in st.session_state:
                st.session_state.vectorstore = setup_vectorstore(documents)
            else:
                st.session_state.vectorstore.add_documents(documents)
            # Save data to persistent storage
            save_vectorstore(st.session_state.vectorstore)
            save_local_documents(documents, processed_files, "docx")
            st.session_state.processed_urls.extend(processed_files)
            # Show processed files
            with st.expander("Processed DOCX Files", expanded=True):
                for file_path in processed_files:
                    st.write(os.path.basename(file_path))
    except Exception as e:
        st.error(f"Error processing DOCX files: {str(e)}")

# Process PDF files
if process_pdf_button and pdf_dir:
    try:
        documents, processed_files = process_local_directory(
            pdf_dir, "pdf", max_pdf_files
        )
        if documents:
            st.success(f"Successfully processed {len(documents)} PDF files")
            # Create or update vectorstore
            if "vectorstore" not in st.session_state:
                st.session_state.vectorstore = setup_vectorstore(documents)
            else:
                st.session_state.vectorstore.add_documents(documents)
            # Save data to persistent storage
            save_vectorstore(st.session_state.vectorstore)
            save_local_documents(documents, processed_files, "pdf")
            st.session_state.processed_urls.extend(processed_files)
            # Show processed files
            with st.expander("Processed PDF Files", expanded=True):
                for file_path in processed_files:
                    st.write(os.path.basename(file_path))
    except Exception as e:
        st.error(f"Error processing PDF files: {str(e)}")

# Process PowerPoint files
if process_pptx_button and pptx_dir:
    try:
        documents, processed_files = process_local_directory(
            pptx_dir, "pptx", max_pptx_files
        )
        if documents:
            st.success(f"Successfully processed {len(documents)} PowerPoint files")
            # Create or update vectorstore
            if "vectorstore" not in st.session_state:
                st.session_state.vectorstore = setup_vectorstore(documents)
            else:
                st.session_state.vectorstore.add_documents(documents)
            # Save data to persistent storage
            save_vectorstore(st.session_state.vectorstore)
            save_local_documents(documents, processed_files, "pptx")
            st.session_state.processed_urls.extend(processed_files)
            # Show processed files
            with st.expander("Processed PowerPoint Files", expanded=True):
                for file_path in processed_files:
                    st.write(os.path.basename(file_path))
    except Exception as e:
        st.error(f"Error processing PowerPoint files: {str(e)}")

# Process Excel files
if process_xlsx_button and xlsx_dir:
    try:
        documents, processed_files = process_local_directory(
            xlsx_dir, "xlsx", max_xlsx_files
        )
        if documents:
            st.success(f"Successfully processed {len(documents)} Excel files")
            # Create or update vectorstore
            if "vectorstore" not in st.session_state:
                st.session_state.vectorstore = setup_vectorstore(documents)
            else:
                st.session_state.vectorstore.add_documents(documents)
            # Save data to persistent storage
            save_vectorstore(st.session_state.vectorstore)
            save_local_documents(documents, processed_files, "xlsx")
            st.session_state.processed_urls.extend(processed_files)
            # Show processed files
            with st.expander("Processed Excel Files", expanded=True):
                for file_path in processed_files:
                    st.write(os.path.basename(file_path))
    except Exception as e:
        st.error(f"Error processing Excel files: {str(e)}")

# Add this right after the Excel processing section
if "vectorstore" in st.session_state:
    st.divider()
    st.subheader("🔧 Diagnostics")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("Diagnose Excel Processing"):
            if xlsx_dir and os.path.exists(xlsx_dir):
                files = glob.glob(os.path.join(xlsx_dir, "*.xlsx"))
                if files:
                    st.write(
                        f"Found Excel files: {[os.path.basename(f) for f in files]}"
                    )
                    diagnose_excel_processing(files[0])
                else:
                    st.error(f"No .xlsx files found in {xlsx_dir}")
            else:
                st.error("Please enter a valid Excel directory path first")

    with col2:
        if st.button("Diagnose Vector Store"):
            diagnose_vectorstore_content()

    with col3:
        if st.button("Diagnose Name Extraction"):
            names = improved_get_all_names_from_vectorstore_with_diagnostics(
                st.session_state.vectorstore
            )
            if names:
                st.write(f"**All {len(names)} unique names found:**")
                # Display in columns for better readability
                cols = st.columns(3)
                for i, name in enumerate(names):
                    with cols[i % 3]:
                        st.write(f"{i + 1}. {name}")
            else:
                st.warning("No names found or error occurred")
    with st.columns(4)[3]:  # Add a 4th column
        if st.button("Test Improved Names"):
            names = get_all_names_from_vectorstore_improved(
                st.session_state.vectorstore
            )
            st.write(f"**Improved extraction found {len(names)} clean names:**")
            if names:
                # Display in columns for better readability
                cols = st.columns(3)
            for i, name in enumerate(names):
                with cols[i % 3]:
                    st.write(f"{i + 1}. {name}")
        else:
            st.warning("No names found")

    # Add this as a new column/button
    with st.columns(6)[5]:  # Or wherever you have space
        if st.button("Compare Counting Methods"):
            comp_count, imp_count, raw_count = compare_counting_methods(
                st.session_state.vectorstore
            )
            st.write(
                f"**Results**: Comprehensive: {comp_count}, Improved: {imp_count}, Raw: {raw_count}"
            )


# Show loaded documents/URLs in sidebar
with st.sidebar:
    st.subheader("Processed Content")
    if st.session_state.processed_urls:
        # Get content statistics
        stats = get_content_statistics()
        st.write(f"**Total items:** {len(st.session_state.processed_urls)}")
        # Show breakdown by type
        if stats.get("web_pages", 0) > 0:
            st.write(f"📄 Web pages: {stats['web_pages']}")
        if stats.get("html_files", 0) > 0:
            st.write(f"📝 HTML files: {stats['html_files']}")
        if stats.get("docx_files", 0) > 0:
            st.write(f"📘 DOCX files: {stats['docx_files']}")
        if stats.get("pdf_files", 0) > 0:
            st.write(f"📕 PDF files: {stats['pdf_files']}")
        if stats.get("pptx_files", 0) > 0:
            st.write(f"📊 PowerPoint files: {stats['pptx_files']}")
        if stats.get("xlsx_files", 0) > 0:
            st.write(f"📈 Excel files: {stats['xlsx_files']}")
        with st.expander("View all items", expanded=False):
            for item in st.session_state.processed_urls:
                if item.startswith("http"):
                    st.write(f"🌐 {item}")
                else:
                    st.write(f"📁 {os.path.basename(item)}")

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
                    cursor.execute("DELETE FROM pages")
                    cursor.execute("DELETE FROM websites")
                    cursor.execute("DELETE FROM documents")
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
        st.info("No content processed yet. Enter paths or URLs above to begin.")

# Add this after the sidebar section and before the chat interface
if "vectorstore" in st.session_state:
    st.divider()
    export_container = st.container()
    with export_container:
        st.subheader("Data Export")
        # Create two columns for export buttons
        col1, col2 = st.columns(2)
        # Export Vector Database
        with col1:
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
                            if hasattr(vs, "index_to_docstore_id") and hasattr(
                                vs, "index"
                            ):
                                for i in range(len(vs.index_to_docstore_id)):
                                    doc_id = vs.index_to_docstore_id.get(i)
                                    if doc_id and doc_id in vs.docstore._dict:
                                        # Get the document
                                        doc = vs.docstore._dict[doc_id]
                                        # Get the vector
                                        vector = vs.index.reconstruct(i)
                                        vectors.append(vector)
                                        ids.append(doc_id)
                                        # Enhanced metadata for different document types
                                        meta = {
                                            "source": doc.metadata.get("source", ""),
                                            "content_preview": doc.page_content[:100],
                                        }
                                        if doc.metadata.get("type") == "web":
                                            meta["title"] = doc.metadata.get(
                                                "title", ""
                                            )
                                            meta["url"] = doc.metadata.get("url", "")
                                        else:
                                            meta["file_name"] = doc.metadata.get(
                                                "file_name", ""
                                            )
                                            meta["file_type"] = doc.metadata.get(
                                                "file_type", ""
                                            )
                                        metadata.append(meta)
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
                            return (
                                True,
                                f"Exported {len(vectors)} vectors to {filename}",
                            )
                        except Exception as e:
                            return False, f"Error exporting vectors: {str(e)}"

                    # Export the vectors to a file in the data directory
                    export_path = os.path.join(DATA_DIR, "faiss_vectors.csv")
                    success, message = export_faiss_vectors(export_path)
                    if success:
                        st.success(message)
                        # Create a download link for the exported file
                        with open(export_path, "rb") as file:
                            st.download_button(
                                label="Download Vector CSV",
                                data=file,
                                file_name="faiss_vectors.csv",
                                mime="text/csv",
                            )
                    else:
                        st.error(message)

        # Export SQLite Database
        with col2:
            if st.button("Export SQLite Database", type="primary"):
                with st.spinner("Preparing SQLite database for export..."):
                    try:
                        # Create a copy of the database for export
                        import shutil

                        export_db_path = os.path.join(
                            DATA_DIR, "website_data_export.db"
                        )
                        # Close any existing connections before copying
                        close_db_connection()
                        # Copy the database file
                        shutil.copy2(DB_PATH, export_db_path)
                        st.success("SQLite database prepared for download!")
                        # Create download button for the SQLite file
                        with open(export_db_path, "rb") as file:
                            st.download_button(
                                label="Download SQLite Database",
                                data=file,
                                file_name="website_data.db",
                                mime="application/x-sqlite3",
                            )
                    except Exception as e:
                        st.error(f"Error exporting SQLite database: {str(e)}")


# Initialize conversation chain if vectorstore exists
if "vectorstore" in st.session_state and "conversation_chain" not in st.session_state:
    st.session_state.conversation_chain = create_chain_with_custom_retriever(
        st.session_state.vectorstore,
        k=50,  # Retrieve top 50 chunks
    )

# Chat interface
if "vectorstore" in st.session_state:
    st.divider()
    st.subheader("Ask Claudia - Your AI Assistant")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
user_input = st.chat_input("Ask about the website content...")

if user_input and user_input.strip():
    # Add user message to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write("Lowly Human")
        st.markdown(user_input)

    # Generate assistant response
    with st.chat_message("assistant"):
        st.write("Claudia")
        with st.spinner("Thinking..."):
            user_query_lower = user_input.lower()

            # Enhanced query routing based on question type
            if any(
                keyword in user_query_lower
                for keyword in [
                    "list all names",
                    "list the names",
                    "all names",
                    "show all names",
                ]
            ):
                # Handle "list all names" requests
                names = get_all_names_from_vectorstore_improved(
                    st.session_state.vectorstore
                )
                if names:
                    assistant_response = (
                        f"Here are all {len(names)} names in the database:\n\n"
                    )
                    for i, name in enumerate(names, 1):
                        assistant_response += f"{i}. {name}\n"
                else:
                    assistant_response = "I couldn't find any names in the database."
                st.markdown(assistant_response)

            elif any(
                keyword in user_query_lower
                for keyword in ["how many names", "count names", "number of names"]
            ):
                # Handle "how many names" requests
                names = get_all_names_from_vectorstore_improved(
                    st.session_state.vectorstore
                )
                assistant_response = f"There are {len(names)} names in the database."
                st.markdown(assistant_response)

            elif any(
                word
                for word in user_input.split()
                if len(word.strip(".,!?")) > 2
                and word.strip(".,!?").replace("'", "").isalpha()
                and word.strip(".,!?").lower()
                not in [
                    "give",
                    "me",
                    "all",
                    "the",
                    "information",
                    "you",
                    "have",
                    "on",
                    "about",
                    "show",
                    "tell",
                    "what",
                    "is",
                    "does",
                    "and",
                    "or",
                    "but",
                    "for",
                    "with",
                    "from",
                    "to",
                ]
            ):
                # Handle specific person lookup queries
                assistant_response = handle_specific_person_query(
                    user_input, st.session_state.vectorstore
                )
                st.markdown(assistant_response)

            else:
                # First try comprehensive query for general questions
                comprehensive_response = handle_comprehensive_query(
                    user_input, st.session_state.vectorstore
                )

                if comprehensive_response:
                    assistant_response = comprehensive_response
                    st.markdown(assistant_response)
                else:
                    # Use the regular conversational chain for other questions
                    response = st.session_state.conversation_chain(
                        {"question": user_input}
                    )
                    assistant_response = response["answer"]
                    st.markdown(assistant_response)

                    # Show source documents for debugging
                    if "source_documents" in response:
                        with st.expander("Source Documents Used", expanded=False):
                            for i, doc in enumerate(response["source_documents"]):
                                st.write(f"**Source {i + 1}:**")
                                st.write(
                                    doc.page_content[:500] + "..."
                                    if len(doc.page_content) > 500
                                    else doc.page_content
                                )

            # Add assistant response to chat history
            st.session_state.chat_history.append(
                {"role": "assistant", "content": assistant_response}
            )

            # Save chat history to database
            save_chat_history(st.session_state.chat_history)
else:
    st.info("Please process content from any of the sources above to start chatting")
