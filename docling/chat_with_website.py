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

# load the environment variables
load_dotenv()
working_dir = os.path.dirname(os.path.abspath(__file__))


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

        # Try to get the full text as markdown
        try:
            full_text = result.document.export_to_markdown()
            if full_text and full_text.strip():
                return Document(
                    page_content=full_text,
                    metadata={
                        "source": url,
                        "filename": url_filename,
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


st.set_page_config(page_title="Chat with Website", page_icon="", layout="centered")
st.title("Chat with Website - LLAMA 3.3")

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = []

# Document loading options
st.subheader("Load Website Content")

sitemap_url = st.text_input(
    "Enter website sitemap URL:", placeholder="e.g., https://example.com/sitemap.xml"
)

col1, col2 = st.columns([2, 1])
with col1:
    max_urls = st.slider(
        "Maximum URLs to process (0 for all)",
        min_value=0,
        max_value=100,
        value=50,
        help="Limit the number of URLs to process to avoid long processing times",
    )
with col2:
    process_button = st.button("Process Sitemap", type="primary")

if process_button and sitemap_url:
    try:
        if max_urls == 0:
            max_urls = None  # Process all URLs

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

            st.session_state.processed_urls.extend(processed_urls)

            # Show processed URLs
            with st.expander("Processed URLs", expanded=True):
                for url in processed_urls:
                    st.write(url)
    except Exception as e:
        st.error(f"Error processing sitemap: {str(e)}")

# Show loaded documents/URLs
if st.session_state.processed_urls:
    with st.sidebar:
        st.subheader("Processed Content")
        for url in st.session_state.processed_urls:
            st.write(f"{url}")
        if st.button("Clear All Content"):
            for key in [
                "vectorstore",
                "conversation_chain",
                "processed_urls",
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
    st.subheader("Chat with your website content")

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_input = st.chat_input("Ask about the website content...")
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
    st.info("Please process a website sitemap to start chatting")
