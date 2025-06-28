import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from datetime import datetime
import glob

# Set up the app
st.set_page_config(
    page_title="Vector Database Exporter",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Vector Database Exporter")
st.markdown("Export your FAISS vector database to CSV or JSON format")

def find_faiss_index_paths():
    """Find all possible FAISS index locations"""
    possible_paths = []
    
    # Primary path - your specific location
    primary_path = "/home/steve/ai_projects/ai_office_assistant/docling/data/faiss_index"
    
    # Common locations to check
    search_locations = [
        primary_path,  # Your path first
        "data/faiss_index",
        "./data/faiss_index", 
        "../data/faiss_index",
        "faiss_index",
        "./faiss_index"
    ]
    
    # Also search in current directory and parent directories
    current_dir = os.getcwd()
    search_locations.extend([
        os.path.join(current_dir, "data", "faiss_index"),
        os.path.join(os.path.dirname(current_dir), "data", "faiss_index"),
        os.path.join(current_dir, "faiss_index")
    ])
    
    # Find existing paths
    for path in search_locations:
        if os.path.exists(path):
            # Check if it contains FAISS files
            faiss_files = glob.glob(os.path.join(path, "*.faiss")) + glob.glob(os.path.join(path, "*.pkl"))
            if faiss_files:
                possible_paths.append(os.path.abspath(path))
    
    # Remove duplicates
    return list(set(possible_paths))

def load_vectorstore(index_path):
    """Load FAISS vectorstore from specified path"""
    try:
        embeddings = HuggingFaceEmbeddings()
        vectorstore = FAISS.load_local(
            index_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        return vectorstore, "Success"
    except Exception as e:
        return None, f"Error loading vectorstore: {str(e)}"

def save_csv_file(vectorstore, save_path):
    """Export FAISS vectors to CSV and save to specified path"""
    try:
        vectors = []
        metadata_list = []
        
        # Extract vectors and metadata
        if hasattr(vectorstore, "index_to_docstore_id") and hasattr(vectorstore, "index"):
            for i in range(len(vectorstore.index_to_docstore_id)):
                doc_id = vectorstore.index_to_docstore_id.get(i)
                if doc_id and doc_id in vectorstore.docstore._dict:
                    # Get the document
                    doc = vectorstore.docstore._dict[doc_id]
                    
                    # Get the vector
                    vector = vectorstore.index.reconstruct(i)
                    vectors.append(vector)
                    
                    # Prepare metadata
                    metadata = {
                        "doc_id": doc_id,
                        "source": doc.metadata.get("source", ""),
                        "file_name": doc.metadata.get("file_name", ""),
                        "file_type": doc.metadata.get("file_type", ""),
                        "content_preview": doc.page_content[:200].replace('\n', ' ').replace('\r', ' '),
                        "content_length": len(doc.page_content),
                        "url": doc.metadata.get("url", ""),
                        "title": doc.metadata.get("title", ""),
                        "type": doc.metadata.get("type", ""),
                        "extraction_method": doc.metadata.get("extraction_method", "")
                    }
                    metadata_list.append(metadata)
        
        if not vectors:
            return False, "No vectors found in the database"
        
        # Create DataFrames
        metadata_df = pd.DataFrame(metadata_list)
        vectors_df = pd.DataFrame(np.array(vectors), columns=[f"vector_{i}" for i in range(len(vectors[0]))])
        
        # Combine metadata and vectors
        result_df = pd.concat([metadata_df, vectors_df], axis=1)
        
        # Save to file
        result_df.to_csv(save_path, index=False)
        
        return True, f"Successfully exported {len(vectors)} vectors to {save_path}"
        
    except Exception as e:
        return False, f"Error exporting to CSV: {str(e)}"

def save_json_file(vectorstore, save_path):
    """Export FAISS vectors to JSON and save to specified path"""
    try:
        export_data = {
            "export_info": {
                "timestamp": datetime.now().isoformat(),
                "total_vectors": 0,
                "vector_dimension": 0
            },
            "documents": []
        }
        
        # Extract vectors and metadata
        if hasattr(vectorstore, "index_to_docstore_id") and hasattr(vectorstore, "index"):
            for i in range(len(vectorstore.index_to_docstore_id)):
                doc_id = vectorstore.index_to_docstore_id.get(i)
                if doc_id and doc_id in vectorstore.docstore._dict:
                    # Get the document
                    doc = vectorstore.docstore._dict[doc_id]
                    
                    # Get the vector
                    vector = vectorstore.index.reconstruct(i)
                    
                    # Prepare document data
                    doc_data = {
                        "doc_id": doc_id,
                        "metadata": {
                            "source": doc.metadata.get("source", ""),
                            "file_name": doc.metadata.get("file_name", ""),
                            "file_type": doc.metadata.get("file_type", ""),
                            "content_length": len(doc.page_content),
                            "url": doc.metadata.get("url", ""),
                            "title": doc.metadata.get("title", ""),
                            "type": doc.metadata.get("type", ""),
                            "extraction_method": doc.metadata.get("extraction_method", "")
                        },
                        "content": doc.page_content,
                        "content_preview": doc.page_content[:200],
                        "vector": vector.tolist()  # Convert numpy array to list for JSON serialization
                    }
                    export_data["documents"].append(doc_data)
        
        if not export_data["documents"]:
            return False, "No documents found in the database"
        
        # Update export info
        export_data["export_info"]["total_vectors"] = len(export_data["documents"])
        if export_data["documents"]:
            export_data["export_info"]["vector_dimension"] = len(export_data["documents"][0]["vector"])
        
        # Save to file
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        return True, f"Successfully exported {len(export_data['documents'])} documents to {save_path}"
        
    except Exception as e:
        return False, f"Error exporting to JSON: {str(e)}"

# Configuration Section
st.subheader("⚙️ Configuration")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📂 FAISS Index Path")
    
    # Auto-detect possible paths
    possible_paths = find_faiss_index_paths()
    
    # Your default path
    default_path = "/home/steve/ai_projects/ai_office_assistant/docling/data/faiss_index"
    
    if possible_paths:
        st.success(f"✅ Found {len(possible_paths)} FAISS index location(s)")
        
        # If the default path is in the found paths, make it the default selection
        default_index = 0
        if default_path in possible_paths:
            default_index = possible_paths.index(default_path)
        
        selected_faiss_path = st.selectbox(
            "Select FAISS index location:",
            possible_paths,
            index=default_index,
            format_func=lambda x: f"{'🏠 ' if x == default_path else ''}.../{'/'.join(x.split('/')[-2:])}"
        )
        
        # Show full path
        st.info(f"📁 Full path: `{selected_faiss_path}`")
        
    else:
        st.warning("⚠️ No FAISS index found automatically")
        st.info(f"🏠 Using default path")
        selected_faiss_path = st.text_input(
            "FAISS index path:",
            value=default_path,
            help="Default path to your FAISS index directory"
        )
    
    # Manual path override option
    with st.expander("🔧 Override with custom path"):
        custom_path = st.text_input(
            "Custom FAISS index path:",
            placeholder="Enter alternative path if needed"
        )
        if custom_path and st.button("Use Custom Path"):
            selected_faiss_path = custom_path
            st.success(f"✅ Switched to custom path: {custom_path}")

with col2:
    st.markdown("#### 💾 Export Directory")
    
    # Default export directory (same base directory as FAISS index)
    default_export_dir = "/home/steve/ai_projects/ai_office_assistant/docling/exports"
    
    export_directory = st.text_input(
        "Choose where to save exported files:",
        value=default_export_dir,
        placeholder="e.g., /home/steve/exports",
        help="Directory where CSV and JSON files will be saved"
    )
    
    # Create directory if it doesn't exist
    if export_directory:
        try:
            os.makedirs(export_directory, exist_ok=True)
            if os.path.exists(export_directory):
                st.success(f"✅ Export directory ready")
                st.info(f"📁 Path: `{export_directory}`")
            else:
                st.error(f"❌ Cannot access directory: {export_directory}")
        except Exception as e:
            st.error(f"❌ Error creating directory: {str(e)}")

# Display current working directory for reference
st.markdown("---")
st.info(f"**Current working directory:** `{os.getcwd()}`")

# Load and export section
if selected_faiss_path and export_directory:
    # Check if paths exist
    faiss_exists = os.path.exists(selected_faiss_path)
    export_exists = os.path.exists(export_directory)
    
    if not faiss_exists:
        st.error(f"❌ FAISS index not found at: `{selected_faiss_path}`")
        st.info("💡 Make sure you've processed documents in your main app first")
    
    if not export_exists:
        st.error(f"❌ Export directory not accessible: `{export_directory}`")
    
    if faiss_exists and export_exists:
        st.subheader("📊 Vector Database Status")
        
        # Try to load the vectorstore
        with st.spinner("Loading vector database..."):
            vectorstore, load_message = load_vectorstore(selected_faiss_path)
        
        if vectorstore:
            # Display database info
            total_vectors = len(vectorstore.index_to_docstore_id) if hasattr(vectorstore, "index_to_docstore_id") else 0
            st.success(f"✅ Vector database loaded successfully!")
            
            # Show detailed info
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Total Vectors", total_vectors)
            with col_info2:
                # Try to determine vector dimension
                if total_vectors > 0:
                    try:
                        sample_vector = vectorstore.index.reconstruct(0)
                        vector_dim = len(sample_vector)
                        st.metric("Vector Dimension", vector_dim)
                    except:
                        st.metric("Vector Dimension", "Unknown")
            
            st.info(f"📁 **Source:** `{selected_faiss_path}`")
            
            # Export section
            st.subheader("🚀 Export Data")
            
            # Generate filenames with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_filename = f"vector_database_{timestamp}.csv"
            json_filename = f"vector_database_{timestamp}.json"
            csv_full_path = os.path.join(export_directory, csv_filename)
            json_full_path = os.path.join(export_directory, json_filename)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📄 CSV Export")
                st.code(f"File: {csv_filename}", language=None)
                st.code(f"Path: {csv_full_path}", language=None)
                
                if st.button("💾 Save as CSV", type="primary", use_container_width=True):
                    with st.spinner("Exporting to CSV..."):
                        success, message = save_csv_file(vectorstore, csv_full_path)
                        if success:
                            st.success("✅ " + message)
                            st.balloons()
                            
                            # Show file info
                            if os.path.exists(csv_full_path):
                                file_size = os.path.getsize(csv_full_path)
                                st.info(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
                            
                            # Also provide download button as backup
                            try:
                                with open(csv_full_path, 'rb') as f:
                                    st.download_button(
                                        label="📥 Download CSV (backup)",
                                        data=f.read(),
                                        file_name=csv_filename,
                                        mime="text/csv",
                                        use_container_width=True
                                    )
                            except:
                                pass
                        else:
                            st.error("❌ " + message)
            
            with col2:
                st.markdown("### 📋 JSON Export")
                st.code(f"File: {json_filename}", language=None)
                st.code(f"Path: {json_full_path}", language=None)
                
                if st.button("💾 Save as JSON", type="primary", use_container_width=True):
                    with st.spinner("Exporting to JSON..."):
                        success, message = save_json_file(vectorstore, json_full_path)
                        if success:
                            st.success("✅ " + message)
                            st.balloons()
                            
                            # Show file info
                            if os.path.exists(json_full_path):
                                file_size = os.path.getsize(json_full_path)
                                st.info(f"📊 File size: {file_size:,} bytes ({file_size/1024/1024:.1f} MB)")
                            
                            # Also provide download button as backup
                            try:
                                with open(json_full_path, 'rb') as f:
                                    st.download_button(
                                                                                label="📥 Download JSON (backup)",
                                        data=f.read(),
                                        file_name=json_filename,
                                        mime="application/json",
                                        use_container_width=True
                                    )
                            except:
                                pass
                        else:
                            st.error("❌ " + message)
            
            # Export both at once
            st.markdown("---")
            col3, col4, col5 = st.columns([1, 2, 1])
            with col4:
                if st.button("💾 Export Both CSV & JSON", type="secondary", use_container_width=True):
                    with st.spinner("Exporting both formats..."):
                        # Export CSV
                        csv_success, csv_message = save_csv_file(vectorstore, csv_full_path)
                        # Export JSON  
                        json_success, json_message = save_json_file(vectorstore, json_full_path)
                        
                        if csv_success and json_success:
                            st.success("✅ Both files exported successfully!")
                            st.balloons()
                            st.write(f"📄 **CSV:** `{csv_full_path}`")
                            st.write(f"📋 **JSON:** `{json_full_path}`")
                            
                            # Show combined file sizes
                            csv_size = os.path.getsize(csv_full_path) if os.path.exists(csv_full_path) else 0
                            json_size = os.path.getsize(json_full_path) if os.path.exists(json_full_path) else 0
                            total_size = csv_size + json_size
                            st.info(f"📊 Total exported: {total_size:,} bytes ({total_size/1024/1024:.1f} MB)")
                        else:
                            if not csv_success:
                                st.error(f"❌ CSV Error: {csv_message}")
                            if not json_success:
                                st.error(f"❌ JSON Error: {json_message}")
            
            # Preview section
            st.subheader("👀 Data Preview")
            
            with st.expander("Preview Document Content"):
                if hasattr(vectorstore, "index_to_docstore_id") and len(vectorstore.index_to_docstore_id) > 0:
                    # Show first few documents
                    preview_count = min(3, len(vectorstore.index_to_docstore_id))
                    
                    for i in range(preview_count):
                        doc_id = vectorstore.index_to_docstore_id.get(i)
                        if doc_id and doc_id in vectorstore.docstore._dict:
                            doc = vectorstore.docstore._dict[doc_id]
                            
                            st.markdown(f"**Document {i+1}:**")
                            st.write(f"- **Source:** {doc.metadata.get('source', 'Unknown')}")
                            st.write(f"- **File:** {doc.metadata.get('file_name', 'Unknown')}")
                            st.write(f"- **Type:** {doc.metadata.get('file_type', 'Unknown')}")
                            st.write(f"- **Content Length:** {len(doc.page_content)} characters")
                            
                            # Show content preview
                            preview_text = doc.page_content[:300]
                            if len(doc.page_content) > 300:
                                preview_text += "..."
                            
                            st.text_area(
                                f"Content Preview {i+1}:",
                                preview_text,
                                height=100,
                                key=f"preview_{i}"
                            )
                            st.markdown("---")
                else:
                    st.info("No documents found for preview")
            
            # Additional info
            st.subheader("ℹ️ Export Information")
            st.info("""
            **CSV Export includes:**
            - Document metadata (source, file name, type, etc.)
            - Content preview (first 200 characters)
            - Vector embeddings as separate columns
            
            **JSON Export includes:**
            - Complete document content
            - Full metadata
            - Vector embeddings as arrays
            - Export timestamp and statistics
            
            **Files are saved to your specified directory and can also be downloaded as backup.**
            """)
            
            # File locations reminder
            st.success(f"""
            **📁 Your files will be saved to:**
            
            `{export_directory}`
            
            **🔍 To view JSON content, you can use:**
            - VS Code or any text editor
            - Online JSON viewers
            - Python scripts for analysis
            """)
            
        else:
            st.error(f"❌ Failed to load vector database: {load_message}")
            
            # Show debug info
            with st.expander("🔧 Debug Information"):
                st.write(f"**FAISS Path:** `{selected_faiss_path}`")
                st.write(f"**Path exists:** {os.path.exists(selected_faiss_path)}")
                
                if os.path.exists(selected_faiss_path):
                    try:
                        files = os.listdir(selected_faiss_path)
                        st.write(f"**Files in directory:** {files}")
                        faiss_files = [f for f in files if f.endswith(('.faiss', '.pkl'))]
                        st.write(f"**FAISS files found:** {faiss_files}")
                        
                        # Show file sizes
                        for file in faiss_files:
                            file_path = os.path.join(selected_faiss_path, file)
                            file_size = os.path.getsize(file_path)
                            st.write(f"  - {file}: {file_size:,} bytes")
                    except Exception as e:
                        st.write(f"**Error reading directory:** {str(e)}")
                else:
                    st.write("**Directory does not exist**")
                    
                    # Suggest alternatives
                    st.write("**💡 Suggestions:**")
                    st.write("1. Make sure you've run your main app and processed documents")
                    st.write("2. Check if the path is correct")
                    st.write("3. Try looking for 'faiss_index' folders in your project")

else:
    st.info("👆 Please configure both FAISS index path and export directory above")

# Footer
st.markdown("---")
st.markdown("*Vector Database Exporter - Export your FAISS embeddings for external analysis*")
st.caption(f"Default FAISS path: `/home/steve/ai_projects/ai_office_assistant/docling/data/faiss_index`")