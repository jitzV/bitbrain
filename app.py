import streamlit as st
import os
import json
import io
import base64
import chromadb
import anydoc
import pypdfium2 as pdfium
from PIL import Image
from docx import Document as WordDocument

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import ollama  # Import raw Ollama client for Vision calls

# ==========================================
# 1. Configuration & Local Setup
# ==========================================
OLLAMA_LLM_MODEL = "hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M"  # Model used for LLM responses
OLLAMA_VISION_MODEL = "maternion/LightOnOCR-2:1b"  # Model used for Local OCR
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Configure LlamaIndex to use Local Ollama with a constrained context window
Settings.llm = Ollama(
    model=OLLAMA_LLM_MODEL, 
    request_timeout=120.0,
    context_window=4096
)
Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL)

DB_PATH = "./chroma_db"
PROFILES_FILE = "profiles.json"
IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff', '.gif']
SUPPORTED_EXTS = ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.rtf', '.odt', '.ods', '.odp', '.epub', '.csv', '.pdf', *IMAGE_EXTS]

# ==========================================
# 2. State & Profile Management
# ==========================================
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient(path=DB_PATH)

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        with open(PROFILES_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_profiles(profiles):
    with open(PROFILES_FILE, 'w') as f:
        json.dump(profiles, f)

def format_chat_as_markdown(messages, profile_name):
    lines = [f"# BitBrain Conversation - {profile_name}", ""]
    for message in messages:
        role = "You" if message["role"] == "user" else "BitBrain"
        lines.extend([f"## {role}", message["content"], ""])
    return "\n".join(lines)

def format_chat_as_docx(messages, profile_name):
    document = WordDocument()
    document.add_heading(f"BitBrain Conversation - {profile_name}", level=1)
    for message in messages:
        role = "You" if message["role"] == "user" else "BitBrain"
        document.add_heading(role, level=2)
        document.add_paragraph(message["content"])

    output = io.BytesIO()
    document.save(output)
    return output.getvalue()

def delete_profile(profile_id):
    profile = st.session_state.profiles[profile_id]
    collection_id = profile.get("collection_id", profile_id)
    try:
        st.session_state.chroma_client.delete_collection(collection_id)
    except Exception:
        pass

    del st.session_state.profiles[profile_id]
    save_profiles(st.session_state.profiles)
    for key in list(st.session_state.keys()):
        if key.startswith(f"chat_history_{profile_id}"):
            del st.session_state[key]

if "profiles" not in st.session_state:
    st.session_state.profiles = load_profiles()

def get_vector_store(collection_name):
    chroma_collection = st.session_state.chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store

def get_index(collection_name):
    vector_store = get_vector_store(collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)

# ==========================================
# 3. Local Ollama Vision OCR Fallback
# ==========================================
def run_ollama_vision_ocr(file_path):
    """Renders PDF pages as images and runs Ollama Vision model for local OCR."""
    pdf = pdfium.PdfDocument(file_path)
    num_pages = len(pdf)
    extracted_pages = []

    status_placeholder = st.empty()

    for page_idx in range(num_pages):
        status_placeholder.info(f"👁️ Running Ollama Vision OCR on Page {page_idx + 1}/{num_pages}...")
        
        # Render PDF page to PIL Image
        page = pdf[page_idx]
        image = page.render(scale=2).to_pil()
        
        # Convert image to PNG bytes for Ollama API
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # Call local Ollama Vision model
        response = ollama.chat(
            model=OLLAMA_VISION_MODEL,
            messages=[{
                'role': 'user',
                'content': 'Perform OCR on this image. Extract and transcribe all visible text exactly as it appears. Output plain text or markdown only without conversational preambles.',
                'images': [img_bytes]
            }]
        )
        page_text = response['message']['content']
        extracted_pages.append(f"--- Page {page_idx + 1} ---\n{page_text}")

    status_placeholder.empty()
    return "\n\n".join(extracted_pages)

def run_ollama_image_ocr(file_path):
    with Image.open(file_path) as source_image:
        image = source_image.convert("RGB")
        image_bytes = io.BytesIO()
        image.save(image_bytes, format="PNG")

    response = ollama.chat(
        model=OLLAMA_VISION_MODEL,
        messages=[{
            'role': 'user',
            'content': 'Perform OCR on this image. Extract and transcribe all visible text exactly as it appears. Output plain text or markdown only without conversational preambles.',
            'images': [image_bytes.getvalue()]
        }]
    )
    return response['message']['content']

def parse_document(file_path):
    if os.path.splitext(file_path)[1].lower() in IMAGE_EXTS:
        return run_ollama_image_ocr(file_path)

    try:
        # 1. Attempt native AnyDoc parsing first (Fast)
        return anydoc.to_markdown(file_path)
    except Exception as e:
        # 2. Catch scanned PDF error and fallback to local Ollama Vision OCR
        if "OCR is required" in str(e):
            st.warning(f"Scanned PDF detected for {os.path.basename(file_path)}. Triggering local Ollama Vision model ({OLLAMA_VISION_MODEL})...")
            return run_ollama_vision_ocr(file_path)
        else:
            raise e

# ==========================================
# 4. Sidebar UI - Profile Manager
# ==========================================
st.sidebar.title("🧠 BitBrain Profiles")

# Create New Profile
with st.sidebar.expander("➕ Create New Profile", expanded=False):
    new_profile_name = st.text_input("Profile Name")
    new_profile_desc = st.text_area("Description")
    if st.button("Create", use_container_width=True):
        if new_profile_name:
            col_id = "".join(c if c.isalnum() else "_" for c in new_profile_name).lower().strip("_")
            st.session_state.profiles[col_id] = {
                "display_name": new_profile_name,
                "description": new_profile_desc,
                "collection_id": col_id
            }
            save_profiles(st.session_state.profiles)
            st.sidebar.success(f"Profile '{new_profile_name}' created!")
            st.rerun()

# Select Active Profile
active_profile_id = None
if st.session_state.profiles:
    profile_options = {k: v["display_name"] for k, v in st.session_state.profiles.items()}
    active_profile_id = st.sidebar.selectbox(
        "Select Active Profile", 
        options=list(profile_options.keys()), 
        format_func=lambda x: profile_options[x]
    )
    
    # Edit/Rename Active Profile
    with st.sidebar.expander("✏️ Edit Profile Settings"):
        edit_name = st.text_input("Rename", value=st.session_state.profiles[active_profile_id]["display_name"])
        edit_desc = st.text_area("Update Description", value=st.session_state.profiles[active_profile_id]["description"])
        if st.button("Save Changes", use_container_width=True):
            st.session_state.profiles[active_profile_id]["display_name"] = edit_name
            st.session_state.profiles[active_profile_id]["description"] = edit_desc
            save_profiles(st.session_state.profiles)
            st.rerun()

    with st.sidebar.expander("🗑️ Delete Profile"):
        st.warning("This permanently deletes the profile and all indexed memories.")
        confirm_delete = st.checkbox(
            f"I confirm deleting '{st.session_state.profiles[active_profile_id]['display_name']}'",
            key=f"confirm_delete_{active_profile_id}",
        )
        if st.button(
            "Delete Profile",
            type="secondary",
            use_container_width=True,
            disabled=not confirm_delete,
        ):
            delete_profile(active_profile_id)
            st.rerun()

# ==========================================
# 5. Main Stage - Knowledge Base & Chat
# ==========================================
if not active_profile_id:
    st.info("👈 Please create or select a memory profile from the sidebar to begin.")
    st.stop()

st.title(f"BitBrain Profile: {st.session_state.profiles[active_profile_id]['display_name']}")
st.caption(st.session_state.profiles[active_profile_id]['description'])

tab1, tab2 = st.tabs(["📚 Knowledge Base (Ingestion)", "💬 Agent Chat (Retrieval)"])

# ----------------------------------------------------
# TAB 1: Ingestion (Conversational & File/Folder Parsing)
# ----------------------------------------------------
with tab1:
    st.subheader("Add Memory to Profile")
    ingest_type = st.radio("Choose Source:", ["Option 1: Conversational Input", "Option 2: Local File or Folder Parsing (AnyDoc + Vision OCR)"])

    if ingest_type == "Option 1: Conversational Input":
        st.markdown("Chat with the system to directly inject facts, rules, or data into the memory vector DB.")
        fact_input = st.chat_input("Tell the agent something to remember...")
        
        if fact_input:
            with st.chat_message("user"):
                st.write(fact_input)
            
            with st.spinner("Processing and saving to Vector DB..."):
                index = get_index(active_profile_id)
                doc = Document(text=fact_input, metadata={"source": "conversational_memory"})
                index.insert(doc)
            
            with st.chat_message("assistant"):
                st.success("✅ Memory parsed and saved to Vector DB.")

    elif ingest_type == "Option 2: Local File or Folder Parsing (AnyDoc + Vision OCR)":
        st.markdown("Enter the path to a **specific file** or an **entire directory**. AnyDoc parses digital docs natively, and scanned PDFs automatically switch to Ollama Vision OCR.")
        
        target_path = st.text_input("Enter local file or folder path (e.g., D:/Scans/Doc.pdf or D:/Jithin_Sync/Docs):")
        
        if st.button("Start Parsing"):
            target_path = target_path.strip().strip('"').strip("'")
            
            if not os.path.exists(target_path):
                st.error("The specified path does not exist. Please check the file/folder path.")
            else:
                files_to_parse = []
                
                # Single file vs Directory
                if os.path.isfile(target_path):
                    if any(target_path.lower().endswith(ext) for ext in SUPPORTED_EXTS):
                        files_to_parse.append(target_path)
                    else:
                        st.error(f"Unsupported file format. Supported extensions: {', '.join(SUPPORTED_EXTS)}")
                elif os.path.isdir(target_path):
                    for root, _, files in os.walk(target_path):
                        for file in files:
                            if any(file.lower().endswith(ext) for ext in SUPPORTED_EXTS):
                                files_to_parse.append(os.path.join(root, file))
                    if not files_to_parse:
                        st.warning("No supported documents found in that folder.")

                # Process identified files
                if files_to_parse:
                    index = get_index(active_profile_id)
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file_path in enumerate(files_to_parse):
                        status_text.text(f"Parsing ({i+1}/{len(files_to_parse)}): {os.path.basename(file_path)}")
                        
                        try:
                            markdown_content = parse_document(file_path)
                            if markdown_content and markdown_content.strip():
                                doc = Document(text=markdown_content, metadata={"source": file_path})
                                index.insert(doc)
                            else:
                                st.warning(f"No text extracted from {os.path.basename(file_path)}")
                        except Exception as e:
                            st.error(f"Error parsing {os.path.basename(file_path)}: {str(e)}")
                            
                        progress_bar.progress((i + 1) / len(files_to_parse))
                        
                    status_text.text(f"✅ Successfully processed {len(files_to_parse)} file(s) into Vector DB!")
                    st.toast("Ingestion complete!")

# ----------------------------------------------------
# TAB 2: Agent Chat (Retrieval)
# ----------------------------------------------------
with tab2:
    chat_key = f"chat_history_{active_profile_id}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = []

    chat_history = st.session_state[chat_key]
    profile_name = st.session_state.profiles[active_profile_id]["display_name"]
    st.subheader("Conversation")
    action_col, export_col = st.columns([1, 2])
    with action_col:
        if st.button("Clear chat", use_container_width=True, disabled=not chat_history):
            st.session_state[chat_key] = []
            st.rerun()
    with export_col:
        markdown_data = format_chat_as_markdown(chat_history, profile_name)
        docx_data = format_chat_as_docx(chat_history, profile_name)
        download_col, word_col = st.columns(2)
        with download_col:
            st.download_button(
                "Download Markdown",
                data=markdown_data,
                file_name=f"bitbrain-{active_profile_id}-conversation.md",
                mime="text/markdown",
                use_container_width=True,
                disabled=not chat_history,
            )
        with word_col:
            st.download_button(
                "Download Word",
                data=docx_data,
                file_name=f"bitbrain-{active_profile_id}-conversation.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                disabled=not chat_history,
            )

    index = get_index(active_profile_id)
    chat_engine = index.as_chat_engine(
        chat_mode="context",
        similarity_top_k=3,
        system_prompt="You are BitBrain, a helpful personal memory assistant. Use the provided context from the memory profile to accurately answer questions."
    )

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    user_query = st.chat_input("Ask a question based on your memory profile...")
    if user_query:
        st.session_state[chat_key].append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.write(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching BitBrain Vector DB & Thinking..."):
                response = chat_engine.chat(user_query)
                st.write(response.response)
                
                if response.source_nodes:
                    with st.expander("View Retrieved Context"):
                        for node in response.source_nodes:
                            st.caption(f"Source: {node.metadata.get('source', 'Unknown')}")
                            st.text(node.text[:300] + "...")

        st.session_state[chat_key].append({"role": "assistant", "content": response.response})