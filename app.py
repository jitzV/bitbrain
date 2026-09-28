import streamlit as st
import os
import json
import io
import base64
import urllib.request
import chromadb
import anydoc
import pypdfium2 as pdfium
from PIL import Image
from docx import Document as WordDocument
from uuid import UUID

from profile_store import cleanup_chroma_collection
from ollama_models import list_ollama_models, resolve_installed_model, is_ollama_embedding_error

from llama_index.core import VectorStoreIndex, Document, Settings, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.llms import ChatMessage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
import ollama  # Import raw Ollama client for Vision calls

st.set_page_config(
    page_title="BitBrain | Enterprise Memory Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================
# 1. Configuration & Local Setup
# ==========================================
OLLAMA_LLM_MODEL = "hf.co/unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL"  # Model used for LLM responses
OLLAMA_VISION_MODEL = "hf.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q8_0"  # Model used for Local OCR
OLLAMA_EMBED_MODEL = "nomic-embed-text"

# Configure LlamaIndex to use Local Ollama with a constrained context window
Settings.llm = Ollama(
    model=OLLAMA_LLM_MODEL, 
    request_timeout=120.0,
    context_window=4096
)
Settings.embed_model = OllamaEmbedding(model_name=OLLAMA_EMBED_MODEL)
Settings.node_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=128)

DB_PATH = "./chroma_db"
PROFILES_FILE = "profiles.json"
IMAGE_EXTS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tif', '.tiff', '.gif']
SUPPORTED_EXTS = ['.docx', '.doc', '.xlsx', '.xls', '.pptx', '.ppt', '.rtf', '.odt', '.ods', '.odp', '.epub', '.csv', '.pdf', *IMAGE_EXTS]

# ==========================================
# 2. State & Profile Management
# ==========================================
if "chroma_client" not in st.session_state:
    st.session_state.chroma_client = chromadb.PersistentClient(path=DB_PATH)

# Initialize model configurations in session state
if "ollama_llm_model" not in st.session_state:
    st.session_state.ollama_llm_model = OLLAMA_LLM_MODEL
if "ollama_vision_model" not in st.session_state:
    st.session_state.ollama_vision_model = OLLAMA_VISION_MODEL
if "ollama_embed_model" not in st.session_state:
    st.session_state.ollama_embed_model = OLLAMA_EMBED_MODEL

def update_ollama_settings():
    """Update LlamaIndex settings with current model selections from session state."""
    try:
        Settings.llm = Ollama(
            model=st.session_state.ollama_llm_model, 
            request_timeout=120.0,
            context_window=16384 # <--- Expands chat context window for retrieval
        )
        Settings.embed_model = OllamaEmbedding(model_name=st.session_state.ollama_embed_model)
        Settings.node_parser = SentenceSplitter(chunk_size=2048, chunk_overlap=128)
        return True
    except Exception as e:
        message = str(e)
        if is_ollama_embedding_error(message):
            st.error(
                "Ollama is running without embedding support. Start the Ollama server with `--embeddings` "
                "and then retry the embedding model selection or reindexing."
            )
        else:
            st.error(f"Failed to update model settings: {message}")
        return False

# Initialize with current models
update_ollama_settings()

def load_profiles():
    if os.path.exists(PROFILES_FILE):
        try:
            with open(PROFILES_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError:
            # Keep the corrupted file for inspection instead of crashing
            backup = PROFILES_FILE + ".corrupt"
            os.replace(PROFILES_FILE, backup)
            st.warning(f"'{PROFILES_FILE}' was corrupted and has been moved to '{backup}'. Starting with an empty profile list.")
    return {}

def save_profiles(profiles):
    def uuid_serializer(obj):
        if isinstance(obj, UUID):
            return str(obj)
        raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

    # Atomic write: write to a temp file, then replace, so a crash mid-write can't truncate profiles.json
    tmp_file = PROFILES_FILE + ".tmp"
    with open(tmp_file, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, default=uuid_serializer, indent=2)
    os.replace(tmp_file, PROFILES_FILE)


def get_profile_ingestion_embed_model(profile_id):
    """Return the embedding model that was used when this profile was last ingested."""
    profile = st.session_state.profiles.get(profile_id, {})
    if profile.get("embed_model"):
        return profile["embed_model"]
    return st.session_state.ollama_embed_model


def requires_profile_reindex(profile_id):
    """True when the active embedding model differs from the model used to index the profile."""
    return get_profile_ingestion_embed_model(profile_id) != st.session_state.ollama_embed_model


def render_model_status(phase, model_name):
    icons = {
        "llm": "🤖",
        "vision": "👁️",
        "embed": "🧠",
    }
    label = {
        "llm": "LLM model",
        "vision": "Vision model",
        "embed": "Embedding model",
    }
    st.caption(f"{icons.get(phase, '🔧')} {label.get(phase, phase)}: {model_name}")


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
    profile_name = profile.get("display_name", profile_id)
    collection_id = profile.get("collection_id", profile_id)
    cleanup_chroma_collection(st.session_state.chroma_client, collection_id, DB_PATH)

    del st.session_state.profiles[profile_id]
    save_profiles(st.session_state.profiles)
    for key in list(st.session_state.keys()):
        if key.startswith(f"chat_history_{profile_id}"):
            del st.session_state[key]

    return profile_name

if "profiles" not in st.session_state:
    st.session_state.profiles = load_profiles()

for profile_id, profile in list(st.session_state.profiles.items()):
    if "embed_model" not in profile:
        profile["embed_model"] = st.session_state.ollama_embed_model
        save_profiles(st.session_state.profiles)

def get_vector_store(collection_name):
    chroma_collection = st.session_state.chroma_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store


def get_chroma_collection_uuid(collection_name):
    """Retrieve the internal ChromaDB UUID for a collection."""
    try:
        collection = st.session_state.chroma_client.get_or_create_collection(collection_name)
        uuid = collection.id if hasattr(collection, 'id') else None
        return str(uuid) if uuid else None
    except Exception:
        return None


def get_collection_disk_folders(collection_uuid):
    """Return the on-disk segment folders under DB_PATH that belong to a collection.

    ChromaDB names data folders by *segment* ID, not collection ID, so we map
    collection -> segments via the internal sqlite catalog.
    """
    import sqlite3
    db_file = os.path.join(DB_PATH, "chroma.sqlite3")
    if not collection_uuid or not os.path.exists(db_file):
        return []
    try:
        conn = sqlite3.connect(db_file)
        try:
            rows = conn.execute(
                "SELECT id FROM segments WHERE collection = ?", (collection_uuid,)
            ).fetchall()
        finally:
            conn.close()
        return [
            os.path.join(DB_PATH, row[0])
            for row in rows
            if os.path.isdir(os.path.join(DB_PATH, row[0]))
        ]
    except Exception:
        return []


def get_loaded_document_sources(collection_name):
    """Return the full file paths previously loaded into a profile's collection."""
    try:
        collection = st.session_state.chroma_client.get_collection(collection_name)
        results = collection.get(include=["metadatas"])
        sources = []
        for metadata in results.get("metadatas", []) or []:
            if not metadata:
                continue
            source = metadata.get("source")
            if source and source != "conversational_memory":
                sources.append(source)
        return sorted(set(sources))
    except Exception:
        return []


def sanitize_document_metadata(metadata):
    """Trim oversized metadata so LlamaIndex chunking does not fail when reindexing."""
    if not metadata:
        return {}

    safe_metadata = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, str):
            safe_metadata[key] = value[:512]
        elif isinstance(value, (int, float, bool)):
            safe_metadata[key] = value
        else:
            safe_metadata[key] = str(value)[:512]

    if "source" in safe_metadata and isinstance(safe_metadata["source"], str):
        safe_metadata["source"] = safe_metadata["source"][:512]
    return safe_metadata


def reindex_profile_collection(profile_id):
    """Rebuild a profile's collection using the currently selected embedding model."""
    profile = st.session_state.profiles.get(profile_id)
    if not profile:
        return 0

    collection_name = profile.get("collection_id", profile_id)
    try:
        collection = st.session_state.chroma_client.get_collection(collection_name)
        results = collection.get(include=["documents", "metadatas"])
    except Exception:
        results = {"documents": [], "metadatas": []}

    items = []
    documents = results.get("documents", []) or []
    metadatas = results.get("metadatas", []) or []
    for doc_text, metadata in zip(documents, metadatas):
        if doc_text is None:
            continue
        items.append({"text": doc_text, "metadata": sanitize_document_metadata(metadata or {})})

    cleanup_chroma_collection(st.session_state.chroma_client, collection_name, DB_PATH)
    st.session_state.profiles[profile_id]["embed_model"] = st.session_state.ollama_embed_model
    save_profiles(st.session_state.profiles)
    if not items:
        return 0

    index = get_index(collection_name)
    for item in items:
        doc = Document(text=item["text"], metadata=item["metadata"])
        index.insert(doc)
    return len(items)


def get_index(collection_name):
    vector_store = get_vector_store(collection_name)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    return VectorStoreIndex.from_vector_store(vector_store, storage_context=storage_context)


def inject_enterprise_theme():
    stylesheet_path = os.path.join(os.path.dirname(__file__), "enterprise.css")
    with open(stylesheet_path, "r", encoding="utf-8") as stylesheet:
        css = stylesheet.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_metric_card(label, value, subtext):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-subtext">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_pill(label):
    st.markdown(f'<span class="status-pill">{label}</span>', unsafe_allow_html=True)


# ==========================================
# 3. Local Ollama Vision OCR Fallback
# ==========================================
def stop_ollama_model(model_name):
    """Explicitly unload a local Ollama model after a session completes."""
    try:
        payload = json.dumps({"name": model_name}).encode("utf-8")
        request = urllib.request.Request(
            "http://localhost:11434/api/stop",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=15) as response:
            response.read()
        return True
    except Exception:
        return False


def run_ollama_vision_ocr(file_path):
    """Renders PDF pages as images and runs Ollama Vision model for local OCR."""
    pdf = pdfium.PdfDocument(file_path)
    num_pages = len(pdf)
    extracted_pages = []
    status_placeholder = st.empty()

    # Highly specific prompt for dense ID/Passport extraction
    ocr_prompt = (
        "You are a strict, highly accurate document parser. Perform dense OCR on this image. "
        "Extract every single piece of visible text, including tiny print, serial numbers, "
        "dates, and Machine Readable Zones (MRZ) exactly as they appear. Do not summarize, "
        "do not reformat, and do not include conversational filler. Output the raw text only."
    )

    try:
        for page_idx in range(num_pages):
            status_placeholder.info(f"👁️ Running Ollama Vision OCR on Page {page_idx + 1}/{num_pages}...")
            
            # 1. Render at a decent scale, but cap the max dimension
            page = pdf[page_idx]
            image = page.render(scale=2).to_pil()
            
            # 2. Cap at 2048x2048 to retain ID card details without crashing speed
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            
            # 3. Use PNG (lossless) to preserve edge sharpness on small fonts
            img_byte_arr = io.BytesIO()
            #image.save(img_byte_arr, format='PNG')
            image.save(img_byte_arr, format='JPEG', quality=85)
            img_bytes = img_byte_arr.getvalue()

            # Call local Ollama Vision model with zero temperature
            response = ollama.chat(
                model=st.session_state.ollama_vision_model,
                messages=[{
                    'role': 'user',
                    'content': ocr_prompt,
                    'images': [img_bytes]
                }],
                options={
                    'temperature': 0.0, 
                    'top_k': 1,
                    'num_predict': 2048, # Ensure it doesn't cut off long document outputs
                    'num_ctx': 16384  # <--- Add this line to restrict memory usage
                }
            )
            page_text = response['message']['content']
            extracted_pages.append(f"--- Page {page_idx + 1} ---\n{page_text}")

        return "\n\n".join(extracted_pages)
    finally:
        status_placeholder.empty()
        pdf.close()
        # Keep the model loaded if you are doing bulk ingestion!
        # stop_ollama_model(st.session_state.ollama_vision_model)

def run_ollama_image_ocr(file_path):
    """Handles direct image uploads (like passport JPGs/PNGs)."""
    ocr_prompt = (
        "You are a strict, highly accurate document parser. Perform dense OCR on this image. "
        "Extract every single piece of visible text, including tiny print, serial numbers, "
        "dates, and Machine Readable Zones (MRZ) exactly as they appear. Do not summarize, "
        "do not reformat, and do not include conversational filler. Output the raw text only."
    )
    
    try:
        with Image.open(file_path) as source_image:
            image = source_image.convert("RGB")
            # Apply the same 2048px ceiling for direct image uploads
            image.thumbnail((2048, 2048), Image.Resampling.LANCZOS)
            
            image_bytes = io.BytesIO()
            image.save(image_bytes, format="PNG")

        response = ollama.chat(
            model=st.session_state.ollama_vision_model,
            messages=[{
                'role': 'user',
                'content': ocr_prompt,
                'images': [image_bytes.getvalue()]
            }],
            options={
                'temperature': 0.0,
                'top_k': 1,
                'num_predict': 2048
            }
        )
        return response['message']['content']
    finally:
        pass # stop_ollama_model(st.session_state.ollama_vision_model)

def parse_document(file_path):
    file_ext = os.path.splitext(file_path)[1].lower()
    if file_ext in IMAGE_EXTS:
        return run_ollama_image_ocr(file_path)

    try:
        markdown = anydoc.to_markdown(file_path)
    except Exception as e:
        if file_ext != ".pdf":
            raise
        st.warning(f"PDF text extraction failed for {os.path.basename(file_path)}. Trying local Ollama Vision OCR ({st.session_state.ollama_vision_model})...")
        return run_ollama_vision_ocr(file_path)

    if file_ext == ".pdf" and not (markdown or "").strip():
        st.warning(f"No text found in {os.path.basename(file_path)}. Trying local Ollama Vision OCR ({st.session_state.ollama_vision_model})...")
        return run_ollama_vision_ocr(file_path)
    return markdown

# ==========================================
# 4. Sidebar UI - Profile Manager
# ==========================================
inject_enterprise_theme()

st.sidebar.markdown("<div class='sidebar-section'><div class='eyebrow' style='letter-spacing:0.08em;'>Enterprise Memory</div><h3 style='margin:0.2rem 0 0; color:#f8fafc;'>BitBrain Profiles</h3></div>", unsafe_allow_html=True)

# Create New Profile
with st.sidebar.expander("⚙️ Model Settings", expanded=False):
    st.markdown("<div class='tiny-label'>Configure Ollama Models</div>", unsafe_allow_html=True)
    
    available_ollama_models = list_ollama_models()
    if available_ollama_models:
        global_model = st.session_state.ollama_llm_model
        if global_model not in available_ollama_models:
            global_model = available_ollama_models[0]
            st.session_state.ollama_llm_model = global_model
            update_ollama_settings()

        if "llm_model_select" not in st.session_state or st.session_state.llm_model_select not in available_ollama_models:
            st.session_state.llm_model_select = global_model
        llm_model = st.selectbox(
            "LLM Model",
            options=available_ollama_models,
            key="llm_model_select",
            help="Default chat model; profiles can override this in their chat tab.",
        )
        if llm_model != st.session_state.ollama_llm_model:
            st.session_state.ollama_llm_model = llm_model
            update_ollama_settings()
    else:
        st.warning("No Ollama models found. Make sure Ollama is running and has models installed.")
    
    if available_ollama_models:
        vision_default = resolve_installed_model(st.session_state.ollama_vision_model, available_ollama_models)
        vision_key = "vision_model_select"
        if st.session_state.get(vision_key) not in available_ollama_models:
            st.session_state[vision_key] = vision_default
        vision_model = st.selectbox(
            "Vision Model",
            options=available_ollama_models,
            key=vision_key,
            help="Ollama model used for image and scanned-PDF OCR.",
        )
        if vision_model != st.session_state.ollama_vision_model:
            st.session_state.ollama_vision_model = vision_model

        embed_default = resolve_installed_model(st.session_state.ollama_embed_model, available_ollama_models)
        embed_key = "embed_model_select"
        if st.session_state.get(embed_key) not in available_ollama_models:
            st.session_state[embed_key] = embed_default
        embed_model = st.selectbox(
            "Embedding Model",
            options=available_ollama_models,
            key=embed_key,
            help="Ollama model used to embed memories for vector search.",
        )
        if embed_model != st.session_state.ollama_embed_model:
            st.session_state.ollama_embed_model = embed_model
            update_ollama_settings()
    


st.sidebar.divider()

# Create New Profile
with st.sidebar.expander("➕ Create New Profile", expanded=False):
    new_profile_name = st.text_input("Profile Name")
    new_profile_desc = st.text_area("Description")
    if st.button("Create", use_container_width=True):
        if new_profile_name:
            col_id = "".join(c if c.isalnum() else "_" for c in new_profile_name).lower().strip("_")
            chroma_uuid = get_chroma_collection_uuid(col_id)
            st.session_state.profiles[col_id] = {
                "display_name": new_profile_name,
                "description": new_profile_desc,
                "collection_id": col_id,
                "chroma_uuid": chroma_uuid,
                "embed_model": st.session_state.ollama_embed_model,
            }
            save_profiles(st.session_state.profiles)
            st.sidebar.success(f"Profile '{new_profile_name}' created!")
            st.rerun()

# Select Active Profile
active_profile_id = None
if st.session_state.profiles:
    profile_options = {k: v["display_name"] for k, v in st.session_state.profiles.items()}
    profile_choices = ["Select a profile..."] + list(profile_options.keys())
    selected_profile = st.sidebar.selectbox(
        "Select Active Profile",
        options=profile_choices,
        index=0,
        format_func=lambda x: "Select a profile..." if x == "Select a profile..." else profile_options[x],
    )
    active_profile_id = None if selected_profile == "Select a profile..." else selected_profile

    st.sidebar.markdown("<div class='sidebar-section'><div class='metric-label' style='margin-bottom:0.35rem;'>Workspace</div><div style='color:#f8fafc; font-weight:600;'>" + str(len(profile_options)) + " profiles available</div></div>", unsafe_allow_html=True)

    if active_profile_id is not None:
        # Display ChromaDB Collection ID
        collection_id = st.session_state.profiles[active_profile_id].get("collection_id", active_profile_id)
        chroma_uuid = st.session_state.profiles[active_profile_id].get("chroma_uuid")
        
        # Always fetch the live internal UUID from ChromaDB
        chroma_uuid = get_chroma_collection_uuid(collection_id)
        if chroma_uuid != st.session_state.profiles[active_profile_id].get("chroma_uuid"):
            st.session_state.profiles[active_profile_id]["chroma_uuid"] = chroma_uuid
            save_profiles(st.session_state.profiles)

        disk_folders = get_collection_disk_folders(chroma_uuid)
        disk_html = (
            "".join(
                f"<div style='font-family: monospace; font-size: 0.68rem; color:#94a3b8; word-break: break-all; margin-top:0.35rem;'>{folder}</div>"
                for folder in disk_folders
            )
            if disk_folders
            else "<div style='font-size: 0.68rem; color:#94a3b8; margin-top:0.35rem;'>No data folder yet (empty collection)</div>"
        )

        st.sidebar.markdown(
            f"""<div class='sidebar-section'>
                <div class='metric-label' style='margin-bottom:0.35rem;'>Disk Folders</div>
                {disk_html}
            </div>""",
            unsafe_allow_html=True
        )
        st.sidebar.divider()
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

            if st.session_state.get("delete_in_progress") == active_profile_id:
                profile_name = st.session_state.profiles[active_profile_id]["display_name"]
                st.info("Deleting profile and cleaning local memory data...")
                with st.spinner("Removing profile collection from ChromaDB..."):
                    deleted_name = delete_profile(active_profile_id)
                st.success(f"Profile '{deleted_name}' deleted.")
                del st.session_state["delete_in_progress"]
                st.rerun()

            confirm_delete = st.checkbox(
                f"I confirm deleting '{st.session_state.profiles[active_profile_id]['display_name']}'",
                key=f"confirm_delete_{active_profile_id}",
            )
            if st.button(
                "Delete Profile",
                type="secondary",
                use_container_width=True,
                disabled=not confirm_delete or st.session_state.get("delete_in_progress") == active_profile_id,
            ):
                st.session_state["delete_in_progress"] = active_profile_id
                st.rerun()

# ==========================================
# 5. Main Stage - Knowledge Base & Chat
# ==========================================
if not active_profile_id:
    st.markdown(
        """
        <div class="enterprise-header">
            <div class="eyebrow workspace-eyebrow" style="display:block; color:#c8ed55; -webkit-text-fill-color:#c8ed55; font-family:Bahnschrift, Segoe UI, sans-serif; font-size:0.8rem; font-weight:700; line-height:1.5; letter-spacing:0.08em; text-transform:uppercase; visibility:visible; opacity:1;">Memory Workspace</div>
            <h1>Welcome</h1>
            <p>Please create or select a memory profile from the sidebar to begin.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info("No active profile is selected yet. Use the sidebar to create a profile or choose an existing one.")
    st.stop()

profile = st.session_state.profiles[active_profile_id]
profile_name = profile["display_name"]
profile_desc = profile["description"]
collection_id = profile.get("collection_id", active_profile_id)
chroma_uuid = profile.get("chroma_uuid")

# Always fetch the live internal UUID from ChromaDB
chroma_uuid = get_chroma_collection_uuid(collection_id)
if chroma_uuid != profile.get("chroma_uuid"):
    st.session_state.profiles[active_profile_id]["chroma_uuid"] = chroma_uuid
    save_profiles(st.session_state.profiles)

header_disk_folders = get_collection_disk_folders(chroma_uuid)
header_disk_html = (
    "".join(
        f"<div style='font-family: monospace; font-size: 0.75rem; color: #cbd5e1; word-break: break-all;'>{folder}</div>"
        for folder in header_disk_folders
    )
    if header_disk_folders
    else "<div style='font-size: 0.75rem; color: #94a3b8;'>No data folder yet (empty collection)</div>"
)

st.markdown(
    f"""
    <div class="enterprise-header">
        <div class="eyebrow workspace-eyebrow" style="display:block; color:#c8ed55; -webkit-text-fill-color:#c8ed55; font-family:Bahnschrift, Segoe UI, sans-serif; font-size:0.8rem; font-weight:700; line-height:1.5; letter-spacing:0.08em; text-transform:uppercase; visibility:visible; opacity:1;">Enterprise Memory Workspace</div>
        <h1>{profile_name}</h1>
        <p>{profile_desc or 'Operational knowledge repository and collaborative memory assistant.'}</p>
        <div style='margin-top: 0.5rem; padding-top: 0.75rem; border-top: 1px solid rgba(148, 163, 184, 0.2);'>
            <div style='font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.1em; margin-bottom: 0.3rem;'>Disk Folders</div>
            {header_disk_html}
        </div>
        <span class="status-badge">Profile online</span>
        <span class="status-badge">Knowledge synced</span>
    </div>
    """,
    unsafe_allow_html=True,
)

kpi_1, kpi_2, kpi_3 = st.columns(3)
with kpi_1:
    render_metric_card("Profiles", str(len(st.session_state.profiles)), "Active workspaces")
with kpi_2:
    loaded_sources = get_loaded_document_sources(active_profile_id)
    render_metric_card("Documents", str(len(loaded_sources)), "Indexed sources")
with kpi_3:
    chat_count = len(st.session_state.get(f"chat_history_{active_profile_id}", []))
    render_metric_card("Conversations", str(chat_count), "Session exchanges")

st.markdown('<div class="command-row">', unsafe_allow_html=True)
render_status_pill("Local LLM online")
render_status_pill("Vision OCR ready")
render_status_pill("Embedding model active")
st.markdown('</div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["📚 Knowledge Base (Ingestion)", "💬 Agent Chat (Retrieval)"])

# ----------------------------------------------------
# TAB 1: Ingestion (Conversational & File/Folder Parsing)
# ----------------------------------------------------
with tab1:
    st.markdown("""
    <div class="section-shell">
        <div class="tiny-label">Knowledge intake</div>
        <div style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin-bottom:0.2rem;">Add Memory to Profile</div>
    </div>
    """, unsafe_allow_html=True)
    ingest_type = st.radio("Choose Source:", ["Option 1: Conversational Input", "Option 2: Local File or Folder Parsing (AnyDoc + Vision OCR)"])

    if ingest_type == "Option 1: Conversational Input":
        st.markdown("<div class='info-panel'>Chat with the system to directly inject facts, rules, or data into the memory vector DB.</div>", unsafe_allow_html=True)
        st.info("This path uses the embedding model to vectorize the text you enter.")
        render_model_status("embed", st.session_state.ollama_embed_model)
        fact_input = st.chat_input("Tell the agent something to remember...")
        
        if fact_input:
            with st.chat_message("user"):
                st.write(fact_input)
            
            with st.spinner("Processing and saving to Vector DB..."):
                index = get_index(active_profile_id)
                st.caption(f"🧠 Embedding model in use: {st.session_state.ollama_embed_model}")
                doc = Document(text=fact_input, metadata=sanitize_document_metadata({"source": "conversational_memory"}))
                index.insert(doc)
                st.session_state.profiles[active_profile_id]["embed_model"] = st.session_state.ollama_embed_model
                save_profiles(st.session_state.profiles)
            
            with st.chat_message("assistant"):
                st.success("✅ Memory parsed and saved to Vector DB.")

    elif ingest_type == "Option 2: Local File or Folder Parsing (AnyDoc + Vision OCR)":
        st.markdown("<div class='info-panel'>Enter the path to a <strong>specific file</strong> or an <strong>entire directory</strong>. AnyDoc parses digital docs natively, and scanned PDFs automatically switch to Ollama Vision OCR.</div>", unsafe_allow_html=True)
        st.info("Model activity is shown below while files are processed.")
        render_model_status("llm", st.session_state.ollama_llm_model)
        render_model_status("vision", st.session_state.ollama_vision_model)
        render_model_status("embed", st.session_state.ollama_embed_model)
        
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
                        file_ext = os.path.splitext(file_path)[1].lower()
                        if file_ext in IMAGE_EXTS:
                            st.caption(f"👁️ Vision model in use for image OCR: {st.session_state.ollama_vision_model}")
                        elif file_ext == ".pdf":
                            st.caption(f"👁️ Vision model in use for scanned PDF OCR fallback: {st.session_state.ollama_vision_model}")
                        else:
                            st.caption(f"📄 Standard document parsing path (no vision model required)")

                        try:
                            markdown_content = parse_document(file_path)
                            if markdown_content and markdown_content.strip():
                                st.caption(f"🧠 Embedding model in use: {st.session_state.ollama_embed_model}")
                                doc = Document(text=markdown_content, metadata=sanitize_document_metadata({"source": file_path}))
                                index.insert(doc)
                                st.session_state.profiles[active_profile_id]["embed_model"] = st.session_state.ollama_embed_model
                                save_profiles(st.session_state.profiles)
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

    loaded_sources = get_loaded_document_sources(active_profile_id)
    st.markdown("""
    <div class="section-shell">
        <div class="tiny-label">Context inventory</div>
        <div style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin-bottom:0.25rem;">Loaded Documents</div>
    </div>
    """, unsafe_allow_html=True)
    if loaded_sources:
        st.markdown(f"<div class='source-list'>{'<br>'.join([f'• {s}' for s in loaded_sources])}</div>", unsafe_allow_html=True)
    else:
        st.info("No local documents have been loaded into this profile yet.")

    st.markdown("""
    <div class="section-shell" style="margin-top:1rem;">
        <div class="tiny-label">Interaction</div>
        <div style="font-size:1.15rem; font-weight:700; color:#f8fafc; margin-bottom:0.25rem;">Conversation</div>
    </div>
    """, unsafe_allow_html=True)
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
    ingestion_model = get_profile_ingestion_embed_model(active_profile_id)
    render_model_status("embed", st.session_state.ollama_embed_model)
    st.caption(f"🧠 Ingestion embedding model: {ingestion_model}")
    if requires_profile_reindex(active_profile_id):
        st.warning(
            "⚠️ The active embedding model is different from the model used to ingest this profile. "
            "If you changed the embedding model, reindex the profile before relying on old memories."
        )
        if st.button("Reindex this profile with the current embedding model", use_container_width=True):
            with st.spinner("Rebuilding this profile's vector store using the active embedding model..."):
                count = reindex_profile_collection(active_profile_id)
            st.success(f"✅ Reindexed {count} memory entries for '{profile_name}'.")
            st.rerun()

    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    if available_ollama_models:
        profile_model = profile.get("llm_model")
        default_model = (
            profile_model if profile_model in available_ollama_models
            else st.session_state.ollama_llm_model if st.session_state.ollama_llm_model in available_ollama_models
            else available_ollama_models[0]
        )
        model_key = f"profile_llm_model_{active_profile_id}"
        if model_key not in st.session_state:
            st.session_state[model_key] = default_model

        selected_chat_model = st.selectbox(
            "Chat LLM model",
            options=available_ollama_models,
            key=model_key,
            help="Choose the Ollama model used for this profile's chat responses.",
        )
        if profile.get("llm_model") != selected_chat_model:
            profile["llm_model"] = selected_chat_model
            save_profiles(st.session_state.profiles)

        render_model_status("llm", selected_chat_model)
        chat_llm = Ollama(
            model=selected_chat_model,
            request_timeout=120.0,
            context_window=16384,
        )
        chat_engine = index.as_chat_engine(
            chat_mode="context",
            llm=chat_llm,
            chat_history=[
                ChatMessage(role=message["role"], content=message["content"])
                for message in st.session_state[chat_key]
            ],
            similarity_top_k=3,
            system_prompt="You are BitBrain, a helpful personal memory assistant. Use the provided context from the memory profile to accurately answer questions."
        )

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
    else:
        st.warning("Chat is unavailable because no Ollama models were found. Start Ollama and install a model to continue.")