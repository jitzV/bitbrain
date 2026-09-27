# BitBrain

BitBrain is a local, profile-based personal memory assistant built with Streamlit, ChromaDB, and LlamaIndex. It lets you create separate knowledge profiles, ingest memories from chat or local documents, and then ask questions against that knowledge using local Ollama models.

## Overview

This app stores memory in a local vector database and supports:

- Multiple named memory profiles
- Conversational memory ingestion
- File and folder parsing for common document formats
- Image document imports with local Ollama Vision OCR
- OCR fallback for scanned PDFs using a local Ollama vision model
- Retrieval-based chat that answers from the current profile’s knowledge

## Features

- Profile manager in the sidebar
- Per-profile isolated vector collections
- Import facts directly from chat input
- Parse supported files and folders using AnyDoc
- Import PNG, JPEG, WEBP, BMP, TIFF, and GIF images
- Convert imported images to PNG before sending them to the local Ollama Vision model
- Automatically detect scanned PDFs and run local OCR
- Ask questions using the uploaded knowledge as context

## Supported Document Types

The project supports these file extensions:

- `.docx`, `.doc`
- `.xlsx`, `.xls`
- `.pptx`, `.ppt`
- `.rtf`, `.odt`, `.ods`, `.odp`
- `.epub`
- `.csv`
- `.pdf`
- `.png`, `.jpg`, `.jpeg`, `.webp`
- `.bmp`, `.tif`, `.tiff`, `.gif`

## Tech Stack

- Python
- Streamlit
- ChromaDB
- LlamaIndex
- Ollama
- AnyDoc
- PyPDFium2
- Pillow

## Prerequisites

Before running the app, ensure the following are installed:

- Python 3.10+
- pip
- Ollama
- Local Ollama models pulled into your environment

Required Ollama models used by the app:

- `hf.co/unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL` (LLM/Chat Model)
- `hf.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q8_0` (Vision/OCR Model)
- `nomic-embed-text` (Embedding Model)

You can pull them with:

```bash
ollama pull hf.co/unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
ollama pull hf.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q8_0
ollama pull nomic-embed-text
```

> Ensure the Ollama service is running locally before launching the app.

## Installation

From the project root:

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirement.txt
```

## Run the App

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Project Structure

```text
bit-brain/
├── app.py                # Main Streamlit application
├── profiles.json         # Saved profile metadata
├── requirement.txt       # Python dependencies
├── chroma_db/            # Local vector database storage
└── README.md             # Project documentation
```

## RAG Architecture Deep Dive

BitBrain employs a sophisticated, local Retrieval-Augmented Generation (RAG) pipeline to ensure all answers are grounded in the user's private memory profiles. This architecture is entirely local, relying on models and databases running on the user's machine via Ollama.

**1. Ingestion (The Memory Collector):**
The system accepts two types of inputs:
*   **Conversational Memory:** Direct facts typed by the user are immediately processed.
*   **Document Ingestion:** Local files and folders are scanned.
    *   **Native Parsing:** Documents like `.docx`, `.pdf` (digital), etc., are parsed using `AnyDoc` to extract clean Markdown content.
    *   **Visual Parsing (OCR):** For images (`.png`, `.jpg`, etc.) and scanned PDFs, the system triggers a local OCR process using `PyPDFium2` and the **Ollama Vision Model** (`hf.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q8_0`). This ensures that visual information from scans is converted into textual data.

**2. Embedding & Storage (The Vector Database):**
The extracted text (from both conversational and document sources) is then vectorized using the **Embedding Model** (`nomic-embed-text`). These vectors are stored in a local **ChromaDB** instance (`chroma_db/`). Crucially, each memory profile maintains an isolated collection, ensuring that one user's knowledge is entirely separate from another's.

**3. Retrieval (The Search):**
When a user asks a question in the chat interface, the system performs a semantic search against the profile's ChromaDB collection. The `llama_index` library's `as_chat_engine` function is used to perform a similarity search (`similarity_top_k=3`), retrieving the top 3 most relevant memory chunks (source nodes) that semantically match the query.

**4. Generation (The Answer):**
The retrieved chunks are injected directly into the prompt context of the **LLM Model** (`hf.co/unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL`). The LLM is then instructed (via the system prompt) to act as a "helpful personal memory assistant," answering the user's question *only* based on the provided context. This tight coupling of retrieval and generation is what defines the RAG pattern, preventing the LLM from hallucinating and grounding the response in the user's stored knowledge.

## Notes

- Each profile has its own collection within ChromaDB.
- `profiles.json` holds the profile metadata.
- Documents are stored locally, so all memory remains on-device unless you change the storage configuration.
- Image imports are supported through the same file or folder path field in the Knowledge Base tab.
- Supported image files are converted to PNG bytes for consistent Vision OCR processing.

## Example Use Cases

- Personal knowledge base for notes, documents, and reference material
- Organization of project documentation by profile
- Long-term memory assistant for ideas, SOPs, and extracted knowledge
- Local-first document Q&A without cloud dependency

## License

This project is free and open-source software, released under the [MIT License](https://opensource.org/licenses/MIT). You are free to use, modify, and distribute it for personal purposes.