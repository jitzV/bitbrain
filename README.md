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

- `hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M`
- `maternion/LightOnOCR-2:1b`
- `nomic-embed-text`

You can pull them with:

```bash
ollama pull hf.co/empero-ai/Qwen3.8-4B-Distill-GGUF:Q4_K_M
ollama pull maternion/LightOnOCR-2:1b
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

## How It Works

1. Create a profile from the sidebar.
2. Choose whether to ingest data as:
   - conversational notes, or
   - local files/folders
3. The system parses documents and stores them in ChromaDB.
4. Image files and scanned PDFs are processed with the local Ollama Vision OCR model.
5. The chat tab retrieves the most relevant memory chunks and uses the LLM to answer your questions.

## Notes

- Each profile has its own collection within ChromaDB.
- `profiles.json` holds the profile metadata.
- Documents are stored locally, so all memory remains on-device unless you change the storage configuration.
- Image imports are supported through the same file or folder path field in the Knowledge Base tab.
- Supported image files are converted to PNG bytes for consistent Vision OCR processing.
- For scanned PDFs, the app renders each page to an image and uses a local vision model for OCR fallback.

## Example Use Cases

- Personal knowledge base for notes, documents, and reference material
- Organization of project documentation by profile
- Long-term memory assistant for ideas, SOPs, and extracted knowledge
- Local-first document Q&A without cloud dependency

## License

This project currently does not include a formal license file. If you intend to share or distribute it publicly, add an appropriate license before release.
