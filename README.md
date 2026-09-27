# BitBrain

BitBrain is a local, profile-based memory assistant built with Streamlit, ChromaDB, LlamaIndex, and Ollama. Create isolated knowledge profiles, add memories from chat or local files, and ask questions grounded in each profile's indexed content.

## Features

- Separate ChromaDB collection for each memory profile
- Add facts directly through conversational memory input
- Parse supported local files and folders with AnyDoc
- OCR for images and scanned PDFs using a local Ollama vision model
- Retrieval chat grounded in the active profile's documents
- Follow-up questions with the profile's prior chat messages included as conversation context
- Choose an installed Ollama chat model independently for each profile
- Choose installed chat, vision, and embedding models from Model Settings
- Export a profile conversation as Markdown or Word
- Dark charcoal and olive interface theme

## Supported Documents

- Word: `.docx`, `.doc`
- Excel: `.xlsx`, `.xls`
- PowerPoint: `.pptx`, `.ppt`
- Other documents: `.rtf`, `.odt`, `.ods`, `.odp`, `.epub`, `.csv`, `.pdf`
- Images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.tif`, `.tiff`, `.gif`

Image files are converted to PNG before Ollama Vision OCR. Digital PDFs are parsed natively when possible; scanned PDFs use the configured vision model for OCR.

## Requirements

- Python 3.10 or newer
- pip
- Ollama installed and running locally
- Ollama models pulled locally for the tasks you intend to use

BitBrain populates its model dropdowns from `ollama list`. The default model names in the application are:

- Chat: `hf.co/unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL`
- Vision/OCR: `hf.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q8_0`
- Embeddings: `nomic-embed-text` (Ollama may display this as `nomic-embed-text:latest`)

Install only the models you want to use. For example:

```powershell
ollama pull hf.co/unsloth/gemma-4-E4B-it-GGUF:UD-Q4_K_XL
ollama pull hf.co/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF:Q8_0
ollama pull nomic-embed-text
```

You can select different installed models in the sidebar's **Model Settings**. The chat tab also provides a per-profile chat model selector. Model selections are saved with profile metadata; chat history is retained per profile for the current Streamlit session.

## Installation

From the repository root, create and activate a virtual environment, then install the application dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirement.txt
```

Make sure the Ollama service is running and the required local models are available before using OCR, embeddings, or chat.

## Run

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>.

## Tests

Run the test suite from the repository root:

```powershell
python -m pytest
```

Install `pytest` in the virtual environment if it is not already available.

## Project Structure

```text
bit-brain/
├── .streamlit/
│   └── config.toml       # Streamlit theme colors
├── app.py                # Streamlit application and profile workflows
├── enterprise.css        # Application UI theme
├── ollama_models.py      # Ollama CLI model discovery and tag resolution
├── profile_store.py      # ChromaDB profile cleanup helpers
├── profiles.json         # Saved profile metadata and model selections
├── requirement.txt       # Application dependencies
├── tests/                # Focused unit tests
├── chroma_db/            # Local vector database (ignored by Git)
└── README.md
```

## How Retrieval Chat Works

1. Documents and conversational memories are added to the active profile's ChromaDB collection. The selected embedding model converts text into vectors.
2. For a chat question, BitBrain retrieves the three most relevant chunks from that profile (`similarity_top_k=3`).
3. The retrieved text is provided to the selected profile chat model along with the conversation history from that profile, allowing follow-up questions to refer to previous turns.
4. The response and its retrieved source context are shown in the chat interface.

Each profile uses a separate vector collection and separate chat history. Chat history is kept in the current Streamlit session; it is not persisted across application sessions.

## Configuration and Data

- `profiles.json` stores profile names, descriptions, ChromaDB collection identifiers, and per-profile chat model choices.
- `chroma_db/` stores the local ChromaDB data and is excluded from Git by `.gitignore`.
- `.streamlit/config.toml` configures Streamlit's base theme colors. `enterprise.css` styles the workspace components. Restart Streamlit after changing its configuration file.
- Documents, vectors, and conversations remain local unless the app or its storage configuration is changed.

## Example Uses

- A personal knowledge base for notes and documents
- Separate repositories for projects or subjects
- Local document question-answering without sending content to a hosted model service

## License

This project is free and open source under the [MIT License](https://opensource.org/licenses/MIT).
